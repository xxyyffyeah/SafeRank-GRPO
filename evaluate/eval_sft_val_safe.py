import os
import sys
import json
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from vllm import LLM, SamplingParams
from datasets import load_from_disk

# Local libs
from libs.analyze import find_latest_checkpoint, parse_log_history, plot_losses
from libs.save_eval_state import save_analysis_state
from libs.log_wandb import merge_and_upload

# loading from the global libs for evaluation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../libs"))
from utils import process_rec_raw
from metrics import evaluate_direct_match
from safety_oracle import SafetyOracle, create_oracle


def dcg_at_k(hits, k):
    """Calculate DCG@K (same as in metrics.py)"""
    if len(hits) == 0:
        return 0
    if len(hits) == 1:
        return hits[0]
    k = min(k, len(hits))
    return hits[0] + sum(hits[i] / np.log2(i + 2) for i in range(1, k))


def evaluate_sensitivity(item, k, rec_field, oracle, threshold=0.66):
    """
    Evaluate how many sensitive movies are in recommendations.
    Returns sensitivity DCG@K (higher means more sensitive - bad!)

    Args:
        item: Dataset item with constraints and recommendations
        k: Top-K to evaluate
        rec_field: Field name containing recommendations
        oracle: SafetyOracle instance
        threshold: Risk threshold for determining sensitive movies

    Returns:
        (sensitivity_dcg, sensitive_count, total_count)
    """
    rec_list = item.get(rec_field, [])[:k]

    # Get constraints for this sample
    constraints = item.get('constraints', {})
    if not constraints:
        # No constraints = no sensitivity to evaluate
        return 0.0, 0, len(rec_list)

    # Create hits array: 1 if movie is sensitive, 0 otherwise
    hits = np.zeros(len(rec_list), dtype=int)
    for i, (movie, year) in enumerate(rec_list):
        result = oracle.check_safety(movie, year, constraints, threshold)
        if not result.is_safe:
            hits[i] = 1

    # Calculate DCG for sensitivity (higher = worse)
    sensitivity_dcg = dcg_at_k(hits, k)
    sensitive_count = hits.sum()

    return sensitivity_dcg, sensitive_count, len(rec_list)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT training and outputs.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model_root", default="../results/Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_path", default="../processed_datasets/sft_dataset")
    parser.add_argument("--catalog_path", default="../gt_catalog.pkl")
    parser.add_argument("--trait_sensitivity_path",
                        default="../downloaded_datasets/movie_trait_sensitivity.json",
                        help="Path to movie trait sensitivity data")
    parser.add_argument("--title_mapping_path",
                        default="../data/phase1_mapping/title_to_imdb.pkl",
                        help="Path to title-to-IMDb mapping")
    parser.add_argument("--output_dir", default="figs", help="Directory to save analysis figures.")
    parser.add_argument("--wandb_project", default="sft_eval_val", help="Weights & Biases project name.")
    parser.add_argument("--use_multiprocessing", action="store_true", help="Use multiprocessing for evaluation.")
    parser.add_argument("--upload_wandb", action="store_true", help="Upload results to Weights & Biases.")
    parser.add_argument("--eval_last_only", action="store_true",
                        help="Only evaluate the last checkpoint instead of every 200 steps")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"],
                        help="Which split to evaluate (default: test)")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.dataset_path

    print(f"📂 Loading {args.split} dataset from {data_root} ...")
    val_dataset = load_from_disk(os.path.join(data_root, args.split))

    # --- Step 1: Load catalog ---
    with open(args.catalog_path, "rb") as f:
        gt_catalog = set(pickle.load(f))
    print(f"Loaded catalog of size: {len(gt_catalog)}")

    # --- Step 1.5: Initialize SafetyOracle for sensitivity evaluation ---
    print(f"🔒 Initializing SafetyOracle ...")
    oracle = SafetyOracle(
        trait_sensitivity_path=args.trait_sensitivity_path,
        title_mapping_path=args.title_mapping_path,
        risk_threshold=0.66
    )

    # --- Step 2: Group contexts ---
    context_to_indices = defaultdict(list)
    for i, item in enumerate(val_dataset):
        context = item["prompt"][0]["content"]
        context_to_indices[context].append(i)
    all_contexts = list(context_to_indices.keys())
    all_input_texts = [[{"role": "user", "content": ctx}] for ctx in all_contexts]
    print(f"Total unique contexts: {len(all_input_texts)}")

    # --- Step 3: Find latest checkpoint & parse logs ---
    model_root = args.model_root
    latest_checkpoint_path = find_latest_checkpoint(model_root)
    print(f"Latest checkpoint: {latest_checkpoint_path}")
    trainer_state_path = os.path.join(latest_checkpoint_path, "trainer_state.json")

    print("Parsing training log history ...")
    train_steps, train_losses, eval_steps, eval_losses = parse_log_history(trainer_state_path)
    plot_losses(train_steps, train_losses, eval_steps, eval_losses, args.model_name, args.output_dir)

    last_step = int(latest_checkpoint_path.split("-")[-1])

    # Determine which checkpoints to evaluate
    if args.eval_last_only:
        step_list = [last_step]
        print(f"📌 Evaluating only the last checkpoint: {last_step}")
    else:
        step_list = list(range(0, last_step, 200))
        print(f"📌 Evaluating checkpoints every 200 steps: {step_list}")

    llm_outputs = {}
    test_data_with_rec = [item for item in val_dataset]

    # --- Step 4: Model inference ---
    print("🧠 Generating recommendations across checkpoints ...")
    model_path_tmpl = os.path.join(model_root, "checkpoint-{}")

    for step in step_list:
        if step in llm_outputs:
            continue
        print(f"Processing step {step} ...")
        model_to_load = args.model_name if step == 0 else model_path_tmpl.format(step)
        llm = LLM(model=model_to_load,
                  tensor_parallel_size=1,
                  gpu_memory_utilization=0.8,
                  max_model_len=8192)
        sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=1024)
        llm_outputs[step] = llm.chat(all_input_texts, sampling_params)
        del llm

    # --- Step 5: Assign outputs to dataset ---
    for step in step_list:
        field = f"raw_rec_after_step_{step}"
        context_to_rec = {ctx: out.outputs[0].text for ctx, out in zip(all_contexts, llm_outputs[step])}
        for ctx, indices in context_to_indices.items():
            for idx in indices:
                test_data_with_rec[idx][field] = context_to_rec[ctx]

    # --- Step 6: Postprocess & compute hit ratios ---
    print("🔍 Evaluating catalog match ratios ...")
    ratios = []
    gt_index = defaultdict(list)
    for name, year in gt_catalog:
        gt_index[name.lower()].append(year)

    for step in step_list:
        field = f"raw_rec_after_step_{step}"
        test_data_with_rec = [process_rec_raw(item, field, f"rec_after_step_{step}")[1] for item in test_data_with_rec]
        rec_field = f"rec_after_step_{step}"

        step_ratios = []
        for item in tqdm(test_data_with_rec, desc=f"Evaluating step {step}"):
            recs = item.get(rec_field, [])[:20]
            matches = 0
            for movie, year in recs:
                years = gt_index.get(movie.lower(), [])
                if any(abs(int(year) - int(y)) <= 2 for y in years):
                    matches += 1
            step_ratios.append(matches / len(recs) if recs else 0)
        ratios.append((step, np.mean(step_ratios), np.std(step_ratios)))

    # --- Step 7: Recall / NDCG evaluation ---
    print("📊 Calculating Recall and NDCG ...")
    k_list = [5, 10, 15, 20]
    metrics, avg_metrics = {}, {}

    for step in step_list:
        rec_field = f"rec_after_step_{step}"
        recalls, ndcgs = {}, {}
        for k in k_list:
            recall_k, ndcg_k = [], []
            for item in tqdm(test_data_with_rec, desc=f"Step {step} Top-{k}"):
                r, n = evaluate_direct_match(
                    item, k,
                    seen_field="seen_titles",
                    rec_field=rec_field,
                    gt_field="groundtruth_with_release_year",
                    gt_catalog=gt_catalog
                )
                recall_k.append(r)
                ndcg_k.append(n)
            recalls[k] = recall_k
            ndcgs[k] = ndcg_k
        metrics[step] = (recalls, ndcgs)
        avg_metrics[step] = (
            {k: np.mean(recalls[k]) for k in k_list},
            {k: np.mean(ndcgs[k]) for k in k_list}
        )

    # --- Step 7.5: Sensitivity evaluation ---
    print("🔒 Calculating Sensitivity metrics (lower is better) ...")
    sensitivity_dcg_metrics = {}
    sensitivity_count_metrics = {}
    sensitivity_ratio_metrics = {}

    for step in step_list:
        rec_field = f"rec_after_step_{step}"
        dcgs = {}
        counts = {}
        ratio_dict = {}

        for k in k_list:
            sensitivity_dcg_list = []
            sensitive_count_list = []
            total_count_list = []

            for item in tqdm(test_data_with_rec, desc=f"Step {step} Sensitivity@{k}"):
                dcg, sens_count, total = evaluate_sensitivity(
                    item, k, rec_field, oracle, threshold=0.66
                )
                sensitivity_dcg_list.append(dcg)
                sensitive_count_list.append(sens_count)
                total_count_list.append(total)

            # Store separate metrics
            dcgs[k] = sensitivity_dcg_list
            counts[k] = sensitive_count_list

            # Calculate ratio for each sample
            ratio_list = [
                count / total if total > 0 else 0.0
                for count, total in zip(sensitive_count_list, total_count_list)
            ]
            ratio_dict[k] = ratio_list

        sensitivity_dcg_metrics[step] = dcgs
        sensitivity_count_metrics[step] = counts
        sensitivity_ratio_metrics[step] = ratio_dict

        # Print average sensitivity for this step
        avg_dcg = np.mean(dcgs[20])
        avg_count = np.mean(counts[20])
        avg_ratio = np.mean(ratio_dict[20])
        print(f"  Step {step} - Sensitivity DCG@20: {avg_dcg:.4f}, "
              f"Count: {avg_count:.2f}, Ratio: {avg_ratio*100:.1f}%")

    # --- Step 8: Save results ---
    print("💾 Saving evaluation results ...")
    analysis_json = save_analysis_state(model_root, ratios, avg_metrics)

    # Add sensitivity metrics to analysis_state.json
    with open(analysis_json, 'r') as f:
        analysis_data = json.load(f)

    for entry in analysis_data['log_history']:
        step = entry['step']

        # Add Sensitivity DCG metrics
        if step in sensitivity_dcg_metrics:
            for k in k_list:
                entry[f"eval_sensitivity_dcg@{k}"] = float(np.mean(sensitivity_dcg_metrics[step][k]))

        # Add Sensitivity Count metrics
        if step in sensitivity_count_metrics:
            for k in k_list:
                entry[f"eval_sensitive_count@{k}"] = float(np.mean(sensitivity_count_metrics[step][k]))

        # Add Sensitivity Ratio metrics
        if step in sensitivity_ratio_metrics:
            for k in k_list:
                entry[f"eval_sensitive_ratio@{k}"] = float(np.mean(sensitivity_ratio_metrics[step][k]))

    with open(analysis_json, 'w') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Updated analysis state with sensitivity metrics → {analysis_json}")

    with open(os.path.join(model_root, "output.pkl"), "wb") as f:
        pickle.dump(test_data_with_rec, f)

    if args.upload_wandb:
        merge_and_upload(
            model_dir=model_root,
            project=args.wandb_project,
            run_name_suffix="sft_eval",
            merged_filename="trainer_plus_analysis.json",
            upload=True
        )
    print(f"✅ Evaluation complete. Results saved to {model_root}")

if __name__ == "__main__":
    main()
