#!/usr/bin/env python3
"""
Evaluate a single SFT checkpoint on SafeRec validation dataset.

Usage:
    python scripts/phase4_sft_eval/eval_single_checkpoint.py \
        --checkpoint_path results/official_sft/Qwen2.5-0.5B-Instruct/checkpoint-1500 \
        --dataset_path downloaded_datasets/processed_datasets/saferec_sft_dataset \
        --catalog_path gt_catalog.pkl
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from vllm import LLM, SamplingParams
from datasets import load_from_disk

# Add libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../libs"))
from utils import process_rec_raw
from metrics import evaluate_direct_match


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate single SFT checkpoint")
    parser.add_argument("--checkpoint_path", required=True, help="Path to checkpoint")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset")
    parser.add_argument("--catalog_path", required=True, help="Path to gt_catalog.pkl")
    parser.add_argument("--output_path", default=None, help="Path to save results JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Single Checkpoint Evaluation on SafeRec Validation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Catalog: {args.catalog_path}")
    print("=" * 60)

    # Load validation dataset
    print("\n📂 Loading validation dataset...")
    val_dataset = load_from_disk(os.path.join(args.dataset_path, "validation"))
    print(f"   Loaded {len(val_dataset)} samples")

    # Load catalog
    print("\n📂 Loading catalog...")
    with open(args.catalog_path, "rb") as f:
        gt_catalog = set(pickle.load(f))
    print(f"   Loaded {len(gt_catalog)} movies")

    # Group contexts (deduplicate prompts)
    print("\n🔄 Grouping unique contexts...")
    context_to_indices = defaultdict(list)
    for i, item in enumerate(val_dataset):
        context = item["prompt"][0]["content"]
        context_to_indices[context].append(i)
    all_contexts = list(context_to_indices.keys())
    all_input_texts = [[{"role": "user", "content": ctx}] for ctx in all_contexts]
    print(f"   {len(all_input_texts)} unique contexts")

    # Load model and generate
    print(f"\n🧠 Loading model from {args.checkpoint_path}...")
    llm = LLM(
        model=args.checkpoint_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=8192
    )

    sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=1024)
    print("   Generating recommendations...")
    outputs = llm.chat(all_input_texts, sampling_params)
    del llm

    # Assign outputs to dataset
    test_data_with_rec = [dict(item) for item in val_dataset]
    context_to_rec = {ctx: out.outputs[0].text for ctx, out in zip(all_contexts, outputs)}
    for ctx, indices in context_to_indices.items():
        for idx in indices:
            test_data_with_rec[idx]["raw_rec"] = context_to_rec[ctx]

    # Process recommendations
    print("\n🔄 Processing recommendations...")
    for item in test_data_with_rec:
        _, item = process_rec_raw(item, "raw_rec", "rec")

    # Evaluate catalog match ratio
    print("\n📊 Evaluating catalog match ratio...")
    gt_index = defaultdict(list)
    for name, year in gt_catalog:
        gt_index[name.lower()].append(year)

    match_ratios = []
    for item in tqdm(test_data_with_rec, desc="Catalog match"):
        recs = item.get("rec", [])[:20]
        if not recs:
            match_ratios.append(0)
            continue
        matches = 0
        for movie, year in recs:
            years = gt_index.get(movie.lower(), [])
            if any(abs(int(year) - int(y)) <= 2 for y in years):
                matches += 1
        match_ratios.append(matches / len(recs))

    rec_num = np.mean([len(item.get("rec", [])) for item in test_data_with_rec])

    # Evaluate Recall and NDCG
    print("\n📊 Evaluating Recall@K and NDCG@K...")
    k_list = [5, 10, 15, 20]
    results = {
        "checkpoint": args.checkpoint_path,
        "num_samples": len(val_dataset),
        "num_unique_contexts": len(all_contexts),
        "avg_rec_num": rec_num,
        "catalog_match_ratio": np.mean(match_ratios),
        "catalog_match_std": np.std(match_ratios),
    }

    for k in k_list:
        recalls, ndcgs = [], []
        for item in tqdm(test_data_with_rec, desc=f"Top-{k}"):
            r, n = evaluate_direct_match(
                item, k,
                seen_field="seen_titles",
                rec_field="rec",
                gt_field="groundtruth_with_release_year",
                gt_catalog=gt_catalog
            )
            recalls.append(r)
            ndcgs.append(n)
        results[f"recall@{k}"] = np.mean(recalls)
        results[f"ndcg@{k}"] = np.mean(ndcgs)

    # Print results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Samples: {results['num_samples']}")
    print(f"Unique contexts: {results['num_unique_contexts']}")
    print(f"Avg recommendations: {results['avg_rec_num']:.2f}")
    print(f"Catalog match ratio: {results['catalog_match_ratio']:.4f} ± {results['catalog_match_std']:.4f}")
    print("-" * 60)
    print(f"Recall@5:  {results['recall@5']:.4f}")
    print(f"Recall@10: {results['recall@10']:.4f}")
    print(f"Recall@15: {results['recall@15']:.4f}")
    print(f"Recall@20: {results['recall@20']:.4f}")
    print("-" * 60)
    print(f"NDCG@5:  {results['ndcg@5']:.4f}")
    print(f"NDCG@10: {results['ndcg@10']:.4f}")
    print(f"NDCG@15: {results['ndcg@15']:.4f}")
    print(f"NDCG@20: {results['ndcg@20']:.4f}")
    print("=" * 60)

    # Save results
    output_path = args.output_path or os.path.join(
        os.path.dirname(args.checkpoint_path), "eval_saferec_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")

    # Save full output for analysis
    output_pkl = output_path.replace(".json", "_output.pkl")
    with open(output_pkl, "wb") as f:
        pickle.dump(test_data_with_rec, f)
    print(f"💾 Full output saved to: {output_pkl}")


if __name__ == "__main__":
    main()
