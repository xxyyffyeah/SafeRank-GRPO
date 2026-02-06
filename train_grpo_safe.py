import os
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
import argparse

from libs.data import load_catalog
from libs.utils import StepLRSchedulerCallback, load_model_with_lora_sft
from libs.safe_reward_funcs import (
    make_relevance_func,
    make_relevance_func_individual,
    make_count_func,
)
from libs.logs import setup_environment, setup_wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Safe-Rank-GRPO with safety-aware rewards.\n\n"
            "Extends Rank-GRPO with per-rank safety penalties:\n"
            "  r_total(x, y^(k)) = r_rel(x, y^(k)) + r_safe(x, y^(k))\n"
            "where r_safe = -lambda_safe * I(violation) * penalty_safe"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    core = parser.add_argument_group("Core setup")
    core.add_argument(
        "--train_path",
        required=True,
        help="Path to the SafeRec training dataset directory (load_from_disk).",
    )
    core.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name or path (Hugging Face format).",
    )
    core.add_argument(
        "--sft_checkpoint",
        type=int,
        default=1500,
        help="Checkpoint ID for the supervised fine-tuned (SFT) reference policy.",
    )
    core.add_argument(
        "--sft_model_path",
        type=str,
        default=None,
        help="Direct path to SFT model. Overrides --model_name/--sft_checkpoint.",
    )
    core.add_argument(
        "--sft_is_lora",
        action="store_true",
        help="Set if --sft_model_path is a LoRA adapter checkpoint (not a merged model).",
    )
    core.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Base model path when --sft_is_lora is set. If None, uses --model_name.",
    )
    core.add_argument(
        "--reward_func",
        default="log_decay",
        choices=["exp_inf", "log_decay"],
        help="Reward shaping function: exp_inf (per-rank independent) or log_decay (DCG-based).",
    )
    core.add_argument(
        "--catalog_path",
        default="gt_catalog.pkl",
        help="Path to the ground-truth catalog pickle.",
    )

    safety = parser.add_argument_group("Safety parameters")
    safety.add_argument(
        "--lambda_safe",
        type=float,
        default=1.0,
        help="Safety penalty weight (coefficient for safety violations).",
    )
    safety.add_argument(
        "--penalty_safe",
        type=float,
        default=1.0,
        help="Penalty magnitude per violation.",
    )
    safety.add_argument(
        "--risk_threshold",
        type=float,
        default=0.66,
        help="Risk threshold for SafetyOracle (movies with risk >= threshold are unsafe).",
    )
    safety.add_argument(
        "--lambda_count",
        type=float,
        default=0.0,
        help="Count reward weight. When > 0, adds a reward signal encouraging exactly --target_count recs.",
    )
    safety.add_argument(
        "--target_count",
        type=int,
        default=10,
        help="Target number of recommendations for the count reward.",
    )
    safety.add_argument(
        "--reward_weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each reward function [relevance, safety, count]. Default: equal weights.",
    )

    opt = parser.add_argument_group("Optimization hyperparameters")
    opt.add_argument("--lr", type=float, default=1e-6, help="Initial learning rate.")
    opt.add_argument(
        "--adam_beta1", type=float, default=0.9, help="Adam optimizer beta1."
    )
    opt.add_argument(
        "--adam_beta2", type=float, default=0.99, help="Adam optimizer beta2."
    )
    opt.add_argument(
        "--kl_beta", type=float, default=1e-3, help="KL-divergence penalty coefficient."
    )
    opt.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=12,
        help="Gradient accumulation steps.",
    )
    opt.add_argument("--optim", default="paged_adamw_8bit", help="Optimizer type.")

    sched = parser.add_argument_group("Training schedule")
    sched.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Training batch size per GPU.",
    )
    sched.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Evaluation batch size per GPU.",
    )
    sched.add_argument(
        "--num_train_epochs", type=int, default=2, help="Number of training epochs."
    )
    sched.add_argument(
        "--eval_strategy",
        default="no",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy: no, steps, or epoch.",
    )
    sched.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps (only used when eval_strategy=steps).",
    )
    sched.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Path to validation dataset. If not set, no evaluation is performed.",
    )
    sched.add_argument(
        "--max_steps", type=int, default=-1, help="Max training steps (-1 = use num_train_epochs)."
    )
    sched.add_argument(
        "--mu",
        type=int,
        default=1,
        help="Number of GRPO iterations (mu=1 = strictly on-policy).",
    )

    gen = parser.add_argument_group("Model / generation parameters")
    gen.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Maximum prompt token length.",
    )
    gen.add_argument(
        "--max_completion_length",
        type=int,
        default=1024,
        help="Maximum generation token length.",
    )
    gen.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of sampled completions per prompt.",
    )
    gen.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )

    vllm = parser.add_argument_group("vLLM inference backend")
    vllm.add_argument(
        "--use_vllm", action="store_true", help="Use vLLM as the inference backend."
    )
    vllm.add_argument(
        "--vllm_mode",
        default="colocate",
        help="vLLM deployment mode: colocate or separate.",
    )
    vllm.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.5,
        help="GPU memory fraction for vLLM.",
    )
    vllm.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=4,
        help="vLLM tensor parallel size.",
    )

    log = parser.add_argument_group("Logging / checkpointing")
    log.add_argument(
        "--wandb_project",
        default="safe_rank_grpo",
        help="Weights & Biases project name.",
    )
    log.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom wandb run name. If not set, auto-generated from model/seed/timestamp.",
    )
    log.add_argument(
        "--wandb_force_new_run",
        action="store_true",
        help="Force creating a new Weights & Biases run (no resume).",
    )
    log.add_argument(
        "--logging_steps", type=int, default=10, help="Logging frequency in steps."
    )
    log.add_argument(
        "--save_strategy",
        default="steps",
        help="Checkpoint save strategy: steps or epoch.",
    )
    log.add_argument(
        "--save_steps", type=int, default=200, help="Save model every N steps."
    )
    log.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )

    lora = parser.add_argument_group("LoRA / PEFT parameters")
    lora.add_argument(
        "--use_lora", action="store_true", help="Enable LoRA (Low-Rank Adaptation) training.",
    )
    lora.add_argument(
        "--lora_r", type=int, default=16, help="LoRA rank (dimension of low-rank matrices).",
    )
    lora.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA scaling factor (alpha / r = effective scale).",
    )
    lora.add_argument(
        "--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.",
    )
    lora.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Module names to apply LoRA to.",
    )

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--seed", type=int, default=3407, help="Random seed for reproducibility."
    )
    misc.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision.")
    misc.add_argument(
        "--verbose", action="store_true", help="Enable verbose LR scheduler output."
    )

    return parser.parse_args()


def _to_sequence_reward(item_reward_func):
    """Convert per-rank reward outputs [B, rec_num] to scalar sequence rewards [B]."""
    def wrapped(completions, groundtruth_with_release_year, seen_titles, **kwargs):
        per_rank = item_reward_func(
            completions=completions,
            groundtruth_with_release_year=groundtruth_with_release_year,
            seen_titles=seen_titles,
            **kwargs,
        )
        return [float(r[0]) if len(r) > 0 else 0.0 for r in per_rank]
    return wrapped


def main():
    args = parse_args()
    accelerator = setup_environment(args.wandb_project)

    gt_catalog = load_catalog(args.catalog_path)
    train_dataset = load_from_disk(os.path.join(args.train_path, "train"))

    # Load validation dataset if specified
    eval_dataset = None
    if args.val_path:
        eval_dataset = load_from_disk(args.val_path)
        accelerator.print(f"[Eval] Loaded validation dataset: {len(eval_dataset)} samples")

    # Original GRPO setup: sequence-level rewards only (no per-item advantage updates).
    if args.reward_func == "exp_inf":
        relevance_item_func = make_relevance_func_individual(rec_num=20, gt_catalog=gt_catalog)
    elif args.reward_func == "log_decay":
        relevance_item_func = make_relevance_func(rec_num=20, gt_catalog=gt_catalog)
    else:
        raise ValueError(f"{args.reward_func} not implemented!")

    reward_func = [_to_sequence_reward(relevance_item_func)]
    accelerator.print("[GRPO] Relevance reward: sequence-level binary hit.")

    if args.lambda_count > 0:
        count_item_func = make_count_func(
            rec_num=20, target_count=args.target_count, lambda_count=args.lambda_count,
        )
        reward_func.append(_to_sequence_reward(count_item_func))
        accelerator.print(
            f"[GRPO] Count penalty enabled: target={args.target_count}, lambda={args.lambda_count}"
        )

    if args.sft_model_path:
        sft_model_path = args.sft_model_path
    else:
        sft_model_path = f"./results/{args.model_name}/checkpoint-{args.sft_checkpoint}"

    # Handle LoRA SFT checkpoint: load base model and merge the adapter
    model_for_trainer = None
    if args.sft_is_lora:
        base_model_path = args.base_model_path or args.model_name
        accelerator.print(
            f"[LoRA-SFT] Loading base model: {base_model_path}"
        )
        accelerator.print(
            f"[LoRA-SFT] Merging LoRA adapter from: {sft_model_path}"
        )
        model_for_trainer = load_model_with_lora_sft(
            base_model_path=base_model_path,
            lora_adapter_path=sft_model_path,
            torch_dtype="bfloat16" if args.bf16 else "auto",
        )
        accelerator.print("[LoRA-SFT] Adapter merged successfully")
    else:
        # Use the path directly (merged model or HuggingFace model ID)
        model_for_trainer = sft_model_path

    lora_suffix = f"_lora_r{args.lora_r}" if args.use_lora else ""
    output_dir = (
        f"./results/safe_grpo/{args.model_name}"
        f"_grpo{lora_suffix}"
        f"_lr{args.lr}_kl{args.kl_beta}_mu{args.mu}"
        f"_hitreward"
    )
    run_name = setup_wandb(
        accelerator,
        output_dir,
        args.model_name,
        args.sft_checkpoint,
        args.seed,
        args.wandb_project,
        args.run_name,
        args.wandb_force_new_run,
    )

    if "0.5B" in args.model_name:
        schedule = [(8000, 1e-6)]
    else:
        schedule = [(8000, 1e-7)]
    callback = StepLRSchedulerCallback(schedule=schedule, verbose=args.verbose)

    config = GRPOConfig(
        importance_sampling_level="item",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        beta=args.kl_beta,
        epsilon=0.2,
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        num_iterations=args.mu,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        optim=args.optim,
        lr_scheduler_type="constant",
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        seed=args.seed,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        run_name=run_name,
        reward_weights=args.reward_weights,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
    )
    if args.reward_weights:
        accelerator.print(f"[GRPO] Reward weights: {args.reward_weights}")

    # Build LoRA config if enabled
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        accelerator.print(
            f"[LoRA] Enabled: r={args.lora_r}, alpha={args.lora_alpha}, "
            f"dropout={args.lora_dropout}, targets={args.lora_target_modules}"
        )

    trainer = GRPOTrainer(
        model=model_for_trainer,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[callback],
        peft_config=peft_config,
    )

    accelerator.print("Training Safe-GRPO (original GRPO trainer, sequence-level rewards)...")
    trainer.train(resume_from_checkpoint=args.resume)
    accelerator.print("✅ Done")


if __name__ == "__main__":
    main()
