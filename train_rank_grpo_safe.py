import os
from datasets import load_from_disk
from trl import GRPOConfig, RankGRPOTrainer
from peft import LoraConfig
import argparse

from libs.data import load_catalog
from libs.utils import StepLRSchedulerCallback
from libs.safe_reward_funcs import (
    make_safe_reward_func,
    make_safe_reward_func_individual,
    make_relevance_func,
    make_relevance_func_individual,
    make_safety_func,
    make_safety_func_individual,
)
from libs.safety_oracle import create_oracle
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
    misc.add_argument(
        "--advantage_mode",
        default="grpo",
        choices=["grpo", "gdpo"],
        help="Advantage computation: grpo (combined normalization) or gdpo (per-reward decoupled normalization).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = setup_environment(args.wandb_project)

    gt_catalog = load_catalog(args.catalog_path)
    train_dataset = load_from_disk(os.path.join(args.train_path, "train"))

    accelerator.print(
        f"[SafetyOracle] Initializing with risk_threshold={args.risk_threshold}"
    )
    safety_oracle = create_oracle(base_path=".", risk_threshold=args.risk_threshold)

    if args.advantage_mode == "gdpo":
        # GDPO: two separate reward functions for decoupled normalization
        accelerator.print(f"[GDPO] Using decoupled normalization with reward_func={args.reward_func}")
        if args.reward_func == "exp_inf":
            relevance_func = make_relevance_func_individual(rec_num=20, gt_catalog=gt_catalog)
            safety_func = make_safety_func_individual(
                rec_num=20, gt_catalog=gt_catalog, safety_oracle=safety_oracle,
                lambda_safe=args.lambda_safe, penalty_safe=args.penalty_safe,
            )
        elif args.reward_func == "log_decay":
            relevance_func = make_relevance_func(rec_num=20, gt_catalog=gt_catalog)
            safety_func = make_safety_func(
                rec_num=20, gt_catalog=gt_catalog, safety_oracle=safety_oracle,
                lambda_safe=args.lambda_safe, penalty_safe=args.penalty_safe,
            )
        else:
            raise ValueError(f"{args.reward_func} not implemented!")
        reward_func = [relevance_func, safety_func]
    else:
        # GRPO: single combined reward function
        if args.reward_func == "exp_inf":
            reward_func = make_safe_reward_func_individual(
                rec_num=20,
                gt_catalog=gt_catalog,
                safety_oracle=safety_oracle,
                lambda_safe=args.lambda_safe,
                penalty_safe=args.penalty_safe,
            )
        elif args.reward_func == "log_decay":
            reward_func = make_safe_reward_func(
                rec_num=20,
                gt_catalog=gt_catalog,
                safety_oracle=safety_oracle,
                lambda_safe=args.lambda_safe,
                penalty_safe=args.penalty_safe,
            )
        else:
            raise ValueError(f"{args.reward_func} not implemented!")

    sft_model_path = f"./results/{args.model_name}/checkpoint-{args.sft_checkpoint}"
    lora_suffix = f"_lora_r{args.lora_r}" if args.use_lora else ""
    output_dir = (
        f"./results/safe_grpo/{args.model_name}"
        f"_{args.advantage_mode}{lora_suffix}"
        f"_lr{args.lr}_kl{args.kl_beta}_mu{args.mu}"
        f"_lambda{args.lambda_safe}_penalty{args.penalty_safe}"
    )
    run_name = setup_wandb(
        accelerator,
        output_dir,
        args.model_name,
        args.sft_checkpoint,
        args.seed,
        args.wandb_project,
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
        epsilon=0.06,
        epsilon_high=0.08,
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
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
        run_name=run_name,
    )

    # Pass advantage_mode to trainer via config attribute
    config.advantage_mode = args.advantage_mode

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

    trainer = RankGRPOTrainer(
        model=sft_model_path,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        callbacks=[callback],
        peft_config=peft_config,
    )

    accelerator.print(
        f"Training Safe-Rank-GRPO [mode={args.advantage_mode}] "
        f"(lambda={args.lambda_safe}, penalty={args.penalty_safe})..."
    )
    trainer.train(resume_from_checkpoint=args.resume)
    accelerator.print("✅ Done")


if __name__ == "__main__":
    main()
