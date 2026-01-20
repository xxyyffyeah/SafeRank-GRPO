#!/usr/bin/env python3
"""
Upload SafeRec Model and Dataset to Hugging Face Hub (Private by default)

Usage:
    # Login first
    huggingface-cli login

    # Upload model (private)
    python scripts/upload_to_huggingface.py --upload_model --repo_id YOUR_USERNAME/saferec-qwen2.5-0.5b

    # Upload dataset (private)
    python scripts/upload_to_huggingface.py --upload_dataset --repo_id YOUR_USERNAME/saferec-dataset

    # Upload both (private)
    python scripts/upload_to_huggingface.py --upload_model --upload_dataset --repo_id YOUR_USERNAME/saferec

    # Upload as public (add --public flag)
    python scripts/upload_to_huggingface.py --upload_model --repo_id YOUR_USERNAME/saferec --public
"""

import os
import argparse
from pathlib import Path


def upload_model(repo_id: str, model_path: str, private: bool = True):
    """Upload trained model to Hugging Face Hub."""
    from huggingface_hub import HfApi, upload_folder

    api = HfApi()

    # Create repo if it doesn't exist
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True
    )
    print(f"📦 Model repo: {repo_url} ({'private' if private else 'public'})")

    # Find the best/latest checkpoint
    checkpoints = sorted([
        d for d in os.listdir(model_path)
        if d.startswith("checkpoint-")
    ], key=lambda x: int(x.split("-")[1]))

    if not checkpoints:
        print("❌ No checkpoints found!")
        return

    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(model_path, latest_checkpoint)
    print(f"📂 Uploading checkpoint: {latest_checkpoint}")

    # Upload the checkpoint folder
    upload_folder(
        folder_path=checkpoint_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload SafeRec SFT model ({latest_checkpoint})"
    )

    print(f"✅ Model uploaded to: https://huggingface.co/{repo_id}")
    return repo_id


def upload_dataset(repo_id: str, dataset_path: str, private: bool = True):
    """Upload dataset to Hugging Face Hub."""
    from huggingface_hub import HfApi
    from datasets import load_from_disk

    api = HfApi()

    # Create dataset repo
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True
    )
    print(f"📦 Dataset repo: {repo_url} ({'private' if private else 'public'})")

    # Load and push dataset
    print(f"📂 Loading dataset from {dataset_path}")

    # Upload train split
    train_path = os.path.join(dataset_path, "train")
    if os.path.exists(train_path):
        train_ds = load_from_disk(train_path)
        print(f"   Train samples: {len(train_ds)}")
        train_ds.push_to_hub(repo_id, split="train", private=private)

    # Upload validation split
    val_path = os.path.join(dataset_path, "validation")
    if os.path.exists(val_path):
        val_ds = load_from_disk(val_path)
        print(f"   Validation samples: {len(val_ds)}")
        val_ds.push_to_hub(repo_id, split="validation", private=private)

    print(f"✅ Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
    return repo_id


def upload_full_data(repo_id: str, data_paths: list, private: bool = True):
    """Upload additional data files (JSON, pickle, etc.)."""
    from huggingface_hub import HfApi

    api = HfApi()

    # Create dataset repo
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True
    )

    for file_path in data_paths:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            print(f"📤 Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"   ✅ {filename} uploaded")

    print(f"✅ Data files uploaded to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload SafeRec to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., 'username/saferec-model')"
    )
    parser.add_argument(
        "--upload_model",
        action="store_true",
        help="Upload the trained model"
    )
    parser.add_argument(
        "--upload_dataset",
        action="store_true",
        help="Upload the dataset"
    )
    parser.add_argument(
        "--upload_raw_data",
        action="store_true",
        help="Upload raw JSON data files"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/Qwen/Qwen2.5-0.5B-Instruct",
        help="Path to model checkpoints"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="downloaded_datasets/processed_datasets/saferec_sft_dataset",
        help="Path to processed dataset"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repo public (default is private)"
    )

    args = parser.parse_args()

    if not args.upload_model and not args.upload_dataset and not args.upload_raw_data:
        print("❌ Please specify at least one of: --upload_model, --upload_dataset, --upload_raw_data")
        return

    # Default to private unless --public is specified
    private = not args.public

    print("="*60)
    print("SafeRec Hugging Face Upload")
    print(f"Mode: {'PUBLIC' if args.public else 'PRIVATE'}")
    print("="*60)

    if args.upload_model:
        print("\n📦 Uploading Model...")
        model_repo = args.repo_id if not args.upload_dataset else f"{args.repo_id}-model"
        upload_model(model_repo, args.model_path, private)

    if args.upload_dataset:
        print("\n📦 Uploading Dataset...")
        dataset_repo = args.repo_id if not args.upload_model else f"{args.repo_id}-dataset"
        upload_dataset(dataset_repo, args.dataset_path, private)

    if args.upload_raw_data:
        print("\n📦 Uploading Raw Data Files...")
        raw_files = [
            "data/phase2_3_saferec/saferec_sft_final.json",
            "gt_catalog.pkl",
            "data/phase0_trait_assignment/saferec_sft_8k_dataset.json",
        ]
        data_repo = args.repo_id if not (args.upload_model or args.upload_dataset) else f"{args.repo_id}-data"
        upload_full_data(data_repo, raw_files, private)

    print("\n" + "="*60)
    print("✅ Upload complete!")
    print("="*60)


if __name__ == "__main__":
    main()
