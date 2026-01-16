#!/usr/bin/env python3
"""
Convert SafeRec JSON dataset to HuggingFace Dataset format for SFT training.

Usage:
    python scripts/phase2_3_saferec/convert_to_hf_dataset.py \
        --input_path data/phase2_3_saferec/saferec_sft_final.json \
        --output_path downloaded_datasets/processed_datasets/saferec_sft_dataset \
        --train_ratio 0.9
"""

import argparse
import json
from pathlib import Path
from datasets import Dataset
import random


def load_saferec_data(input_path: str) -> list:
    """Load SafeRec JSON data."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected data format in {input_path}")


def prepare_sample_for_sft(sample: dict) -> dict:
    """
    Extract SFT training fields from SafeRec sample.

    Returns dict with:
    - prompt: list of message dicts
    - completion: list of message dicts (with safe recommendations + CoT)
    - seen_titles: list of strings
    - groundtruth_with_release_year: list of [title, year] pairs
    """
    return {
        "prompt": sample["prompt"],
        "completion": sample["completion"],
        "seen_titles": sample.get("seen_titles", []),
        "groundtruth_with_release_year": sample.get("groundtruth_with_release_year", []),
    }


def split_data(samples: list, train_ratio: float = 0.9, seed: int = 42):
    """Split data into train/validation sets."""
    random.seed(seed)

    # Shuffle
    shuffled = samples.copy()
    random.shuffle(shuffled)

    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_samples = shuffled[:split_idx]
    val_samples = shuffled[split_idx:]

    return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser(description="Convert SafeRec JSON to HuggingFace Dataset")

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/phase2_3_saferec/saferec_sft_final.json",
        help="Path to SafeRec JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="downloaded_datasets/processed_datasets/saferec_sft_dataset",
        help="Output directory for HuggingFace dataset"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train/validation split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print(f"Loading SafeRec data from {args.input_path}...", flush=True)
    samples = load_saferec_data(args.input_path)
    print(f"Loaded {len(samples)} samples", flush=True)

    # Prepare samples for SFT
    print("Preparing samples for SFT training...", flush=True)
    sft_samples = [prepare_sample_for_sft(s) for s in samples]

    # Split into train/val
    print(f"Splitting data (train_ratio={args.train_ratio})...", flush=True)
    train_samples, val_samples = split_data(sft_samples, args.train_ratio, args.seed)
    print(f"Train: {len(train_samples)}, Validation: {len(val_samples)}", flush=True)

    # Create HuggingFace Datasets
    print("Creating HuggingFace Datasets...", flush=True)
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    # Save to disk
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving train dataset to {output_path}/train...", flush=True)
    train_dataset.save_to_disk(str(output_path / "train"))

    print(f"Saving validation dataset to {output_path}/validation...", flush=True)
    val_dataset.save_to_disk(str(output_path / "validation"))

    print("\n=== Conversion Complete ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Dataset columns: {train_dataset.column_names}")
    print(f"\nDataset saved to: {output_path}")
    print(f"\nYou can now train with:")
    print(f"  python train_sft_safe.py --dataset_path {output_path}")


if __name__ == "__main__":
    main()
