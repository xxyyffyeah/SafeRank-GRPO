#!/usr/bin/env python3
"""
Convert SafeRec JSON dataset to HuggingFace Dataset format.

Usage:
    python scripts/phase2_3_saferec/convert_to_hf_dataset.py \
        --input_path data/phase0_trait_assignment/expanded/sft_test_1k_final.json \
        --output_path downloaded_datasets/processed_datasets/saferec_sft_dataset/test \
        --split_name test
"""

import argparse
import json
from pathlib import Path
from datasets import Dataset


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
    Extract training fields from SafeRec sample.

    Works for both SFT (with completion) and GRPO (without completion).

    Returns dict with:
    - prompt: list of message dicts
    - completion: list of message dicts (SFT only, omitted for GRPO)
    - seen_titles: list of strings
    - groundtruth_with_release_year: list of [title, year] pairs
    - constraints: dict of trait -> bool (for sensitivity evaluation)
    - assigned_trait: str, the assigned user sensitivity trait
    - assignment_reason: str, the reasoning for why this trait was assigned
    """
    result = {
        "prompt": sample["prompt"],
        "seen_titles": sample.get("seen_titles", []),
        "groundtruth_with_release_year": sample.get("groundtruth_with_release_year", []),
        "constraints": sample.get("constraints", {}),
        "assigned_trait": sample.get("assigned_trait", ""),
        "assignment_reason": sample.get("assignment_reason", ""),
    }
    if "completion" in sample:
        result["completion"] = sample["completion"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert SafeRec JSON to HuggingFace Dataset")

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to SafeRec JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for HuggingFace dataset"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="Name of the split (e.g., train, test, validation)"
    )

    args = parser.parse_args()

    print(f"Loading SafeRec data from {args.input_path}...", flush=True)
    samples = load_saferec_data(args.input_path)
    print(f"Loaded {len(samples)} samples", flush=True)

    # Prepare samples for SFT
    print("Preparing samples for SFT...", flush=True)
    sft_samples = [prepare_sample_for_sft(s) for s in samples]

    # Create HuggingFace Dataset
    print("Creating HuggingFace Dataset...", flush=True)
    dataset = Dataset.from_list(sft_samples)

    # Save to disk
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {args.split_name} dataset to {output_path}...", flush=True)
    dataset.save_to_disk(str(output_path))

    print("\n=== Conversion Complete ===")
    print(f"Split: {args.split_name}")
    print(f"Samples: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")
    print(f"\nDataset saved to: {output_path}")
    print(f"\nExample usage:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{output_path}')")


if __name__ == "__main__":
    main()
