#!/usr/bin/env python3
"""
Filter SFT samples with high-quality groundtruth (GT >= min_groundtruth).

Usage:
    python scripts/filter_sft_samples.py \
        --input_path downloaded_datasets/processed_datasets/sft_dataset/train \
        --output_path data/sft_filtered_8k.json \
        --min_groundtruth 3 \
        --target_samples 8000
"""

import argparse
import json
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm


def filter_samples(dataset, min_groundtruth: int, target_samples: int):
    """
    Filter samples with sufficient groundtruth.

    Args:
        dataset: HuggingFace dataset
        min_groundtruth: Minimum number of GT movies required
        target_samples: Target number of samples to extract

    Returns:
        List of filtered samples
    """
    filtered = []
    gt_distribution = {}

    print(f"Filtering samples with GT >= {min_groundtruth}...")

    for i, sample in enumerate(tqdm(dataset, desc="Scanning")):
        num_gt = len(sample["groundtruth_with_release_year"])

        # Track distribution
        gt_distribution[num_gt] = gt_distribution.get(num_gt, 0) + 1

        # Filter by min_groundtruth
        if num_gt >= min_groundtruth:
            filtered_sample = {
                "sample_id": f"train_{i}",
                "prompt": sample["prompt"],
                "completion": sample["completion"],
                "seen_titles": sample["seen_titles"],
                "groundtruth_with_release_year": sample["groundtruth_with_release_year"],
                "num_groundtruth": num_gt
            }
            filtered.append(filtered_sample)

            # Stop if we have enough
            if len(filtered) >= target_samples:
                break

    print(f"\n✅ Filtered {len(filtered):,} samples")
    print(f"\n📊 GT Distribution (in scanned samples):")
    for num_gt in sorted(gt_distribution.keys())[:15]:
        print(f"  {num_gt} GT: {gt_distribution[num_gt]:,} samples")

    return filtered, gt_distribution


def main():
    parser = argparse.ArgumentParser(description="Filter SFT samples by groundtruth count")
    parser.add_argument("--input_path",
                        default="downloaded_datasets/processed_datasets/sft_dataset/train",
                        help="Path to input SFT dataset")
    parser.add_argument("--output_path",
                        default="data/sft_filtered_8k.json",
                        help="Path to output JSON file")
    parser.add_argument("--min_groundtruth", type=int, default=3,
                        help="Minimum number of groundtruth movies")
    parser.add_argument("--target_samples", type=int, default=8000,
                        help="Target number of samples to extract")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input_path}...")
    dataset = load_from_disk(args.input_path)
    print(f"Total samples: {len(dataset):,}")

    # Filter
    filtered_samples, gt_dist = filter_samples(
        dataset, args.min_groundtruth, args.target_samples
    )

    # Calculate stats
    gt_counts = [s["num_groundtruth"] for s in filtered_samples]
    avg_gt = sum(gt_counts) / len(gt_counts) if gt_counts else 0

    stats = {
        "total_samples": len(filtered_samples),
        "min_groundtruth": args.min_groundtruth,
        "avg_groundtruth": round(avg_gt, 2),
        "max_groundtruth": max(gt_counts) if gt_counts else 0,
        "min_groundtruth_actual": min(gt_counts) if gt_counts else 0,
    }

    # Prepare output
    output = {
        "samples": filtered_samples,
        "stats": stats
    }

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Saved to {output_path}")
    print(f"\n📊 Statistics:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Avg GT per sample: {stats['avg_groundtruth']:.2f}")
    print(f"  GT range: {stats['min_groundtruth_actual']} - {stats['max_groundtruth']}")


if __name__ == "__main__":
    main()
