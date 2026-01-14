#!/usr/bin/env python3
"""
Analyze trait distribution and generate statistics.

Usage:
    python scripts/analyze_trait_distribution.py \
        --input_path data/saferec_sft_8k_dataset.json \
        --output_dir data/trait_stats/
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict


def analyze_trait_distribution(samples: list) -> dict:
    """Analyze trait distribution across samples."""
    trait_counts = Counter()
    trait_gt_stats = defaultdict(lambda: {"total_gt": 0, "removed_gt": 0})

    for sample in samples:
        if not sample.get("assignment_success"):
            continue

        trait = sample.get("assigned_trait", "Unknown")
        trait_counts[trait] += 1

        # Track GT stats per trait
        num_gt = len(sample.get("groundtruth_with_release_year", []))
        num_removed = sample.get("num_gt_removed", 0)

        trait_gt_stats[trait]["total_gt"] += num_gt
        trait_gt_stats[trait]["removed_gt"] += num_removed

    return dict(trait_counts), dict(trait_gt_stats)


def analyze_gt_filtering(samples: list) -> dict:
    """Analyze groundtruth filtering statistics."""
    total_samples = len(samples)
    samples_with_removal = sum(1 for s in samples if s.get("num_gt_removed", 0) > 0)

    gt_before = sum(len(s.get("groundtruth_with_release_year", [])) + s.get("num_gt_removed", 0)
                    for s in samples)
    gt_after = sum(len(s.get("groundtruth_with_release_year", [])) for s in samples)
    gt_removed = sum(s.get("num_gt_removed", 0) for s in samples)

    return {
        "total_samples": total_samples,
        "samples_with_removal": samples_with_removal,
        "gt_before_filtering": gt_before,
        "gt_after_filtering": gt_after,
        "gt_removed": gt_removed,
        "removal_rate": gt_removed / gt_before * 100 if gt_before > 0 else 0,
        "avg_gt_before": gt_before / total_samples if total_samples > 0 else 0,
        "avg_gt_after": gt_after / total_samples if total_samples > 0 else 0,
    }


def analyze_assignment_success(samples: list) -> dict:
    """Analyze assignment success rate."""
    total = len(samples)
    successful = sum(1 for s in samples if s.get("assignment_success", False))
    failed = total - successful

    return {
        "total_samples": total,
        "successful_assignments": successful,
        "failed_assignments": failed,
        "success_rate": successful / total * 100 if total > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze trait distribution")
    parser.add_argument("--input_path", default="data/saferec_sft_8k_dataset.json",
                        help="Path to SafeRec SFT dataset")
    parser.add_argument("--output_dir", default="data/trait_stats/",
                        help="Output directory for stats")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_path}...")
    with open(args.input_path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    print(f"Loaded {len(samples):,} samples")

    # Analyze
    print("\nAnalyzing...")

    # 1. Trait distribution
    trait_counts, trait_gt_stats = analyze_trait_distribution(samples)

    # 2. GT filtering
    gt_stats = analyze_gt_filtering(samples)

    # 3. Assignment success
    assignment_stats = analyze_assignment_success(samples)

    # Combine stats
    full_stats = {
        "trait_distribution": trait_counts,
        "trait_gt_stats": trait_gt_stats,
        "gt_filtering": gt_stats,
        "assignment": assignment_stats,
    }

    # Print summary
    print("\n" + "=" * 80)
    print("📊 TRAIT ASSIGNMENT ANALYSIS")
    print("=" * 80)

    print("\n1. Assignment Success:")
    print(f"   Total samples: {assignment_stats['total_samples']:,}")
    print(f"   Successful: {assignment_stats['successful_assignments']:,} ({assignment_stats['success_rate']:.1f}%)")
    print(f"   Failed: {assignment_stats['failed_assignments']:,}")

    print("\n2. Groundtruth Filtering:")
    print(f"   Samples with GT removed: {gt_stats['samples_with_removal']:,}")
    print(f"   GT before: {gt_stats['gt_before_filtering']:,} (avg: {gt_stats['avg_gt_before']:.2f}/sample)")
    print(f"   GT after: {gt_stats['gt_after_filtering']:,} (avg: {gt_stats['avg_gt_after']:.2f}/sample)")
    print(f"   GT removed: {gt_stats['gt_removed']:,} ({gt_stats['removal_rate']:.1f}%)")

    print("\n3. Trait Distribution (top 15):")
    for i, (trait, count) in enumerate(sorted(trait_counts.items(), key=lambda x: -x[1])[:15], 1):
        percentage = count / len(samples) * 100 if samples else 0
        avg_removed = trait_gt_stats[trait]["removed_gt"] / count if count > 0 else 0
        print(f"   {i:2}. {trait[:50]:52} {count:5,} ({percentage:5.1f}%) | Avg GT removed: {avg_removed:.2f}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(full_stats, f, indent=2)

    print(f"\n💾 Saved detailed stats to {stats_path}")

    # Save trait distribution for easy access
    dist_path = output_dir / "trait_distribution.json"
    with open(dist_path, "w") as f:
        sorted_dist = dict(sorted(trait_counts.items(), key=lambda x: -x[1]))
        json.dump(sorted_dist, f, indent=2)

    print(f"💾 Saved trait distribution to {dist_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
