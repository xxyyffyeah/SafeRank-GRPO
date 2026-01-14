#!/usr/bin/env python3
"""
Filter groundtruth movies that violate the assigned trait.

Usage:
    python scripts/filter_violating_groundtruth.py \
        --input_path data/sft_with_assigned_traits.json \
        --output_path data/saferec_sft_8k_dataset.json \
        --trait_sensitivity_path downloaded_datasets/movie_trait_sensitivity.json \
        --title_mapping_path data/title_to_imdb.pkl \
        --min_groundtruth_after_filter 1
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


def normalize_title(title: str) -> str:
    """Normalize title for matching."""
    if not title:
        return ""
    title = title.lower()
    title = re.sub(r"[^\w\s']", " ", title)
    title = " ".join(title.split())
    return title


def parse_year(year_str: str) -> int | None:
    """Parse year from string."""
    if not year_str:
        return None
    try:
        if isinstance(year_str, int):
            return year_str
        return int(str(year_str).strip())
    except ValueError:
        return None


def lookup_imdb_id(title: str, year: int, mapping: dict, year_tolerance: int = 2) -> str | None:
    """Lookup imdbId with year tolerance."""
    norm_title = normalize_title(title)

    # Exact match
    if (norm_title, year) in mapping:
        return mapping[(norm_title, year)]

    # Year tolerance
    for d in range(1, year_tolerance + 1):
        for y in [year + d, year - d]:
            if (norm_title, y) in mapping:
                return mapping[(norm_title, y)]

    return None


def check_movie_violates_trait(
    title: str,
    year: int,
    trait_name: str,
    title_mapper: dict,
    trait_data: dict,
    threshold: float = 0.66
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if a movie violates the given trait.

    Args:
        title: Movie title
        year: Release year
        trait_name: Assigned trait
        title_mapper: Title to imdbId mapping
        trait_data: {imdbId: {"traits": {...}}}
        threshold: Unsafe threshold (default 0.66)

    Returns:
        (violates, trait_info)
    """
    # Lookup imdbId
    imdb_id = lookup_imdb_id(title, year, title_mapper)

    if not imdb_id:
        # Unknown movie - assume safe (don't filter)
        return False, None

    if imdb_id not in trait_data:
        # No trait data - assume safe
        return False, None

    # Get trait info
    movie_traits = trait_data[imdb_id]["traits"]

    if trait_name not in movie_traits:
        # Trait not available for this movie
        return False, None

    trait_info = movie_traits[trait_name]

    # Check if unsafe
    is_unsafe = trait_info.get("unsafe", False)
    risk_score = trait_info.get("final", 0.0)

    # Violates if marked as unsafe
    return is_unsafe and risk_score >= threshold, trait_info


def filter_sample_groundtruth(
    sample: Dict,
    title_mapper: dict,
    trait_data: dict,
    min_groundtruth: int = 1
) -> Optional[Dict]:
    """
    Filter groundtruth movies that violate the assigned trait.

    Args:
        sample: Sample with assigned_trait
        title_mapper: Title to imdbId mapping
        trait_data: Trait sensitivity data
        min_groundtruth: Minimum GT to keep after filtering

    Returns:
        Filtered sample or None if insufficient GT remains
    """
    if not sample.get("assignment_success"):
        # If assignment failed, keep original
        return sample

    assigned_trait = sample.get("assigned_trait", "")

    # Skip if no trait or "None" or "Unknown"
    if not assigned_trait or assigned_trait in ["None", "Unknown"]:
        return sample

    original_gt = sample["groundtruth_with_release_year"]
    filtered_gt = []
    removed_gt = []

    for title, year in original_gt:
        year_int = parse_year(year)
        if not year_int:
            # Keep if year parsing fails
            filtered_gt.append([title, year])
            continue

        # Check violation
        violates, trait_info = check_movie_violates_trait(
            title, year_int, assigned_trait, title_mapper, trait_data
        )

        if violates:
            # Remove this GT
            removed_gt.append({
                "title": title,
                "year": year,
                "trait_info": trait_info
            })
        else:
            # Keep
            filtered_gt.append([title, year])

    # Check if enough GT remains
    if len(filtered_gt) < min_groundtruth:
        return None

    # Update sample
    sample["groundtruth_with_release_year"] = filtered_gt
    sample["num_gt_removed"] = len(removed_gt)
    sample["removed_groundtruth"] = removed_gt
    sample["num_groundtruth_after_filter"] = len(filtered_gt)

    return sample


def main():
    parser = argparse.ArgumentParser(description="Filter violating groundtruth movies")
    parser.add_argument("--input_path", default="data/sft_with_assigned_traits.json",
                        help="Path to samples with assigned traits")
    parser.add_argument("--output_path", default="data/saferec_sft_8k_dataset.json",
                        help="Path to output JSON")
    parser.add_argument("--trait_sensitivity_path",
                        default="downloaded_datasets/movie_trait_sensitivity.json",
                        help="Path to trait sensitivity JSON")
    parser.add_argument("--title_mapping_path", default="data/title_to_imdb.pkl",
                        help="Path to title mapping pickle")
    parser.add_argument("--min_groundtruth_after_filter", type=int, default=1,
                        help="Minimum GT to keep sample")
    parser.add_argument("--threshold", type=float, default=0.66,
                        help="Unsafe threshold")
    args = parser.parse_args()

    # Load title mapping
    print(f"Loading title mapping from {args.title_mapping_path}...")
    with open(args.title_mapping_path, "rb") as f:
        title_mapper = pickle.load(f)
    print(f"Loaded {len(title_mapper):,} mappings")

    # Load trait sensitivity
    print(f"Loading trait sensitivity from {args.trait_sensitivity_path}...")
    with open(args.trait_sensitivity_path) as f:
        trait_list = json.load(f)
    trait_data = {m["imdbId"]: m for m in trait_list}
    print(f"Loaded {len(trait_data):,} movies with trait data")

    # Load input samples
    print(f"Loading samples from {args.input_path}...")
    with open(args.input_path) as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"Loaded {len(samples):,} samples")

    # Filter
    filtered_samples = []
    dropped_samples = 0
    total_gt_removed = 0

    for sample in tqdm(samples, desc="Filtering GT"):
        filtered_sample = filter_sample_groundtruth(
            sample, title_mapper, trait_data, args.min_groundtruth_after_filter
        )

        if filtered_sample:
            filtered_samples.append(filtered_sample)
            total_gt_removed += filtered_sample.get("num_gt_removed", 0)
        else:
            dropped_samples += 1

    print(f"\n✅ Filtering complete:")
    print(f"  Original samples: {len(samples):,}")
    print(f"  Kept samples: {len(filtered_samples):,}")
    print(f"  Dropped samples: {dropped_samples:,}")
    print(f"  Total GT movies removed: {total_gt_removed:,}")

    # Calculate stats
    gt_before = sum(len(s["groundtruth_with_release_year"]) + s.get("num_gt_removed", 0)
                    for s in filtered_samples)
    gt_after = sum(len(s["groundtruth_with_release_year"]) for s in filtered_samples)

    avg_gt_before = gt_before / len(filtered_samples) if filtered_samples else 0
    avg_gt_after = gt_after / len(filtered_samples) if filtered_samples else 0

    stats = {
        "original_samples": len(samples),
        "kept_samples": len(filtered_samples),
        "dropped_samples": dropped_samples,
        "retention_rate": len(filtered_samples) / len(samples) * 100 if samples else 0,
        "total_gt_removed": total_gt_removed,
        "avg_gt_before": round(avg_gt_before, 2),
        "avg_gt_after": round(avg_gt_after, 2),
    }

    print(f"\n📊 Statistics:")
    print(f"  Retention rate: {stats['retention_rate']:.1f}%")
    print(f"  Avg GT before: {stats['avg_gt_before']:.2f}")
    print(f"  Avg GT after: {stats['avg_gt_after']:.2f}")

    # Save
    output = {
        "samples": filtered_samples,
        "stats": stats,
        "config": {
            "min_groundtruth_after_filter": args.min_groundtruth_after_filter,
            "threshold": args.threshold
        }
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Saved to {output_path}")


if __name__ == "__main__":
    main()
