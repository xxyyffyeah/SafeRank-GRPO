#!/usr/bin/env python3
"""
Phase 2 & 3: Generate SafeRec Dataset

Takes Phase 0 output and generates SafeRec training data with:
- Constraint injection into prompts
- Safe recommendation filtering
- CoT reasoning generation
- IMDb ID lookup for safe recommendations

Usage:
    python scripts/phase2_3_saferec/generate_saferec_dataset.py \
        --input_path data/phase0_trait_assignment/saferec_sft_8k_dataset.json \
        --output_path data/phase2_3_saferec/saferec_sft_final.json \
        --injection_rate 1.0
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
import random
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs.safety_oracle import SafetyOracle, create_oracle
from libs.constraint_injector import ConstraintInjector, BatchConstraintInjector


# Regex pattern to parse movie titles with year
MOVIE_PATTERN = re.compile(r'^(.+?)\s*\((\d{4})\)$')


def parse_movie_string(movie_str: str) -> Tuple[str, Optional[int]]:
    """Parse 'Title (Year)' format into (title, year)."""
    movie_str = movie_str.strip()
    match = MOVIE_PATTERN.match(movie_str)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return movie_str, None


def extract_movies_from_completion(completion_text: str) -> List[Tuple[str, Optional[int]]]:
    """Extract list of (title, year) from completion text."""
    movies = []
    for line in completion_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        title, year = parse_movie_string(line)
        if title:
            movies.append((title, year))
    return movies


def filter_safe_recommendations(
    movies: List[Tuple[str, Optional[int]]],
    constraints: Dict[str, bool],
    oracle: SafetyOracle,
    threshold: float = 0.66
) -> Tuple[List[Tuple[str, Optional[int], Optional[str]]], List[Dict]]:
    """
    Filter movies based on constraints.

    Returns:
        (safe_movies_with_imdb, filtered_movies_info)
        safe_movies_with_imdb: List of (title, year, imdb_id)
        filtered_movies_info: List of dicts with title, year, violations
    """
    safe_movies = []
    filtered_movies = []

    for title, year in movies:
        result = oracle.check_safety(title, year, constraints, threshold)

        if result.is_safe:
            safe_movies.append((title, year, result.matched_imdb_id))
        else:
            filtered_movies.append({
                "title": title,
                "year": year,
                "violations": result.violations,
                "imdb_id": result.matched_imdb_id
            })

    return safe_movies, filtered_movies


def generate_cot_reasoning(
    constraints: Dict[str, bool],
    filtered_movies: List[Dict],
    safe_count: int
) -> str:
    """Generate Chain-of-Thought reasoning for safety-aware filtering."""
    active_traits = [t for t, v in constraints.items() if v]

    if not active_traits:
        return ""

    parts = []

    # Describe user constraints
    parts.append("Safety Analysis:")
    parts.append(f"User preferences to avoid: {', '.join(active_traits)}")
    parts.append("")

    # Explain filtered movies
    if filtered_movies:
        parts.append("Movies filtered due to safety concerns:")
        for movie in filtered_movies[:5]:  # Limit to 5 examples
            title = movie.get("title", "Unknown")
            year = movie.get("year", "")
            violations = movie.get("violations", [])
            year_str = f" ({year})" if year else ""
            parts.append(f"- {title}{year_str}: {'; '.join(violations)}")
        parts.append("")

    parts.append(f"Safe recommendations selected: {safe_count} movies")
    parts.append("")

    return "\n".join(parts)


def format_safe_recommendations(safe_movies: List[Tuple[str, Optional[int], Optional[str]]]) -> str:
    """Format safe movies into completion text."""
    lines = []
    for title, year, imdb_id in safe_movies:
        if year:
            lines.append(f"{title} ({year})")
        else:
            lines.append(title)
    return "\n".join(lines)


def load_input_data(input_path: str) -> List[Dict]:
    """Load samples from Phase 0 output."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "samples" in data:
        return data["samples"]
    else:
        raise ValueError(f"Unexpected data format in {input_path}")


def process_sample(
    sample: Dict,
    batch_injector: BatchConstraintInjector,
    oracle: SafetyOracle,
    threshold: float = 0.66,
    force_inject: bool = False,
    include_cot: bool = True
) -> Dict:
    """
    Process a single sample to generate SafeRec training data.

    Returns sample with:
    - Original columns preserved
    - Injected prompt
    - Safe recommendations
    - CoT reasoning
    - IMDb IDs for safe recommendations
    """
    processed = sample.copy()

    assigned_trait = sample.get("assigned_trait")
    prompt_messages = sample.get("prompt", [])
    completion = sample.get("completion", [])

    # Extract original completion text
    original_completion = ""
    if completion and len(completion) > 0:
        original_completion = completion[0].get("content", "")

    # Extract movies from original completion
    original_movies = extract_movies_from_completion(original_completion)

    # Inject constraints into prompt
    modified_messages, injection_meta = batch_injector.process_sample(
        prompt_messages,
        assigned_trait=assigned_trait,
        force_inject=force_inject
    )

    # Build constraints dict
    injected_traits = injection_meta.get("traits", [])
    constraints = {t: True for t in injected_traits}

    # Filter safe recommendations
    safe_movies, filtered_movies = filter_safe_recommendations(
        original_movies, constraints, oracle, threshold
    )

    # Generate CoT reasoning
    cot_reasoning = ""
    if include_cot and injected_traits and filtered_movies:
        cot_reasoning = generate_cot_reasoning(
            constraints, filtered_movies, len(safe_movies)
        )

    # Format safe recommendations
    safe_completion_text = format_safe_recommendations(safe_movies)

    # Build new completion with XML structure
    if cot_reasoning:
        # Use XML format: <reasoning>...</reasoning><solution>...</solution>
        new_completion_content = f"<reasoning>\n{cot_reasoning}</reasoning>\n<solution>\n{safe_completion_text}\n</solution>"
    else:
        # No CoT, just wrap solution in XML tags
        new_completion_content = f"<solution>\n{safe_completion_text}\n</solution>"

    # Update sample with new fields
    processed["prompt"] = modified_messages
    processed["completion"] = [{"role": "assistant", "content": new_completion_content}]

    # Add new columns
    processed["has_constraint_injection"] = injection_meta.get("injected", False)
    processed["injected_traits"] = injected_traits
    processed["constraint_text"] = injection_meta.get("constraint_text", "")
    processed["constraints"] = constraints

    # Original recommendations
    processed["original_recommendations"] = [
        {"title": t, "year": y} for t, y in original_movies
    ]
    processed["original_recommendation_count"] = len(original_movies)

    # Safe recommendations with IMDb IDs
    processed["safe_recommendations"] = [
        {"title": t, "year": y, "imdb_id": imdb} for t, y, imdb in safe_movies
    ]
    processed["safe_recommendation_count"] = len(safe_movies)
    processed["safe_recommendation_imdb_ids"] = [
        imdb for _, _, imdb in safe_movies if imdb
    ]

    # Filtered recommendations
    processed["filtered_recommendations"] = filtered_movies
    processed["filtered_recommendation_count"] = len(filtered_movies)

    # CoT
    processed["has_cot"] = bool(cot_reasoning)
    processed["cot_reasoning"] = cot_reasoning

    return processed


def main():
    parser = argparse.ArgumentParser(description="Generate SafeRec SFT dataset")

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/phase0_trait_assignment/saferec_sft_8k_dataset.json",
        help="Path to Phase 0 output"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/phase2_3_saferec/saferec_sft_final.json",
        help="Path for output"
    )
    parser.add_argument(
        "--trait_sensitivity_path",
        type=str,
        default="downloaded_datasets/movie_trait_sensitivity.json",
        help="Path to movie trait sensitivity data"
    )
    parser.add_argument(
        "--title_mapping_path",
        type=str,
        default="data/phase1_mapping/title_to_imdb.pkl",
        help="Path to title-to-imdb mapping"
    )
    parser.add_argument(
        "--injection_rate",
        type=float,
        default=1.0,
        help="Rate of constraint injection (0.0-1.0)"
    )
    parser.add_argument(
        "--risk_threshold",
        type=float,
        default=0.66,
        help="Risk threshold for safety filtering"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no_cot",
        action="store_true",
        help="Disable CoT reasoning generation"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Resolve paths
    def resolve_path(p):
        if not Path(p).is_absolute():
            return str(PROJECT_ROOT / p)
        return p

    input_path = resolve_path(args.input_path)
    output_path = resolve_path(args.output_path)
    trait_sensitivity_path = resolve_path(args.trait_sensitivity_path)
    title_mapping_path = resolve_path(args.title_mapping_path)

    print(f"[SafeRec] Loading input from {input_path}...", flush=True)
    samples = load_input_data(input_path)
    print(f"[SafeRec] Loaded {len(samples)} samples", flush=True)

    print(f"[SafeRec] Initializing SafetyOracle...", flush=True)
    oracle = SafetyOracle(
        trait_sensitivity_path=trait_sensitivity_path,
        title_mapping_path=title_mapping_path,
        risk_threshold=args.risk_threshold
    )

    print(f"[SafeRec] Initializing constraint injector...", flush=True)
    batch_injector = BatchConstraintInjector(
        injection_rate=args.injection_rate,
        min_traits=1,
        max_traits=1,  # Use only the assigned trait
        seed=args.seed
    )

    print(f"[SafeRec] Processing samples...", flush=True)
    processed_samples = []
    stats = {
        "total_samples": len(samples),
        "samples_with_injection": 0,
        "samples_with_cot": 0,
        "total_original_recs": 0,
        "total_safe_recs": 0,
        "total_filtered_recs": 0,
        "trait_distribution": {},
    }

    for i, sample in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(samples)}...", flush=True)

        processed = process_sample(
            sample,
            batch_injector,
            oracle,
            threshold=args.risk_threshold,
            force_inject=(args.injection_rate >= 1.0),
            include_cot=not args.no_cot
        )
        processed_samples.append(processed)

        # Update stats
        if processed.get("has_constraint_injection"):
            stats["samples_with_injection"] += 1
        if processed.get("has_cot"):
            stats["samples_with_cot"] += 1

        stats["total_original_recs"] += processed.get("original_recommendation_count", 0)
        stats["total_safe_recs"] += processed.get("safe_recommendation_count", 0)
        stats["total_filtered_recs"] += processed.get("filtered_recommendation_count", 0)

        for trait in processed.get("injected_traits", []):
            stats["trait_distribution"][trait] = stats["trait_distribution"].get(trait, 0) + 1

    # Calculate rates
    stats["injection_rate_actual"] = stats["samples_with_injection"] / len(samples)
    stats["cot_rate_actual"] = stats["samples_with_cot"] / len(samples)
    stats["avg_original_recs"] = stats["total_original_recs"] / len(samples)
    stats["avg_safe_recs"] = stats["total_safe_recs"] / len(samples)
    stats["avg_filtered_recs"] = stats["total_filtered_recs"] / len(samples)

    # Prepare output
    output_data = {
        "metadata": {
            "source": args.input_path,
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "config": {
                "injection_rate": args.injection_rate,
                "risk_threshold": args.risk_threshold,
                "include_cot": not args.no_cot,
                "seed": args.seed,
            },
            "stats": stats,
        },
        "samples": processed_samples,
    }

    # Save output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[SafeRec] Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n=== Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples with injection: {stats['samples_with_injection']} ({stats['injection_rate_actual']:.1%})")
    print(f"Samples with CoT: {stats['samples_with_cot']} ({stats['cot_rate_actual']:.1%})")
    print(f"Avg original recs: {stats['avg_original_recs']:.1f}")
    print(f"Avg safe recs: {stats['avg_safe_recs']:.1f}")
    print(f"Avg filtered recs: {stats['avg_filtered_recs']:.1f}")
    print(f"\nTrait distribution:")
    for trait, count in sorted(stats["trait_distribution"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {trait}: {count}")


if __name__ == "__main__":
    main()
