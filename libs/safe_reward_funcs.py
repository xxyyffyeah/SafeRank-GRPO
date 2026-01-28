# libs/safe_reward_funcs.py
"""
Safe reward functions for Safe-Rank-GRPO training.

Extends the standard DCG-based reward with per-rank safety penalties:
    r_total(x, y^(k)) = r_rel(x, y^(k)) + r_safe(x, y^(k))

Where:
    - r_rel = DCG contribution (existing log_decay reward)
    - r_safe = -lambda_safe * I(violation) * penalty_safe

The key insight is that Rank-GRPO computes advantages per-rank independently,
so penalizing rank k only affects the tokens generating movie k.
"""

import numpy as np
from functools import wraps
from typing import Dict, List, Optional, Tuple, Any

from libs.utils import process_rec_raw
from libs.metrics_align import _discounts, evaluate_direct_match_aligned
from libs.safety_oracle import SafetyOracle


def safe_reward_func_log_decay(
    completions: List[List[Dict[str, str]]],
    groundtruth_with_release_year: List[List[Tuple[str, int]]],
    seen_titles: List[List[str]],
    constraints: List[Dict[str, bool]],
    rec_num: int,
    gt_catalog: Any,
    safety_oracle: SafetyOracle,
    lambda_safe: float = 1.0,
    penalty_safe: float = 1.0,
    **kwargs,
) -> List[List[float]]:
    """
    Compute per-rank rewards with safety penalties.

    Algorithm:
        For each rank k in [1, rec_num]:
            r_rel[k] = DCG contribution (via log decay)
            r_safe[k] = -lambda_safe * I(movie_k violates constraints) * penalty_safe
            r_total[k] = r_rel[k] + r_safe[k]

    Args:
        completions: List of model completions, each is [{"content": "..."}]
        groundtruth_with_release_year: List of ground truth [(title, year), ...]
        seen_titles: List of seen titles for each sample
        constraints: List of constraint dicts per sample, e.g., {"Anti-gore": True, ...}
        rec_num: Number of recommendations expected (20)
        gt_catalog: Ground truth catalog for hit matching
        safety_oracle: SafetyOracle instance for violation checking
        lambda_safe: Safety penalty weight (default: 1.0)
        penalty_safe: Penalty magnitude per violation (default: 1.0)
        **kwargs: Additional args (title_normalizer, year_tolerance, etc.)

    Returns:
        List of per-rank rewards for each sample in batch.
        Shape: [batch_size, rec_num]
    """
    if not (
        len(completions)
        == len(groundtruth_with_release_year)
        == len(seen_titles)
        == len(constraints)
    ):
        raise ValueError(
            "Batch inputs must have equal lengths: "
            f"{len(completions)=}, "
            f"{len(groundtruth_with_release_year)=}, "
            f"{len(seen_titles)=}, "
            f"{len(constraints)=}"
        )

    title_normalizer = kwargs.get("title_normalizer", None)
    year_tolerance = int(kwargs.get("year_tolerance", 2))
    rec_num = int(rec_num)
    discounts = _discounts(rec_num)

    batch_rewards = []
    for recs, gt_with_year, seen, user_constraints in zip(
        completions, groundtruth_with_release_year, seen_titles, constraints
    ):
        recs_text = recs[0]["content"]
        item = {
            "raw_recs": recs_text,
            "groundtruth_with_release_year": gt_with_year,
            "seen_titles": seen,
        }

        error, item = process_rec_raw(item, "raw_recs", "recs")
        if error:
            batch_rewards.append([0.0] * rec_num)
            continue

        hits = evaluate_direct_match_aligned(
            item=item,
            rec_num=rec_num,
            seen_field="seen_titles",
            rec_field="recs",
            gt_field="groundtruth_with_release_year",
            gt_catalog=gt_catalog,
            title_normalizer=title_normalizer,
            year_tolerance=year_tolerance,
        ).astype(np.float64)

        gains = hits * discounts
        total_dcg = float(gains.sum())
        prefix_excl = np.concatenate(([0.0], np.cumsum(gains)[:-1]))
        rewards_rel = total_dcg - prefix_excl

        parsed_recs: List[Tuple[str, int]] = item.get("recs", [])
        safety_penalties = np.zeros(rec_num, dtype=np.float64)

        for k in range(rec_num):
            if k < len(parsed_recs):
                title, year = parsed_recs[k]
                result = safety_oracle.check_safety(
                    title=title, year=year, constraints=user_constraints
                )
                if not result.is_safe:
                    safety_penalties[k] = -lambda_safe * penalty_safe

        rewards_total = rewards_rel + safety_penalties
        batch_rewards.append(rewards_total.tolist())

    return batch_rewards


def make_safe_reward_func(
    rec_num: int,
    gt_catalog: Any,
    safety_oracle: SafetyOracle,
    lambda_safe: float = 1.0,
    penalty_safe: float = 1.0,
) -> callable:
    """
    Factory function to create a safe reward function with pre-bound parameters.

    This matches the interface expected by RankGRPOTrainer, which calls:
        reward_func(completions, groundtruth_with_release_year, seen_titles, **kwargs)

    For Safe-Rank-GRPO, we also need constraints per sample, which are passed via kwargs.

    Args:
        rec_num: Number of recommendations (20)
        gt_catalog: Ground truth catalog
        safety_oracle: SafetyOracle instance
        lambda_safe: Safety penalty weight
        penalty_safe: Penalty magnitude

    Returns:
        A reward function compatible with RankGRPOTrainer
    """

    @wraps(safe_reward_func_log_decay)
    def wrapped(
        completions: List[List[Dict[str, str]]],
        groundtruth_with_release_year: List[List[Tuple[str, int]]],
        seen_titles: List[List[str]],
        **kwargs,
    ) -> List[List[float]]:
        constraints = kwargs.pop("constraints", None)
        if constraints is None:
            constraints = [{}] * len(completions)

        return safe_reward_func_log_decay(
            completions=completions,
            groundtruth_with_release_year=groundtruth_with_release_year,
            seen_titles=seen_titles,
            constraints=constraints,
            rec_num=rec_num,
            gt_catalog=gt_catalog,
            safety_oracle=safety_oracle,
            lambda_safe=lambda_safe,
            penalty_safe=penalty_safe,
            **kwargs,
        )

    return wrapped


def safe_reward_func_exp_inf(
    completions: List[List[Dict[str, str]]],
    groundtruth_with_release_year: List[List[Tuple[str, int]]],
    seen_titles: List[List[str]],
    constraints: List[Dict[str, bool]],
    rec_num: int,
    gt_catalog: Any,
    safety_oracle: SafetyOracle,
    lambda_safe: float = 1.0,
    penalty_safe: float = 1.0,
    **kwargs,
) -> List[List[float]]:
    """
    Per-rank independent reward with safety penalties.

    Unlike log_decay, this gives each rank an independent hit/miss reward (0 or 1),
    then adds safety penalty for violations.

    r_total[k] = hit[k] + r_safe[k]

    Where:
        hit[k] = 1 if rank k matches ground truth, else 0
        r_safe[k] = -lambda_safe * penalty_safe if violation, else 0
    """
    title_normalizer = kwargs.get("title_normalizer")
    year_tolerance = int(kwargs.get("year_tolerance", 2))
    rec_num = int(rec_num)

    batch_rewards = []
    for recs, gt_with_year, seen, user_constraints in zip(
        completions, groundtruth_with_release_year, seen_titles, constraints
    ):
        recs_text = recs[0]["content"]
        item = {
            "raw_recs": recs_text,
            "groundtruth_with_release_year": gt_with_year,
            "seen_titles": seen,
        }

        error, item = process_rec_raw(item, "raw_recs", "recs")
        if error:
            batch_rewards.append([0.0] * rec_num)
            continue

        hits = evaluate_direct_match_aligned(
            item=item,
            rec_num=rec_num,
            seen_field="seen_titles",
            rec_field="recs",
            gt_field="groundtruth_with_release_year",
            gt_catalog=gt_catalog,
            title_normalizer=title_normalizer,
            year_tolerance=year_tolerance,
        ).astype(np.float64)

        parsed_recs: List[Tuple[str, int]] = item.get("recs", [])
        safety_penalties = np.zeros(rec_num, dtype=np.float64)

        for k in range(rec_num):
            if k < len(parsed_recs):
                title, year = parsed_recs[k]
                result = safety_oracle.check_safety(
                    title=title, year=year, constraints=user_constraints
                )
                if not result.is_safe:
                    safety_penalties[k] = -lambda_safe * penalty_safe

        rewards_total = hits + safety_penalties
        batch_rewards.append(rewards_total.tolist())

    return batch_rewards


def make_safe_reward_func_individual(
    rec_num: int,
    gt_catalog: Any,
    safety_oracle: SafetyOracle,
    lambda_safe: float = 1.0,
    penalty_safe: float = 1.0,
) -> callable:
    """Factory for individual (exp_inf) safe reward function."""

    @wraps(safe_reward_func_exp_inf)
    def wrapped(
        completions: List[List[Dict[str, str]]],
        groundtruth_with_release_year: List[List[Tuple[str, int]]],
        seen_titles: List[List[str]],
        **kwargs,
    ) -> List[List[float]]:
        constraints = kwargs.pop("constraints", None)
        if constraints is None:
            constraints = [{}] * len(completions)

        return safe_reward_func_exp_inf(
            completions=completions,
            groundtruth_with_release_year=groundtruth_with_release_year,
            seen_titles=seen_titles,
            constraints=constraints,
            rec_num=rec_num,
            gt_catalog=gt_catalog,
            safety_oracle=safety_oracle,
            lambda_safe=lambda_safe,
            penalty_safe=penalty_safe,
            **kwargs,
        )

    return wrapped
