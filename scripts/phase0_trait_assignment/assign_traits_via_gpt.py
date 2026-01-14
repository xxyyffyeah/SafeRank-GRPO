#!/usr/bin/env python3
"""
Assign user traits to SFT samples using ChatGPT API.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/assign_traits_via_gpt.py \
        --input_path data/sft_filtered_8k.json \
        --output_path data/sft_with_assigned_traits.json \
        --traits_path traits_warnings.json \
        --model gpt-4o \
        --batch_size 1 \
        --max_samples 100
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import openai


SYSTEM_PROMPT_TEMPLATE = """You are a movie safety expert analyzing user conversations and movie recommendations.

Your task: Identify which user trait sensitivity would be MOST violated by the recommended movies, but NOT conflict with the user's explicit preferences.

Available Traits:
{trait_list}

Guidelines:
1. The assigned trait should NOT conflict with user's explicit preferences in the conversation
2. The recommended movies should be LIKELY to violate this trait (contain sensitive content)
3. Choose the trait with highest violation risk among recommended movies
4. If no clear violations exist, choose "None"
5. Return ONLY the trait name exactly as shown above, nothing else

Example:
User: "I love action movies with intense fight scenes"
Recommendations: "Saw (2004), Hostel (2005), The Human Centipede (2009)"
Output: Anti-gore / squeamish

Reason: User wants action (not conflicting), but recommendations are extreme horror/gore films."""

USER_PROMPT_TEMPLATE = """User Conversation:
{prompt}

Recommended Movies:
{completion}

Which trait is MOST likely to be violated by these recommendations?
Return only the trait name (or "None" if no clear violation)."""


def load_traits(traits_path: str) -> List[str]:
    """Load trait names from traits_warnings.json."""
    with open(traits_path) as f:
        traits_data = json.load(f)
    return [t["trait"] for t in traits_data]


def create_prompt(sample: Dict, traits: List[str]) -> str:
    """Create user prompt for GPT."""
    # Extract prompt content
    prompt_content = sample["prompt"][0]["content"]

    # Extract completion content
    completion_content = sample["completion"][0]["content"]

    # Format trait list
    trait_list = "\n".join([f"- {trait}" for trait in traits])
    trait_list += "\n- None"

    # Create system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(trait_list=trait_list)

    # Create user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        prompt=prompt_content,
        completion=completion_content
    )

    return system_prompt, user_prompt


def call_gpt(system_prompt: str, user_prompt: str,
             model: str, temperature: float, max_retries: int = 3) -> Dict:
    """
    Call OpenAI API with retry logic.

    Returns:
        dict with 'trait', 'success', 'error'
    """
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_completion_tokens=100
            )

            trait = response.choices[0].message.content.strip()

            return {
                "trait": trait,
                "success": True,
                "error": None,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  ⚠️ Attempt {attempt + 1} failed: {error_msg}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return {
                    "trait": "Unknown",
                    "success": False,
                    "error": error_msg,
                    "usage": None
                }

    return {"trait": "Unknown", "success": False, "error": "Max retries exceeded", "usage": None}


def process_samples(samples: List[Dict], traits: List[str],
                   model: str, temperature: float,
                   max_samples: int = None) -> List[Dict]:
    """Process samples and assign traits."""
    total_tokens = 0
    successful = 0
    failed = 0

    samples_to_process = samples[:max_samples] if max_samples else samples

    results = []

    for i, sample in enumerate(tqdm(samples_to_process, desc="Assigning traits")):
        # Create prompts
        system_prompt, user_prompt = create_prompt(sample, traits)

        # Call GPT
        result = call_gpt(system_prompt, user_prompt, model, temperature)

        # Update stats
        if result["success"]:
            successful += 1
            if result["usage"]:
                total_tokens += result["usage"]["total_tokens"]
        else:
            failed += 1

        # Create output sample
        output_sample = {
            **sample,
            "assigned_trait": result["trait"],
            "assignment_success": result["success"],
            "assignment_error": result["error"],
            "gpt_usage": result["usage"]
        }

        results.append(output_sample)

        # Progress update every 50 samples
        if (i + 1) % 50 == 0:
            avg_tokens = total_tokens / successful if successful > 0 else 0
            print(f"\n  Progress: {i+1}/{len(samples_to_process)} | Success: {successful} | Failed: {failed} | Avg tokens: {avg_tokens:.0f}")

    print(f"\n✅ Processing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total tokens: {total_tokens:,}")

    # Estimate cost (gpt-4o pricing: $2.50/1M input, $10/1M output)
    # Rough estimate assuming 80% input tokens
    estimated_cost = (total_tokens * 0.8 * 2.50 / 1_000_000) + \
                     (total_tokens * 0.2 * 10.00 / 1_000_000)
    print(f"  Estimated cost: ${estimated_cost:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Assign traits using ChatGPT API")
    parser.add_argument("--input_path", default="data/sft_filtered_8k.json",
                        help="Path to filtered samples JSON")
    parser.add_argument("--output_path", default="data/sft_with_assigned_traits.json",
                        help="Path to output JSON")
    parser.add_argument("--traits_path", default="traits_warnings.json",
                        help="Path to traits definition JSON")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (currently only 1 supported)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (for testing)")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Load data
    print(f"Loading samples from {args.input_path}...")
    with open(args.input_path) as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"Loaded {len(samples):,} samples")

    # Load traits
    print(f"Loading traits from {args.traits_path}...")
    traits = load_traits(args.traits_path)
    print(f"Loaded {len(traits)} traits")

    # Process
    if args.max_samples:
        print(f"⚠️ Processing only {args.max_samples} samples (test mode)")

    results = process_samples(
        samples, traits, args.model, args.temperature, args.max_samples
    )

    # Save
    output = {
        "samples": results,
        "stats": {
            "total_samples": len(results),
            "successful_assignments": sum(1 for s in results if s["assignment_success"]),
            "failed_assignments": sum(1 for s in results if not s["assignment_success"]),
        },
        "config": {
            "model": args.model,
            "temperature": args.temperature,
        }
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Saved to {output_path}")

    # Print trait distribution
    trait_counts = {}
    for s in results:
        if s["assignment_success"]:
            trait = s["assigned_trait"]
            trait_counts[trait] = trait_counts.get(trait, 0) + 1

    print(f"\n📊 Trait Distribution (top 10):")
    for trait, count in sorted(trait_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {trait}: {count}")


if __name__ == "__main__":
    main()
