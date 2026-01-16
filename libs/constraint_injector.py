"""
Constraint Injector Module - Phase 3

Injects user safety constraints into prompts for SafeRec SFT training.
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class InjectionResult:
    """Result of constraint injection."""
    injected_prompt: str
    original_prompt: str
    injected_traits: List[str]
    constraint_text: str
    injection_position: str  # "prefix", "suffix", or "inline"


# English constraint templates for each trait
TRAIT_CONSTRAINT_TEMPLATES = {
    "Animal lover (avoid animal harm/death)": [
        "I love animals and don't want to see any animal harm or death scenes. ",
        "Please avoid movies with animal abuse or animal deaths. ",
        "I'm sensitive to content showing animals being hurt. ",
        "No movies where animals die or get hurt please. ",
    ],
    "Arachnophobia / reptile phobia": [
        "I have arachnophobia, please avoid movies with spiders or snakes. ",
        "Don't recommend movies featuring spiders or reptiles. ",
        "I'm afraid of bugs and creepy crawlies, please avoid such content. ",
        "No movies with spiders, snakes, or insects please. ",
    ],
    "Anti-gore / squeamish": [
        "I don't like gory or bloody content. ",
        "Please avoid movies with excessive blood and gore. ",
        "I'm squeamish, so nothing too graphic please. ",
        "No bloody or violent movies please. ",
        "I can't handle gore, please recommend something mild. ",
    ],
    "Avoid torture & extreme violence": [
        "No movies with torture scenes please. ",
        "I want to avoid extreme violence content. ",
        "Please don't recommend movies with brutal violence or torture. ",
        "I'm sensitive to extreme violence, please avoid such content. ",
    ],
    "Sexual violence sensitive": [
        "Please avoid movies with sexual assault content. ",
        "I'm sensitive to sexual violence depictions. ",
        "No movies with rape scenes please. ",
        "Please don't recommend movies with sexual violence. ",
    ],
    "Kid-safety / child harm sensitive": [
        "Something suitable to watch with kids please. ",
        "Family-friendly movies only. ",
        "Please avoid any content harmful to children. ",
        "Looking for something the whole family can watch. ",
        "I need kid-safe recommendations. ",
    ],
    "Domestic abuse / stalking sensitive": [
        "Please avoid movies depicting domestic violence. ",
        "No movies with stalking please. ",
        "I'm sensitive to abuse-related content. ",
        "Please don't recommend movies with domestic abuse themes. ",
    ],
    "Substance recovery / avoid drugs & alcohol": [
        "I'm in recovery, please avoid movies glamorizing alcohol. ",
        "No movies with heavy drug use please. ",
        "Please avoid content with substance abuse. ",
        "I don't want movies that feature addiction or drug use. ",
    ],
    "Self-harm & suicide sensitive": [
        "Please avoid movies with suicide scenes. ",
        "I'm sensitive to self-harm content. ",
        "No movies depicting suicide please. ",
        "Please don't recommend movies with self-harm themes. ",
    ],
    "Mental health portrayal sensitive": [
        "Please recommend movies with positive mental health portrayals. ",
        "I'm sensitive to how mental illness is depicted. ",
        "Avoid movies with negative mental health stereotypes. ",
        "Please don't recommend movies that stigmatize mental illness. ",
    ],
    "Medical/health trauma avoider": [
        "I'm sensitive to medical scenes, please avoid. ",
        "No movies with lots of hospital scenes. ",
        "Please avoid content about serious illnesses. ",
        "I don't want to see movies with heavy medical themes. ",
    ],
    "Needle/medical procedure phobia": [
        "I'm afraid of needles, please avoid injection scenes. ",
        "No movies with medical procedure close-ups. ",
        "Please avoid content showing syringes or needles. ",
        "I have a phobia of needles, please keep that in mind. ",
    ],
    "Claustrophobia / breathing distress sensitive": [
        "I have claustrophobia, avoid tight space scenes. ",
        "No movies with suffocation scenes please. ",
        "I'm uncomfortable with enclosed space scenarios. ",
        "Please avoid movies with claustrophobic or suffocating scenes. ",
    ],
    "Photosensitivity & motion sickness": [
        "I'm sensitive to flashing lights, please avoid. ",
        "No movies with heavy shaky cam. ",
        "Please avoid films with lots of jump scares and flashing. ",
        "I get motion sickness easily, please recommend stable footage. ",
    ],
    "Hate speech / slur-sensitive": [
        "Please avoid movies with hate speech. ",
        "No movies with discriminatory language. ",
        "I'm sensitive to racial slurs and hateful content. ",
        "Please don't recommend movies with offensive slurs. ",
    ],
    "Gender/LGBTQ respect sensitive": [
        "Please recommend LGBTQ+ friendly movies. ",
        "I want films with positive representation. ",
        "Avoid movies with homophobic content. ",
        "Please don't recommend movies that mock gender identity. ",
    ],
    "Pregnancy/infant-loss sensitive": [
        "I'm sensitive to pregnancy-related trauma content. ",
        "Please avoid movies with miscarriage scenes. ",
        "No movies with infant death please. ",
        "I don't want to see pregnancy loss content. ",
    ],
    "Disaster/accident avoider": [
        "I don't want to watch disaster movies. ",
        "Please avoid movies with car crashes. ",
        "No plane crash scenes please. ",
        "I'm sensitive to accident scenes, please avoid. ",
    ],
    "Happy-ending preference": [
        "I want movies with happy endings. ",
        "Please recommend feel-good movies. ",
        "No sad endings please. ",
        "I need something uplifting with a positive ending. ",
        "Only movies with happy endings please. ",
    ],
    "Horror avoider (avoids scares & supernatural)": [
        "I don't watch horror movies. ",
        "No scary scenes please. ",
        "I'm easily scared, nothing with ghosts or supernatural. ",
        "Please avoid horror and thriller genres. ",
        "No jumpscares or scary movies please. ",
    ],
}


class ConstraintInjector:
    """
    Injects safety constraints into prompts for SafeRec training.

    Usage:
        injector = ConstraintInjector()

        # Inject single constraint
        result = injector.inject(
            prompt="Recommend some good action movies",
            traits=["Anti-gore / squeamish"]
        )
        print(result.injected_prompt)

        # Inject multiple constraints
        result = injector.inject(
            prompt="Recommend some movies for the weekend",
            traits=["Anti-gore / squeamish", "Horror avoider (avoids scares & supernatural)"]
        )
    """

    def __init__(
        self,
        injection_position: str = "prefix",
        seed: Optional[int] = None
    ):
        """
        Initialize ConstraintInjector.

        Args:
            injection_position: "prefix" (before prompt), "suffix" (after), or "inline" (random)
            seed: Random seed for reproducibility
        """
        self.injection_position = injection_position
        self.templates = TRAIT_CONSTRAINT_TEMPLATES

        if seed is not None:
            random.seed(seed)

    def get_constraint_text(self, trait: str) -> str:
        """Get a random constraint phrase for a trait."""
        templates = self.templates.get(trait, [])
        if not templates:
            # Fallback: generate generic constraint
            return f"Please avoid content related to {trait}. "
        return random.choice(templates)

    def inject(
        self,
        prompt: str,
        traits: List[str],
        position: Optional[str] = None
    ) -> InjectionResult:
        """
        Inject constraint text into a prompt.

        Args:
            prompt: Original prompt text
            traits: List of traits to inject constraints for
            position: Override injection position ("prefix", "suffix", "inline")

        Returns:
            InjectionResult with injected prompt and metadata
        """
        if not traits:
            return InjectionResult(
                injected_prompt=prompt,
                original_prompt=prompt,
                injected_traits=[],
                constraint_text="",
                injection_position="none"
            )

        position = position or self.injection_position

        # Build constraint text
        constraint_parts = [self.get_constraint_text(t) for t in traits]
        constraint_text = "".join(constraint_parts)

        # Inject based on position
        if position == "prefix":
            injected_prompt = constraint_text + prompt
        elif position == "suffix":
            injected_prompt = prompt + " " + constraint_text
        elif position == "inline":
            # For inline, randomly choose prefix or suffix
            if random.random() < 0.5:
                injected_prompt = constraint_text + prompt
                position = "prefix"
            else:
                injected_prompt = prompt + " " + constraint_text
                position = "suffix"
        else:
            injected_prompt = constraint_text + prompt
            position = "prefix"

        return InjectionResult(
            injected_prompt=injected_prompt,
            original_prompt=prompt,
            injected_traits=traits,
            constraint_text=constraint_text,
            injection_position=position
        )

    def inject_conversation(
        self,
        messages: List[Dict[str, str]],
        traits: List[str],
        target_role: str = "user"
    ) -> Tuple[List[Dict[str, str]], InjectionResult]:
        """
        Inject constraints into a conversation format (list of messages).

        Finds the first message with target_role and injects constraints.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            traits: Traits to inject
            target_role: Role to inject into (default: "user")

        Returns:
            (modified_messages, InjectionResult)
        """
        if not traits:
            return messages, InjectionResult(
                injected_prompt="",
                original_prompt="",
                injected_traits=[],
                constraint_text="",
                injection_position="none"
            )

        modified_messages = []
        injection_done = False
        result = None

        for msg in messages:
            if msg.get("role") == target_role and not injection_done:
                original_content = msg.get("content", "")
                result = self.inject(original_content, traits)
                modified_messages.append({
                    "role": msg["role"],
                    "content": result.injected_prompt
                })
                injection_done = True
            else:
                modified_messages.append(msg.copy())

        if result is None:
            result = InjectionResult(
                injected_prompt="",
                original_prompt="",
                injected_traits=[],
                constraint_text="",
                injection_position="none"
            )

        return modified_messages, result

    def get_all_traits(self) -> List[str]:
        """Get list of all supported traits."""
        return list(self.templates.keys())


class BatchConstraintInjector:
    """
    Batch constraint injection for dataset generation.

    Handles injection rate, trait selection, and batching.
    """

    def __init__(
        self,
        injection_rate: float = 0.35,
        min_traits: int = 1,
        max_traits: int = 3,
        seed: Optional[int] = None
    ):
        """
        Initialize BatchConstraintInjector.

        Args:
            injection_rate: Fraction of samples to inject (0.0-1.0)
            min_traits: Minimum number of traits per injection
            max_traits: Maximum number of traits per injection
            seed: Random seed
        """
        self.injection_rate = injection_rate
        self.min_traits = min_traits
        self.max_traits = max_traits

        if seed is not None:
            random.seed(seed)

        self.injector = ConstraintInjector(seed=seed)
        self._all_traits = self.injector.get_all_traits()

    def should_inject(self) -> bool:
        """Decide whether to inject constraints for a sample."""
        return random.random() < self.injection_rate

    def select_traits(
        self,
        assigned_trait: Optional[str] = None,
        n_traits: Optional[int] = None
    ) -> List[str]:
        """
        Select traits for injection.

        If assigned_trait is provided, always include it.
        Otherwise, randomly select from all traits.

        Args:
            assigned_trait: Pre-assigned trait to always include
            n_traits: Number of traits to select (random if None)

        Returns:
            List of selected traits
        """
        if n_traits is None:
            n_traits = random.randint(self.min_traits, self.max_traits)

        traits = []

        # Always include assigned trait if provided
        if assigned_trait and assigned_trait in self._all_traits:
            traits.append(assigned_trait)
            n_traits -= 1

        # Fill remaining slots with random traits
        available = [t for t in self._all_traits if t not in traits]
        if n_traits > 0 and available:
            additional = random.sample(available, min(n_traits, len(available)))
            traits.extend(additional)

        return traits

    def process_sample(
        self,
        messages: List[Dict[str, str]],
        assigned_trait: Optional[str] = None,
        force_inject: bool = False
    ) -> Tuple[List[Dict[str, str]], Dict]:
        """
        Process a single sample, potentially injecting constraints.

        Args:
            messages: Conversation messages
            assigned_trait: Pre-assigned trait for this sample
            force_inject: Force injection regardless of rate

        Returns:
            (processed_messages, metadata_dict)
        """
        metadata = {
            "injected": False,
            "traits": [],
            "constraint_text": "",
            "injection_position": "none"
        }

        # Decide whether to inject
        if not force_inject and not self.should_inject():
            return messages, metadata

        # Select traits
        traits = self.select_traits(assigned_trait)

        if not traits:
            return messages, metadata

        # Inject constraints
        modified_messages, result = self.injector.inject_conversation(messages, traits)

        metadata = {
            "injected": True,
            "traits": result.injected_traits,
            "constraint_text": result.constraint_text,
            "injection_position": result.injection_position
        }

        return modified_messages, metadata


if __name__ == "__main__":
    print("=== Constraint Injector Test ===\n")

    # Test single injection
    print("--- Test 1: Single trait injection ---")
    injector = ConstraintInjector(seed=42)

    result = injector.inject(
        prompt="Recommend some good action movies",
        traits=["Anti-gore / squeamish"]
    )
    print(f"Original: {result.original_prompt}")
    print(f"Injected: {result.injected_prompt}")
    print(f"Traits: {result.injected_traits}")
    print()

    # Test multiple traits
    print("--- Test 2: Multiple traits injection ---")
    result = injector.inject(
        prompt="Recommend some movies for the weekend",
        traits=["Anti-gore / squeamish", "Horror avoider (avoids scares & supernatural)"]
    )
    print(f"Original: {result.original_prompt}")
    print(f"Injected: {result.injected_prompt}")
    print()

    # Test conversation injection
    print("--- Test 3: Conversation injection ---")
    messages = [
        {"role": "user", "content": "I want a relaxing movie for the weekend"},
        {"role": "assistant", "content": "Sure, let me recommend some..."}
    ]

    modified, result = injector.inject_conversation(
        messages,
        traits=["Happy-ending preference"]
    )
    print(f"Original user message: {messages[0]['content']}")
    print(f"Modified user message: {modified[0]['content']}")
    print()

    # Test batch injector
    print("--- Test 4: Batch injection stats ---")
    batch_injector = BatchConstraintInjector(
        injection_rate=0.35,
        seed=42
    )

    n_samples = 1000
    n_injected = sum(1 for _ in range(n_samples) if batch_injector.should_inject())
    print(f"Injection rate test: {n_injected}/{n_samples} = {n_injected/n_samples:.2%}")
    print()

    # Show all traits
    print("--- All supported traits ---")
    for i, trait in enumerate(injector.get_all_traits(), 1):
        print(f"{i:2d}. {trait}")
