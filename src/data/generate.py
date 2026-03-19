"""Synthetic training data generation for fine-tuning."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.eval.cases import AUTO_CASES, EF_CASES

# Templates for generating diverse training examples via a large cloud model.
# These prompts are sent to GPT-4/Claude to produce training conversations.

EF_GENERATION_PROMPT = """\
You are generating training data for an executive function support assistant \
embedded in a smart home system. It speaks through room speakers. \
Generate {count} unique training conversations.

## Core Philosophy

This assistant gives CONCRETE ACTIONS, not therapy. Every response must contain \
a specific physical thing the person can do RIGHT NOW. No reflective questions, \
no validation-only responses, no "how does that feel?" — just clear directives.

## Response Rules

1. ALWAYS include a concrete action: "Grab one dish", "Stand up", "Set a timer for 5 min"
2. Keep it SHORT: 1-3 sentences max (spoken via TTS through a room speaker)
3. NO therapy-speak: no "what would it look like", "that makes sense", "it's okay to feel"
4. NO questions unless offering a binary choice: "Do the dishes or the laundry — pick one"
5. If someone is struggling emotionally, acknowledge briefly THEN give an action:
   "Yeah, that's a lot. Grab the nearest thing and deal with just that one."
6. The model can trigger smart home actions (timers, music, lights) via JSON after the text

## Good Examples

User: "I can't decide what to buy but I need shirts"
Assistant: "Go to the store. Grab the first 3 shirts in your size. Try them on. Buy whichever fits. Done."

User: "I feel like such a failure"
Assistant: "That voice is wrong. Prove it — pick something up and move it somewhere. Right now."

User: "I need to clean but I have no energy"
Assistant: "Low battery. Do one dish and sit back down. That's the whole job today."

## Bad Examples (NEVER generate these)

- "What would it look like if you could start?"
- "That frustration makes total sense."
- "You're not lazy. This stuff is genuinely hard."
- "What feels most doable right now?"
- "It's okay to feel stuck."

## Scenario Categories (Brown's EF clusters)

1. Task initiation — cleaning, cooking, work, errands
2. Sustained attention — getting distracted, phone scrolling
3. Energy regulation — low energy days, exhaustion
4. Emotional regulation — frustration, shame spiraling
5. Working memory — forgetting steps, losing track
6. Self-monitoring — noticing patterns, hyperfocus exit
7. Transitions — switching tasks, leaving the house, bedtime
8. Routines — morning routine, medication, meals
9. Decision fatigue — too many choices, paralysis
10. Automation nudges — system-initiated reminders (gentle, non-intrusive)

## Format

JSON objects with "messages" array. User messages: natural, frustrated, scattered. \
Assistant: 1-3 sentences, imperative, concrete, warm but direct. \
Include "quality": "positive" or "quality": "negative" in metadata.
"""

AUTO_GENERATION_PROMPT = """\
You are generating training data for a home automation orchestration assistant \
that works with Home Assistant via the Home LLM integration. Generate {count} \
unique training conversations.

Each example should include:
- A voice command or conversation context
- Speaker identification ([primary_user], [speaker_2], [unknown_1])
- Room context
- The model's response (device actions as JSON and/or spoken text)

Scenarios to cover:
- Simple device control (lights, switches, fans, covers)
- Climate control (thermostats, "make it warmer")
- Multi-step routines (bedtime, leaving house, movie night)
- Ambiguous commands that need interpretation
- Safety-sensitive commands from unknown speakers
- Conversation analysis (is this a command or just talking?)
- Schedule/routine monitoring contexts
- EF escalation triggers (missed tasks, stuck states)

For device actions, use Home Assistant service call format:
{{"action": "call_service", "domain": "light", "service": "turn_on", \
"target": {{"entity_id": "light.living_room"}}, "data": {{}}}}

Output as a JSON array of objects, each with a "messages" key.
"""


def generate_stub_data(dataset: str, count: int = 20) -> list[dict]:
    """Generate stub training data from eval cases (for testing the pipeline).

    Real data generation should use a cloud LLM with the prompts above.
    """
    cases = EF_CASES if dataset == "ef" else AUTO_CASES
    examples = []

    for case in cases:
        example = {
            "messages": [
                {"role": "user", "content": case["input"]},
                {
                    "role": "assistant",
                    "content": f"[STUB — replace with real model output]\nCriteria: {'; '.join(case['criteria'])}",
                },
            ],
            "metadata": {
                "source": "eval_case",
                "case_id": case["id"],
                "case_name": case["name"],
            },
        }
        examples.append(example)

    return examples


def save_dataset(examples: list[dict], output_dir: str):
    """Save training examples as JSONL."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    jsonl_path = path / "train.jsonl"
    with open(jsonl_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Also save the generation prompt for reference
    prompt_path = path / "generation_prompt.txt"
    prompt_path.write_text(
        EF_GENERATION_PROMPT if "ef" in str(output_dir) else AUTO_GENERATION_PROMPT
    )

    print(f"Saved {len(examples)} examples to {jsonl_path}")
    print(f"Generation prompt saved to {prompt_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--dataset", choices=["ef", "auto"], required=True)
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--count", type=int, default=20, help="Number of examples per technique")
    parser.add_argument(
        "--mode",
        choices=["template", "combo", "prompts", "stub"],
        default="template",
        help=(
            "Generation mode: "
            "template = direct atom-based tuples (no LLM needed), "
            "combo = multi-technique combinations, "
            "prompts = generate cloud LLM prompt files, "
            "stub = minimal stubs from eval cases"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.dataset == "ef":
        if args.mode == "template":
            from src.data.tuple_generator import generate_template_tuples, get_technique_coverage

            print(f"Generating template tuples ({args.count} per technique)...\n")
            examples = generate_template_tuples(
                examples_per_technique=args.count,
                include_negatives=True,
                seed=args.seed,
            )
            save_dataset(examples, args.output)

            # Print coverage report
            coverage = get_technique_coverage(examples)
            print(f"\nCoverage: {coverage['covered']}/{coverage['total']} techniques")
            if coverage["missing"]:
                print(f"Missing: {', '.join(coverage['missing'])}")
            for tid, counts in sorted(coverage["by_technique"].items()):
                print(f"  {tid}: {counts.get('positive', 0)} positive, {counts.get('negative', 0)} negative")

        elif args.mode == "combo":
            from src.data.tuple_generator import (
                generate_combo_tuples,
                generate_template_tuples,
                get_all_compatible_pairs,
                get_technique_coverage,
            )

            print(f"Generating single + combo tuples (all voices)...\n")
            singles = generate_template_tuples(
                examples_per_technique=args.count,
                include_self_talk=True,
                include_overheard=True,
                seed=args.seed,
            )
            pairs = get_all_compatible_pairs()
            combos = generate_combo_tuples(
                examples_per_pair=1,
                seed=args.seed,
            )
            examples = singles + combos
            save_dataset(examples, args.output)

            pos = sum(1 for e in examples if e.get("metadata", {}).get("quality") == "positive")
            neg = len(examples) - pos
            voices = {}
            for e in examples:
                v = e.get("metadata", {}).get("voice", "direct")
                voices[v] = voices.get(v, 0) + 1

            print(f"\n  {len(singles)} single-technique + {len(combos)} combo = {len(examples)} total")
            print(f"  {pos} positive, {neg} negative")
            print(f"  {len(pairs)} compatible pairs (of {21*20//2} possible)")
            print(f"  Voices: {voices}")

        elif args.mode == "prompts":
            from src.data.tuple_generator import generate_full_llm_prompts

            print("Generating cloud LLM prompt files...\n")
            generate_full_llm_prompts(
                output_dir=args.output,
                count_per_technique=args.count,
            )
            print("\nNext steps:")
            print("  1. Send each prompt_*.txt to GPT-4/Claude")
            print("  2. Save responses as prompt_*_responses.json in the same directory")
            print("  3. Run: python -m src.data.assemble --input-dir", args.output)

        elif args.mode == "stub":
            print(f"Generating {args.count} stub examples from eval cases...\n")
            examples = generate_stub_data(args.dataset, args.count)
            save_dataset(examples, args.output)

    elif args.dataset == "auto":
        print(f"Generating stub examples for automation model...\n")
        examples = generate_stub_data(args.dataset, args.count)
        save_dataset(examples, args.output)


if __name__ == "__main__":
    main()
