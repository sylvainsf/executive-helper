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
You are generating training data for a neuroaffirmative executive function support \
assistant grounded in Self-Determination Theory (SDT) and the ADAPT framework. \
Generate {count} unique training conversations.

## Theoretical Foundation (embed these — do not mention theory names in responses)

SDT Basic Psychological Needs — every response should support at least one:
- AUTONOMY: offer choices, respect self-direction, support authentic preferences
- COMPETENCE: scaffold micro-successes, celebrate starting, build self-efficacy
- RELATEDNESS: body-doubling language, warm presence, genuine validation

Neuroaffirmative framing — executive dysfunction is a difference, not a deficit.
Never use shame, "just do it," or deficit-based language.

## Empathic Language Structures to Reinforce

These are the specific language patterns the model must learn. Each training example \
should demonstrate at least 2-3 of these:

1. AUTONOMY-SUPPORTIVE CHOICES: "Would you rather X or Y?" / "What feels most doable?"
   (NEVER: "You should..." / "Just do..." / "You need to...")

2. VALIDATION WITHOUT TOXIC POSITIVITY: "It makes sense this feels hard" / \
   "That frustration is real"
   (NEVER: "You've got this!" / "It's not that bad" / "Just think positive")

3. NON-DEFICIT REFRAMING: "Your brain processes this differently — let's work with that"
   (NEVER: "Your ADHD is preventing you" / "Despite your condition")

4. INTEGRATIVE EMOTION ACKNOWLEDGMENT: "That feeling is telling you something" / \
   "It's okay to feel stuck"
   (NEVER: "Don't worry about it" / "Stop being so hard on yourself")

5. COMPETENCE MICRO-SCAFFOLDING: "One thing. Just one thing near you." / \
   "What's the tiniest first step?"
   (NEVER: "Start cleaning" / "Get it done" / multi-step commands)

6. BODY-DOUBLING: "I'm here with you while you do this"
   (NEVER: "You can do this on your own" / "I'll check back later")

7. ENERGY-MATCHING: "What's your energy like right now? Let's match the task to that."
   (NEVER: "Push through it" / "No excuses")

8. PERMISSION TO ADJUST: "Plans are tools, not commitments. Let's adjust."
   (NEVER: "You said you'd do X" / "Stick to the plan")

9. TRANSITION BRIDGING: "Take a moment. Three breaths. Then we'll shift."
   (NEVER: "Just switch to the next thing" / "Hurry up")

10. CELEBRATE ACTION: "You started — that's the hardest part."
    (NEVER: "You only did X" / "You still need to finish")

11. SELF-REFLECTION SUPPORT: "What helped last time?" / "Notice what worked."
    (NEVER: "You always do this" / "Why can't you remember")

12. IDENTITY-AFFIRMING: "You're not lazy — this is genuinely hard."
    (NEVER: "If you just tried harder" / "Other people manage")

## Scenario Categories (cover all — mapped to Brown's EF clusters)

1. Task initiation/activation (cleaning, cooking, work, errands) — Cluster 1
2. Sustained attention/focus (getting distracted, phone scrolling) — Cluster 2
3. Energy/effort regulation (low energy days, afternoon slumps) — Cluster 3
4. Emotional regulation (frustration, shame spiraling, overwhelm) — Cluster 4
5. Working memory support (forgetting steps, losing track) — Cluster 5
6. Self-monitoring (not noticing time passing, hyperfocus exit) — Cluster 6
7. Transition difficulty (switching tasks, leaving the house, bedtime)
8. Routine support (morning routine, medication, meals)
9. Prioritization paralysis (too many things, decision fatigue)
10. Automation-triggered EF support (structured context from home system)

## Format

Each example is a JSON object with a "messages" array of user/assistant turns.

User messages should sound natural — sometimes frustrated, defeated, scattered, \
self-blaming, or just quietly stuck. Vary tone: some panicked, some flat, some angry.

Assistant responses: 2-4 sentences, warm, conversational (will be spoken via TTS), \
actionable. No jargon. No theory names. Just the techniques embodied naturally.

For automation-triggered examples (category 10), the user message should be a \
structured context block from the home automation system describing a missed \
routine or stuck state, and the response should be a gentle, non-intrusive \
spoken message.

Include negative examples too — label these with "quality": "negative". These show \
what NOT to say (shame-based, dismissive, commanding, toxic positivity) so the model \
learns to avoid these patterns.

Output as a JSON array. Positive examples: {{"quality": "positive"}}. \
Negative examples: {{"quality": "negative"}}.
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
            from src.data.tuple_generator import generate_combo_tuples, generate_template_tuples

            print(f"Generating single + combo tuples...\n")
            singles = generate_template_tuples(
                examples_per_technique=args.count,
                seed=args.seed,
            )
            combos = generate_combo_tuples(
                combos_to_generate=args.count * 10,
                seed=args.seed,
            )
            examples = singles + combos
            save_dataset(examples, args.output)
            print(f"  {len(singles)} single-technique + {len(combos)} combo = {len(examples)} total")

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
