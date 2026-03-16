"""Generate training tuples from technique atoms.

Two modes:
1. TEMPLATE MODE (offline, no cloud LLM needed):
   Combines technique atoms directly into training tuples using
   scenario × pattern combinations. Fast, deterministic, but less diverse.

2. GUIDED LLM MODE (uses cloud LLM for diversity):
   Sends technique atoms as structured constraints to a cloud LLM,
   which generates diverse, natural conversations that must demonstrate
   the specified techniques. Higher quality but requires API access.

Both modes produce tagged tuples traceable to their source techniques.
"""

import itertools
import json
import random
from pathlib import Path

from src.data.techniques import ALL_TECHNIQUES, TechniqueAtom, get_coverage_report


def generate_template_tuples(
    techniques: list[TechniqueAtom] | None = None,
    examples_per_technique: int = 5,
    include_negatives: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Generate training tuples by combining technique atoms.

    For each technique:
      - Pairs each user_scenario with appropriate language_patterns
      - Creates negative examples from anti_patterns
      - Tags each tuple with technique IDs for coverage tracking

    Returns:
        List of training examples in chat format with metadata.
    """
    rng = random.Random(seed)
    techniques = techniques or ALL_TECHNIQUES
    examples = []

    for tech in techniques:
        scenarios = tech.user_scenarios
        patterns = tech.language_patterns

        # Generate positive examples
        pairs = list(itertools.product(scenarios, patterns))
        rng.shuffle(pairs)
        selected = pairs[:examples_per_technique]

        for scenario, pattern in selected:
            # Fill in simple template variables if present
            response = _fill_template(pattern, scenario, tech)

            examples.append({
                "messages": [
                    {"role": "user", "content": scenario},
                    {"role": "assistant", "content": response},
                ],
                "metadata": {
                    "technique_ids": [tech.id],
                    "category": tech.category,
                    "source": tech.source,
                    "quality": "positive",
                    "generation_method": "template",
                },
            })

        # Generate negative examples (what NOT to do)
        if include_negatives and tech.anti_patterns:
            neg_pairs = list(itertools.product(scenarios, tech.anti_patterns))
            rng.shuffle(neg_pairs)
            neg_selected = neg_pairs[:max(2, examples_per_technique // 2)]

            for scenario, anti_pattern in neg_selected:
                response = _fill_template(anti_pattern, scenario, tech)

                examples.append({
                    "messages": [
                        {"role": "user", "content": scenario},
                        {"role": "assistant", "content": response},
                    ],
                    "metadata": {
                        "technique_ids": [tech.id],
                        "category": tech.category,
                        "source": tech.source,
                        "quality": "negative",
                        "generation_method": "template",
                    },
                })

    rng.shuffle(examples)
    return examples


def generate_combo_tuples(
    techniques: list[TechniqueAtom] | None = None,
    combos_to_generate: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Generate tuples that combine 2-3 techniques naturally.

    Real conversations often call for multiple techniques at once:
    e.g., a user who's overwhelmed (emotion_integrative) AND can't start
    (competence_micro) AND is alone (relatedness_body_double).

    This generates richer training examples by combining compatible techniques.
    """
    rng = random.Random(seed)
    techniques = techniques or ALL_TECHNIQUES

    # Define natural technique pairings
    compatible_pairs = [
        ("emotion_integrative", "sdt_competence_micro"),
        ("emotion_identity_affirm", "sdt_autonomy_flex"),
        ("sdt_relatedness_body_double", "sdt_competence_micro"),
        ("emotion_integrative", "sdt_relatedness_body_double"),
        ("ef_energy", "sdt_autonomy_flex"),
        ("ef_transition", "ef_sensory"),
        ("sdt_autonomy_choice", "sdt_competence_micro"),
        ("emotion_integrative", "sdt_autonomy_choice"),
        ("emotion_identity_affirm", "sdt_competence_celebrate"),
        ("ef_working_memory", "sdt_competence_micro"),
        ("sdt_autonomy_aic", "emotion_integrative"),
        ("ef_self_monitor", "sdt_competence_celebrate"),
    ]

    tech_map = {t.id: t for t in techniques}
    examples = []

    for _ in range(combos_to_generate):
        pair = rng.choice(compatible_pairs)
        t1 = tech_map.get(pair[0])
        t2 = tech_map.get(pair[1])
        if not t1 or not t2:
            continue

        # Pick a scenario from either technique
        scenario = rng.choice(t1.user_scenarios + t2.user_scenarios)

        # Combine patterns from both techniques
        p1 = rng.choice(t1.language_patterns)
        p2 = rng.choice(t2.language_patterns)

        response_parts = [
            _fill_template(p1, scenario, t1),
            _fill_template(p2, scenario, t2),
        ]
        response = " ".join(response_parts)

        examples.append({
            "messages": [
                {"role": "user", "content": scenario},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "technique_ids": [t1.id, t2.id],
                "category": f"{t1.category}+{t2.category}",
                "source": f"{t1.source}; {t2.source}",
                "quality": "positive",
                "generation_method": "combo_template",
            },
        })

    return examples


def generate_llm_prompt_for_technique(
    tech: TechniqueAtom,
    count: int = 10,
) -> str:
    """Generate a cloud LLM prompt constrained by a specific technique atom.

    This produces a prompt you'd send to GPT-4/Claude to generate diverse
    training conversations that MUST demonstrate this specific technique.
    Much higher quality than template generation but requires API access.
    """
    return f"""\
You are generating training data for an executive function support assistant.

## Technique to Demonstrate: {tech.name}
Source: {tech.source}
Category: {tech.category}

## Description
{tech.description}

## Language Patterns to Use (vary these, don't copy verbatim)
{chr(10).join(f'- {p}' for p in tech.language_patterns)}

## Language to AVOID (never use these patterns)
{chr(10).join(f'- {p}' for p in tech.anti_patterns)}

## Situations Where This Applies
{chr(10).join(f'- {s}' for s in tech.when_to_use)}

## Instructions
Generate {count} unique training conversations. Each must:
1. Have a natural, diverse user message (frustrated, flat, anxious, scattered, self-blaming — vary the tone)
2. Have an assistant response that DEMONSTRABLY uses the "{tech.name}" technique
3. Be 2-4 sentences for the assistant response (will be spoken via TTS)
4. Sound natural and conversational (no jargon, no theory names)
5. Be a different scenario each time — cover the full range of situations

Also generate 3 NEGATIVE examples showing what NOT to say — mark these with "quality": "negative".

Output as a JSON array. Each object has:
- "messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]
- "quality": "positive" or "negative"
- "technique_id": "{tech.id}"
"""


def generate_full_llm_prompts(
    output_dir: str = "data/prompts",
    count_per_technique: int = 10,
) -> None:
    """Generate one prompt file per technique for cloud LLM batch processing."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for tech in ALL_TECHNIQUES:
        prompt = generate_llm_prompt_for_technique(tech, count_per_technique)
        path = out / f"prompt_{tech.id}.txt"
        path.write_text(prompt)

    # Also generate a coverage manifest
    manifest = {
        "techniques": [
            {
                "id": t.id,
                "name": t.name,
                "category": t.category,
                "source": t.source,
                "prompt_file": f"prompt_{t.id}.txt",
                "scenarios": len(t.user_scenarios),
                "patterns": len(t.language_patterns),
            }
            for t in ALL_TECHNIQUES
        ],
        "total": len(ALL_TECHNIQUES),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Generated {len(ALL_TECHNIQUES)} prompts in {out}/")
    print(f"Send each prompt to a cloud LLM, save responses as prompt_<id>_responses.json")


def get_technique_coverage(examples: list[dict]) -> dict:
    """Analyze which techniques are covered in a dataset."""
    covered = {}
    for ex in examples:
        meta = ex.get("metadata", {})
        for tid in meta.get("technique_ids", []):
            covered.setdefault(tid, {"positive": 0, "negative": 0})
            quality = meta.get("quality", "positive")
            covered[tid][quality] = covered[tid].get(quality, 0) + 1

    all_ids = {t.id for t in ALL_TECHNIQUES}
    missing = all_ids - set(covered.keys())

    return {
        "covered": len(covered),
        "total": len(all_ids),
        "missing": sorted(missing),
        "by_technique": covered,
    }


def _fill_template(pattern: str, scenario: str, tech: TechniqueAtom) -> str:
    """Fill simple template variables in a pattern."""
    result = pattern

    # Generic fills — these are approximate, meant for template mode.
    # Cloud LLM mode produces much more natural responses.
    fills = {
        "{option_a}": "the easiest thing",
        "{option_b}": "the most urgent thing",
        "{easy_thing}": "the small quick task",
        "{important_thing}": "the bigger one",
        "{task}": _extract_task(scenario),
        "{tiny_action}": "pick up one thing near you",
        "{small_thing}": "that small thing you did",
        "{room}": "kitchen",
        "{app/drawer/door}": "the app",
        "{step1}": "start",
        "{step2}": "keep going from there",
        "{thing}": "that approach",
        "{time}": "30",
        "{n}": "three",
        "{tiny_step}": "just walking over there",
    }

    for key, value in fills.items():
        result = result.replace(key, value)

    return result


def _extract_task(scenario: str) -> str:
    """Extract a rough task description from a user scenario."""
    # Simple heuristic — pull the verb phrase after "need to" / "should" / "have to"
    for trigger in ["need to ", "should ", "have to ", "gotta ", "want to "]:
        if trigger in scenario.lower():
            idx = scenario.lower().index(trigger) + len(trigger)
            task = scenario[idx:].split(".")[0].split(",")[0].split(" but")[0].strip()
            return task[:50]
    return "the task"
