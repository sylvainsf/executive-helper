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

Scenario voices:
- DIRECT: User addresses the assistant ("I can't start cleaning")
- SELF_TALK: User thinking aloud / muttering ("ugh, this mess...")
- OVERHEARD: Third party asks user to do something ("Hey, can you take out the trash?")
"""

import itertools
import json
import random
from pathlib import Path

from src.data.techniques import ALL_TECHNIQUES, TechniqueAtom, get_coverage_report


# ── Scenario voice wrappers ──────────────────────────────────────────────────
# These transform therapy-style scenarios into natural home-context utterances.

SELF_TALK_TEMPLATES = [
    # Muttering / thinking out loud
    "ugh... {core}",
    "*sigh* {core}",
    "okay... {core}",
    "I just... {core}",
    "man, {core}",
    "god, {core}",
    "why can't I just... {core}",
    "I don't even... {core}",
    "{core}... whatever",
    "great, {core}",
    "*staring at the {room}* ...{core}",
]

OVERHEARD_REQUEST_TEMPLATES = [
    # Housemate/partner/family asks user to do something
    'My partner just said "Hey, can you {task}?" and I froze.',
    'My roommate asked me to {task} like an hour ago and I still haven\'t moved.',
    '"Can you {task} before dinner?" — that was 30 minutes ago.',
    'My mom called and asked if I could {task} today and now I\'m spiraling.',
    '"Hey babe, can you {task}?" Sure, I said. That was two hours ago.',
    'My housemate said "it\'d be great if you could {task}" and I just... can\'t.',
    '"You said you\'d {task} this morning" — yeah, I know...',
    'Someone asked me to {task} and now I feel like I can\'t do anything at all.',
    '"All you have to do is {task}" — if only it were that simple.',
    'I told my partner I\'d {task} and now the guilt is eating me alive.',
]

HOME_TASKS = [
    "take out the trash", "do the dishes", "clean the kitchen",
    "fold the laundry", "vacuum the living room", "take the dog out",
    "start dinner", "clean the bathroom", "put away the groceries",
    "mop the floor", "water the plants", "sort the mail",
    "wipe down the counters", "tidy up the bedroom", "empty the dishwasher",
    "pick up the living room", "put your clothes away", "clean out the fridge",
    "make the bed", "sweep the porch",
]

ROOMS = ["kitchen", "living room", "bedroom", "bathroom", "hallway", "office"]


def _make_self_talk(scenario: str, rng: random.Random) -> str:
    """Transform a direct scenario into self-talk / muttering."""
    # Strip "I " prefix and lowercase for natural muttering
    core = scenario
    if core.startswith("I "):
        core = core[2:].rstrip(".")
    elif core.startswith("I'm "):
        core = core[4:].rstrip(".")
    template = rng.choice(SELF_TALK_TEMPLATES)
    room = rng.choice(ROOMS)
    return template.format(core=core, room=room)


def _make_overheard(rng: random.Random) -> str:
    """Generate an overheard third-party request scenario."""
    task = rng.choice(HOME_TASKS)
    template = rng.choice(OVERHEARD_REQUEST_TEMPLATES)
    return template.format(task=task)


# ── Incompatibility rules for auto-pairing ───────────────────────────────────
# Instead of hand-picking 26 pairs from 210 possible, we generate ALL pairs
# and exclude only the ones that don't make sense together.

# Techniques that are too similar to combine (redundant, not complementary)
SAME_NICHE = [
    {"sdt_autonomy_choice", "ef_decision_fatigue"},      # both about choosing
    {"sdt_autonomy_choice", "sdt_autonomy_aic"},          # both autonomy-choice variants
    {"sdt_autonomy_flex", "ef_routine_repair"},            # both about plan adjustment
    {"sdt_competence_micro", "sdt_competence_celebrate"},  # celebrate is the follow-up to micro
    {"ef_transition", "ef_bedtime"},                        # bedtime IS a transition
    {"ef_self_monitor", "ef_post_success"},                # post_success IS self-monitoring
]

# auto_nudge has a unique format (system-triggered, not user speech) — only
# combine it with techniques that make sense as system-initiated responses
AUTO_NUDGE_COMPATIBLE = {
    "sdt_competence_micro", "sdt_autonomy_choice", "ef_energy",
    "sdt_relatedness_body_double", "emotion_integrative",
}


def _is_compatible_pair(t1: TechniqueAtom, t2: TechniqueAtom) -> bool:
    """Check if two techniques make sense combined in a single response."""
    ids = {t1.id, t2.id}

    # Same technique
    if t1.id == t2.id:
        return False

    # Too similar / redundant
    for niche in SAME_NICHE:
        if ids == niche:
            return False

    # auto_nudge has limited compatibility
    if "auto_nudge" in ids:
        other = (ids - {"auto_nudge"}).pop()
        return other in AUTO_NUDGE_COMPATIBLE

    return True


def get_all_compatible_pairs(
    techniques: list[TechniqueAtom] | None = None,
) -> list[tuple[str, str]]:
    """Generate all compatible technique pairs automatically."""
    techniques = techniques or ALL_TECHNIQUES
    pairs = []
    for t1, t2 in itertools.combinations(techniques, 2):
        if _is_compatible_pair(t1, t2):
            pairs.append((t1.id, t2.id))
    return pairs


def generate_template_tuples(
    techniques: list[TechniqueAtom] | None = None,
    examples_per_technique: int = 5,
    include_negatives: bool = True,
    include_self_talk: bool = True,
    include_overheard: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Generate training tuples by combining technique atoms.

    For each technique:
      - Pairs each user_scenario with appropriate language_patterns (direct voice)
      - Generates self-talk variants (muttering, thinking aloud)
      - Generates overheard-request scenarios (housemate asks, user freezes)
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

        # ── Direct voice (original) ──
        pairs = list(itertools.product(scenarios, patterns))
        rng.shuffle(pairs)
        selected = pairs[:examples_per_technique]

        for scenario, pattern in selected:
            response = _fill_template(pattern, scenario, tech)
            response = _maybe_append_action(response, tech, rng, scenario=scenario)
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
                    "voice": "direct",
                },
            })

        # ── Self-talk voice (muttering, thinking aloud) ──
        if include_self_talk:
            self_talk_count = max(3, examples_per_technique // 2)
            st_pairs = list(itertools.product(scenarios, patterns))
            rng.shuffle(st_pairs)

            for scenario, pattern in st_pairs[:self_talk_count]:
                self_talk_scenario = _make_self_talk(scenario, rng)
                response = _fill_template(pattern, scenario, tech)
                response = _maybe_append_action(response, tech, rng, scenario=scenario)
                examples.append({
                    "messages": [
                        {"role": "user", "content": self_talk_scenario},
                        {"role": "assistant", "content": response},
                    ],
                    "metadata": {
                        "technique_ids": [tech.id],
                        "category": tech.category,
                        "source": tech.source,
                        "quality": "positive",
                        "generation_method": "template",
                        "voice": "self_talk",
                    },
                })

        # ── Overheard voice (third-party request) ──
        if include_overheard and tech.id != "auto_nudge":
            overheard_count = max(2, examples_per_technique // 3)
            for _ in range(overheard_count):
                overheard_scenario = _make_overheard(rng)
                pattern = rng.choice(patterns)
                response = _fill_template(pattern, overheard_scenario, tech)
                response = _maybe_append_action(response, tech, rng, scenario=overheard_scenario)
                examples.append({
                    "messages": [
                        {"role": "user", "content": overheard_scenario},
                        {"role": "assistant", "content": response},
                    ],
                    "metadata": {
                        "technique_ids": [tech.id],
                        "category": tech.category,
                        "source": tech.source,
                        "quality": "positive",
                        "generation_method": "template",
                        "voice": "overheard",
                    },
                })

        # ── Negative examples (what NOT to do) ──
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
    examples_per_pair: int = 3,
    max_pairs: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate tuples that combine 2 techniques naturally.

    Auto-generates ALL compatible pairs (instead of hand-picking a few).
    With 21 techniques this yields ~190 pairs × examples_per_pair examples.

    Real conversations often call for multiple techniques at once:
    e.g., a user who's overwhelmed (emotion_integrative) AND can't start
    (competence_micro) AND is alone (relatedness_body_double).
    """
    rng = random.Random(seed)
    techniques = techniques or ALL_TECHNIQUES

    compatible_pairs = get_all_compatible_pairs(techniques)
    if max_pairs and len(compatible_pairs) > max_pairs:
        rng.shuffle(compatible_pairs)
        compatible_pairs = compatible_pairs[:max_pairs]

    tech_map = {t.id: t for t in techniques}
    examples = []

    for pair in compatible_pairs:
        t1 = tech_map.get(pair[0])
        t2 = tech_map.get(pair[1])
        if not t1 or not t2:
            continue

        for _ in range(examples_per_pair):
            # Pick voice type: 50% direct, 30% self-talk, 20% overheard
            voice_roll = rng.random()

            if voice_roll < 0.2 and t1.id != "auto_nudge":
                scenario = _make_overheard(rng)
                voice = "overheard"
            elif voice_roll < 0.5:
                base_scenario = rng.choice(t1.user_scenarios + t2.user_scenarios)
                scenario = _make_self_talk(base_scenario, rng)
                voice = "self_talk"
            else:
                scenario = rng.choice(t1.user_scenarios + t2.user_scenarios)
                voice = "direct"

            # Combine patterns from both techniques
            p1 = rng.choice(t1.language_patterns)
            p2 = rng.choice(t2.language_patterns)

            response_parts = [
                _fill_template(p1, scenario, t1),
                _fill_template(p2, scenario, t2),
            ]
            response = " ".join(response_parts)

            # For combos, pick an action from either technique
            action_tech = rng.choice([t1, t2])
            response = _maybe_append_action(response, action_tech, rng, scenario=scenario)

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
                    "voice": voice,
                },
            })

    rng.shuffle(examples)
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
        "{option_a}": "the dishes",
        "{option_b}": "the laundry",
        "{easy_thing}": "the quick one",
        "{important_thing}": "the big one",
        "{task}": _extract_task(scenario),
        "{tiny_action}": "grab the first thing you see and deal with it",
        "{small_thing}": "that thing you just did",
        "{room}": "kitchen",
        "{app/drawer/door}": "the app",
        "{step1}": "the first thing",
        "{step2}": "the next one",
        "{thing}": "that approach",
        "{time}": "30",
        "{n}": "three",
        "{tiny_step}": "just walking over there",
    }

    for key, value in fills.items():
        result = result.replace(key, value)

    return result


def _maybe_append_action(
    response: str,
    tech: TechniqueAtom,
    rng: random.Random,
    scenario: str = "",
    action_probability: float = 0.4,
) -> str:
    """Possibly append a system action JSON to the response.

    The EF model is part of a smart home system. When appropriate, it can
    trigger actions like setting timers, playing music, or dimming lights.
    The action JSON is appended as a separate line after the spoken text.

    Not every response should have an action — only when the technique
    naturally leads to one and the scenario makes sense for the action.
    """
    if not tech.actions or rng.random() > action_probability:
        return response

    # Filter actions that don't make sense for this scenario
    viable = _filter_actions_for_scenario(tech.actions, scenario, tech.id)
    if not viable:
        return response

    action = rng.choice(viable)
    return f"{response}\n{action}"


# Keywords that signal high-urgency or time-critical scenarios
_URGENT_KEYWORDS = [
    "medication", "meds", "urgency: high", "appointment", "office closes",
    "leave for", "travel time",
]

# Keywords that signal discrete/instant tasks (not sustained 5-min activities)
_INSTANT_TASK_KEYWORDS = [
    "medication", "meds", "take a pill", "swallow",
    "reply to", "text back", "respond to",
]

# Keywords indicating media consumption (episodes, videos, scrolling)
_MEDIA_KEYWORDS = [
    "watching", "videos", "scrolling", "episode", "show",
    "gaming", "game", "youtube",
]

# Keywords for hygiene/self-care that shouldn't be casually dismissed
_SELFCARE_KEYWORDS = [
    "shower", "hygiene", "brush teeth", "eat", "sleep",
]

# Keywords for committed social or deadline obligations
_COMMITTED_KEYWORDS = [
    "can't cancel", "expecting", "friend waiting", "interview",
    "insurance plan", "health insurance",
]


def _filter_actions_for_scenario(
    actions: list[str], scenario: str, tech_id: str,
) -> list[str]:
    """Remove actions that don't make sense for a given scenario.

    This prevents nonsensical training pairs like medication + "set_timer for
    quick restart" or guilt-insomnia + "last episode timer".
    """
    if not scenario:
        return actions  # No scenario context — allow all (overheard voice etc.)

    low = scenario.lower()
    viable = []

    for action_str in actions:
        # Parse the action type
        if "dismiss_intent" in action_str:
            # Don't dismiss urgent, self-care, or committed obligations
            if any(kw in low for kw in _URGENT_KEYWORDS):
                continue
            if tech_id == "sdt_autonomy_aic" and any(kw in low for kw in _SELFCARE_KEYWORDS):
                continue
            if any(kw in low for kw in _COMMITTED_KEYWORDS):
                continue

        elif "set_timer" in action_str:
            # Don't set "work timers" for instant/discrete tasks (medication, texts)
            if any(kw in low for kw in _INSTANT_TASK_KEYWORDS):
                # Allow timers only for bedtime (wind-down is different)
                if tech_id != "ef_bedtime":
                    continue
            # For time-pressure, skip timer if deadline <= timer duration
            if tech_id == "ef_time_pressure" and _deadline_too_tight(low):
                continue

        elif "set_reminder" in action_str:
            # Don't delay urgent items with a reminder
            if any(kw in low for kw in _URGENT_KEYWORDS):
                # Medication reminders and appointments shouldn't get a 15-min delay
                if "urgency: high" in low or "medication" in low or "meds" in low:
                    continue

        # Action survived all filters
        viable.append(action_str)

    return viable


def _deadline_too_tight(scenario_lower: str) -> bool:
    """Check if the scenario mentions a deadline of 10 minutes or less."""
    import re
    # Match patterns like "in 10 minutes", "10 min", "in 10 min"
    m = re.search(r"in\s+(\d+)\s*min", scenario_lower)
    if m and int(m.group(1)) <= 10:
        return True
    return False


def _extract_task(scenario: str) -> str:
    """Extract a rough task description from a user scenario."""
    # Simple heuristic — pull the verb phrase after "need to" / "should" / "have to"
    for trigger in ["need to ", "should ", "have to ", "gotta ", "want to "]:
        if trigger in scenario.lower():
            idx = scenario.lower().index(trigger) + len(trigger)
            task = scenario[idx:].split(".")[0].split(",")[0].split(" but")[0].strip()
            return task[:50]
    return "the task"
