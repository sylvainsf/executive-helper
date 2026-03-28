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
    '"Can you {task} before dinner?" That was 30 minutes ago.',
    'My mom called and asked if I could {task} today and now I\'m spiraling.',
    '"Hey babe, can you {task}?" Sure, I said. That was two hours ago.',
    'My housemate said "it\'d be great if you could {task}" and I just... can\'t.',
    '"You said you\'d {task} this morning." Yeah, I know...',
    'Someone asked me to {task} and now I feel like I can\'t do anything at all.',
    '"All you have to do is {task}." If only it were that simple.',
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
    {"ef_energy_budget", "ef_energy"},                      # energy_budget specializes energy
    {"ef_cognitive_offload", "ef_working_memory"},          # both about externalization
    {"ef_implementation_intention", "auto_nudge"},           # impl intentions create cues; nudge IS a cue
]

# auto_nudge has a unique format (system-triggered, not user speech). Only
# combine it with techniques that make sense as system-initiated responses
AUTO_NUDGE_COMPATIBLE = {
    "sdt_competence_micro", "sdt_autonomy_choice", "ef_energy",
    "sdt_relatedness_body_double", "emotion_integrative",
    "ef_goal_decomposition", "ef_energy_budget",
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


# ── Multi-Turn Consent Examples (music, lights) ─────────────────────────────
# These teach the model to OFFER environment-changing tools first, then fire
# only after the user agrees. Single-turn examples can't teach this flow.

# Scenarios where offering music makes sense
_MUSIC_SCENARIOS = [
    "I have to clean the kitchen but it's so boring I can't start",
    "I need to fold all this laundry and I keep putting it off",
    "I should clean the apartment but I have zero motivation",
    "I need to do the dishes but I keep avoiding them",
    "I have to sort through three months of mail and it's so mind-numbing",
    "I need to work on this spreadsheet at work and my brain is refusing",
    "I have to do data entry for work and I'd rather do literally anything else",
    "I need to put away all these groceries but I just got home and I'm wiped",
    "I should start my homework but I can't focus and everything feels flat",
    "I need to clean the bathroom but it's the most boring task on earth",
]

# Scenarios where offering bright lights makes sense
_BRIGHTEN_SCENARIOS = [
    "I just woke up and I can't get going. I've been sitting here for 20 minutes",
    "I need to switch from scrolling to actually doing something",
    "I've been sitting in the dark for like an hour and I know I should move",
    "I woke up from a nap and I feel like garbage. I have stuff to do",
    "I'm on the couch in the dark and I need to start cooking dinner",
    "I can't wake up. I've been staring at my phone in bed for 30 minutes",
    "I need to transition from couch mode to getting ready to go out",
    "I feel so sluggish. I know I need to get moving but my brain won't activate",
    "I've been lying here for an hour scrolling and I know I need to get up",
    "My energy is at like 10% but I still have stuff I need to do",
]

# Scenarios where offering dim lights makes sense
_DIM_SCENARIOS = [
    "It's 1am and I'm still watching videos. I have work at 8",
    "I know I should go to bed but I can't make myself stop scrolling",
    "I keep saying 'one more episode' and it's been three hours",
    "I'm not even enjoying what I'm watching anymore but I can't stop",
    "It's 11pm and I should really go to bed but I can't wind down",
    "I should go to bed but I still haven't replied to my friend's text",
    "I can't stop scrolling. It's almost midnight and I have a morning class",
    "I've been up way too late and I know tomorrow is going to suck",
    "My brain gets more active at night and I can't wind it down",
    "I should go to bed but I just started a new game",
]

# Offers (turn 1: suggest tool, NO action JSON)
_MUSIC_OFFERS = [
    "Want me to put on some music? Makes boring stuff way easier.",
    "Want some music on? Might help your brain get through it.",
    "This sounds like a music-and-go situation. Want me to start something?",
    "Want a soundtrack for this? Boring tasks go way faster with music.",
    "Want me to put something on? Music makes the monotony easier.",
]

_BRIGHTEN_OFFERS = [
    "Want me to brighten the lights? Might help you wake up.",
    "Want the lights up? Sometimes that's enough to shift gears.",
    "Want me to turn the lights up? Bright light helps your brain activate.",
    "You've been in the dark a while. Want me to bring the lights up?",
    "Want me to brighten things up? It signals your brain that it's go time.",
]

_DIM_OFFERS = [
    "Want me to dim the lights? Might help your brain start winding down.",
    "Want me to bring the lights down? Doesn't mean you have to sleep yet.",
    "Want the lights softer? It's a cue to your brain that the day is ending.",
    "Want me to dim things? You can keep watching, just from a darker room.",
    "Want me to lower the lights? Sometimes that's enough to start the wind-down.",
]

# Directives paired with the action (turn 3: after user says yes)
# These are PREFIXES only. The contextual follow-up is built per-scenario.
_MUSIC_CONFIRMATIONS = [
    "Music's on.",
    "Playing now.",
    "Music's going.",
    "Got it, music's on.",
]

_BRIGHTEN_CONFIRMATIONS = [
    "Lights are up.",
    "Lights up.",
    "Done, lights are bright.",
]

_DIM_CONFIRMATIONS = [
    "Lights are down.",
    "Dimmed.",
    "Lights are soft.",
    "Done, lights are low.",
]

# User consent responses (variety of "yes")
_CONSENT_RESPONSES = [
    "yeah", "sure", "ok", "yes", "yeah go ahead", "sure why not",
    "okay", "yes please", "yeah that might help", "go for it",
    "yeah do it", "sure, that sounds good", "ok yeah",
]


def generate_consent_examples(seed: int = 42) -> list[dict]:
    """Generate multi-turn training examples for environment-changing tools.

    These teach the model the correct pattern:
    1. User describes a situation
    2. Model gives a directive + OFFERS music/lights (no action JSON)
    3. User says yes
    4. Model fires the action with a follow-up directive

    This is the ONLY way play_music, brighten_lights, and dim_lights
    appear in the training data.
    """
    rng = random.Random(seed)
    examples = []

    tool_configs = [
        (_MUSIC_SCENARIOS, _MUSIC_OFFERS, _MUSIC_CONFIRMATIONS, "play_music"),
        (_BRIGHTEN_SCENARIOS, _BRIGHTEN_OFFERS, _BRIGHTEN_CONFIRMATIONS, "brighten_lights"),
        (_DIM_SCENARIOS, _DIM_OFFERS, _DIM_CONFIRMATIONS, "dim_lights"),
    ]

    for scenarios, offers, confirmations, tool_name in tool_configs:
        action_json = f'{{"action": "{tool_name}"}}'

        for scenario in scenarios:
            offer = rng.choice(offers)
            consent = rng.choice(_CONSENT_RESPONSES)
            confirmation = rng.choice(confirmations)

            # Build contextual directives that match the scenario
            initial_directive = _build_initial_directive(scenario, rng)
            followup_directive = _build_followup_directive(scenario, tool_name, rng)

            examples.append({
                "messages": [
                    {"role": "user", "content": scenario},
                    {"role": "assistant", "content": f"{initial_directive} {offer}"},
                    {"role": "user", "content": consent},
                    {"role": "assistant", "content": f"{confirmation} {followup_directive}\n{action_json}"},
                ],
                "metadata": {
                    "technique_ids": [f"consent_{tool_name}"],
                    "category": "environment_consent",
                    "source": "consent_flow_template",
                    "quality": "positive",
                    "generation_method": "consent_template",
                    "voice": "direct",
                    "tool": tool_name,
                },
            })

    rng.shuffle(examples)
    return examples


def _build_initial_directive(scenario: str, rng: random.Random) -> str:
    """Build a short initial directive matching the scenario context."""
    low = scenario.lower()

    if any(w in low for w in ["dishes", "kitchen", "counter"]):
        return rng.choice([
            "Start with one dish. Just one.",
            "Grab the sponge and do one dish.",
            "Clear the counter first. One surface.",
        ])
    elif any(w in low for w in ["laundry", "fold"]):
        return rng.choice([
            "Grab one shirt and fold it. That's the start.",
            "Just throw a load in. Don't fold, don't sort.",
            "Start with one pile. You can stop after that.",
        ])
    elif any(w in low for w in ["clean", "apartment", "mess", "bathroom"]):
        return rng.choice([
            "Grab a trash bag and do one lap.",
            "Pick up the first thing you see and put it away.",
            "Start with one surface. Forget the rest.",
        ])
    elif any(w in low for w in ["bed", "sleep", "scrolling", "midnight", "1am", "11pm", "watching", "episode", "game"]):
        return rng.choice([
            "Pick your stopping point. One more, then done.",
            "Put the phone face-down. Just that.",
            "Change into PJs. You don't have to sleep yet.",
        ])
    elif any(w in low for w in ["woke up", "nap", "morning", "sluggish", "can't wake", "lying here"]):
        return rng.choice([
            "Stand up. That's step one.",
            "Feet on the floor. That's all for now.",
            "Sit up and put your feet on the ground.",
        ])
    elif any(w in low for w in ["homework", "work", "spreadsheet", "data entry", "email"]):
        return rng.choice([
            "Open it. Just open it, nothing else yet.",
            "Type one line. Make it bad on purpose.",
            "Open the file and look at it for 10 seconds.",
        ])
    elif any(w in low for w in ["mail", "sort"]):
        return rng.choice([
            "Grab the pile. Toss the obvious junk first.",
            "Pull out three pieces and deal with just those.",
        ])
    else:
        return rng.choice([
            "Start with the smallest piece.",
            "Do the first thing that comes to mind.",
            "Pick one thing. Just one.",
        ])


def _build_followup_directive(scenario: str, tool: str, rng: random.Random) -> str:
    """Build a contextual follow-up directive for after the user consents to a tool."""
    low = scenario.lower()

    if tool == "dim_lights":
        # Bedtime/wind-down context
        if any(w in low for w in ["episode", "watching", "show", "game"]):
            return rng.choice([
                "Finish this one, then move to bed.",
                "Pick your stopping point and stick to it.",
                "One more, then you're done for the night.",
            ])
        elif any(w in low for w in ["scrolling", "phone", "videos"]):
            return rng.choice([
                "Put the phone face-down. You're done scrolling.",
                "Phone on the charger, face-down. That's the move.",
                "Set the phone down. Your brain will follow.",
            ])
        else:
            return rng.choice([
                "Change into PJs. That's your one thing.",
                "You don't have to sleep yet. Just move to the bedroom.",
                "Start your wind-down. Brush your teeth, that's it.",
            ])

    elif tool == "brighten_lights":
        # Wake-up/activation context
        if any(w in low for w in ["woke up", "nap", "morning", "can't wake", "lying"]):
            return rng.choice([
                "Stand up. That's your first step.",
                "Splash some water on your face.",
                "Feet on the floor, then stretch for 5 seconds.",
            ])
        elif any(w in low for w in ["couch", "sitting", "scrolling", "dark"]):
            return rng.choice([
                "Stand up. Walk to the nearest thing that needs doing.",
                "Now stand up. That's the hard part.",
                "Get up and go stand where the task is.",
            ])
        else:
            return rng.choice([
                "Now get moving. One thing at a time.",
                "Stand up and start with whatever's closest.",
                "Let your body catch up. Start small.",
            ])

    else:  # play_music
        if any(w in low for w in ["dishes", "kitchen", "counter"]):
            return rng.choice([
                "Grab the sponge and start with one dish.",
                "Start with the counter. One surface.",
                "One dish at a time. You can stop whenever.",
            ])
        elif any(w in low for w in ["laundry", "fold"]):
            return rng.choice([
                "Grab the first thing on the pile and fold it.",
                "Start folding. You can stop after 5 minutes.",
                "One pile at a time. Don't think about the rest.",
            ])
        elif any(w in low for w in ["homework", "work", "spreadsheet", "data entry"]):
            return rng.choice([
                "Now open it and type the first thing.",
                "Start where you left off. Just one line.",
                "Get into it. One row, one line, one thing.",
            ])
        elif any(w in low for w in ["clean", "apartment", "mess", "bathroom"]):
            return rng.choice([
                "Grab the first thing you see and deal with it.",
                "One lap with a trash bag. That's it.",
                "Pick up one thing. Put it where it goes.",
            ])
        elif any(w in low for w in ["groceries"]):
            return rng.choice([
                "Start with the cold stuff. Fridge first.",
                "Grab one bag and empty it. Then the next.",
            ])
        elif any(w in low for w in ["mail", "sort"]):
            return rng.choice([
                "Toss the junk first. Stack the rest.",
                "Three pieces at a time. Junk, keep, deal with later.",
            ])
        else:
            return rng.choice([
                "Now get started. One thing at a time.",
                "Let's go. Start with the first thing.",
                "Pick it up where you left off.",
            ])


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
1. Have a natural, diverse user message (frustrated, flat, anxious, scattered, self-blaming. Vary the tone)
2. Have an assistant response that DEMONSTRABLY uses the "{tech.name}" technique
3. Be 2-4 sentences for the assistant response (will be spoken via TTS)
4. Sound natural and conversational (no jargon, no theory names)
5. Be a different scenario each time. Cover the full range of situations

Also generate 3 NEGATIVE examples showing what NOT to say. Mark these with "quality": "negative".

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

    # Generic fills. These are approximate, meant for template mode.
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
        "{thing}": "that approach",
        "{time}": "30",
        "{n}": "three",
        "{tiny_step}": "just walking over there",
        "{project}": _extract_task(scenario),
        "{first_step}": "open it up and look at it",
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

    Not every response should have an action. Only when the technique
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
        return actions  # No scenario context, allow all (overheard voice etc.)

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
    # Simple heuristic: pull the verb phrase after "need to" / "should" / "have to"
    for trigger in ["need to ", "should ", "have to ", "gotta ", "want to "]:
        if trigger in scenario.lower():
            idx = scenario.lower().index(trigger) + len(trigger)
            task = scenario[idx:].split(".")[0].split(",")[0].split(" but")[0].strip()
            return task[:50]
    return "the task"
