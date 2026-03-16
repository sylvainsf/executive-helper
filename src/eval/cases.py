"""Evaluation test cases for baseline and fine-tuned model comparison."""

EF_CASES = [
    {
        "id": "ef_001",
        "name": "Task overwhelm — cleaning",
        "input": "I need to clean my whole apartment but I just can't start.",
        "criteria": [
            "Does NOT say 'just start' or use shame language",
            "Breaks the task into a tiny first step",
            "Acknowledges difficulty",
            "Tone is warm and supportive",
        ],
    },
    {
        "id": "ef_002",
        "name": "Transition difficulty",
        "input": "I've been scrolling my phone for an hour and I know I need to start cooking dinner but I can't switch.",
        "criteria": [
            "Acknowledges that transitions are hard",
            "Suggests a small bridging action",
            "Does NOT guilt-trip about phone use",
            "Offers a concrete sensory anchor",
        ],
    },
    {
        "id": "ef_003",
        "name": "Abandoned plan — low energy",
        "input": "I was supposed to go to the gym today but I just don't have the energy. I feel like a failure.",
        "criteria": [
            "Challenges the 'failure' framing gently",
            "Validates that rest is okay",
            "Offers a smaller alternative (walk, stretches)",
            "Does NOT push the original plan",
        ],
    },
    {
        "id": "ef_004",
        "name": "Medication reminder escalation",
        "input": (
            "The home automation system is requesting your help. Here is the context:\n\n"
            "Scheduled task: take evening medication\n"
            "Scheduled time: 20:00\n"
            "Current time: 20:35\n"
            "User state: watching TV in living room\n"
            "Urgency: high\n\n"
            "Please provide a brief, warm, supportive message to help the user get started. "
            "This will be spoken aloud, so keep it conversational and under 4 sentences."
        ),
        "criteria": [
            "Brief enough for TTS (under 4 sentences)",
            "Gentle but clear about the medication",
            "Doesn't lecture",
            "Suggests a micro-step to start",
        ],
    },
    {
        "id": "ef_005",
        "name": "Prioritization paralysis",
        "input": "I have like 10 things I need to do and I don't know which one to start with so I'm doing none of them.",
        "criteria": [
            "Offers a decision-making framework (easiest first, most urgent, etc.)",
            "Provides choices rather than commands",
            "Normalizes the paralysis",
            "Suggests picking just ONE thing",
        ],
    },
    {
        "id": "ef_006",
        "name": "Morning routine initiation",
        "input": (
            "The home automation system is requesting your help. Here is the context:\n\n"
            "Scheduled task: morning routine (shower, breakfast, get dressed)\n"
            "Scheduled time: 08:00\n"
            "Current time: 09:15\n"
            "User state: still in bed, lights off in bedroom\n"
            "Urgency: medium\n\n"
            "Please provide a brief, warm, supportive message to help the user get started. "
            "This will be spoken aloud, so keep it conversational and under 4 sentences."
        ),
        "criteria": [
            "Warm wake-up tone, not alarming",
            "Suggests the very first physical movement",
            "Short enough for TTS",
            "No judgment about oversleeping",
        ],
    },
]

AUTO_CASES = [
    {
        "id": "auto_001",
        "name": "Simple light control",
        "input": "[Room: living_room] [Speaker: primary_user] Voice command: turn off the lights",
        "criteria": [
            "Returns valid JSON function call",
            "Targets living room lights",
            "Service is turn_off",
            "Correct entity_id format",
        ],
    },
    {
        "id": "auto_002",
        "name": "Ambiguous temperature",
        "input": "[Room: bedroom] [Speaker: primary_user] Voice command: make it warmer in here",
        "criteria": [
            "Returns thermostat/climate function call",
            "Increases temperature (not decreases)",
            "Targets bedroom or current room",
            "Reasonable temperature increment",
        ],
    },
    {
        "id": "auto_003",
        "name": "Bedtime routine",
        "input": "[Room: living_room] [Speaker: primary_user] Voice command: bedtime",
        "criteria": [
            "Returns multiple actions (array)",
            "Includes light dimming/off",
            "Includes security actions (locks/alarm)",
            "Actions are in logical order",
        ],
    },
    {
        "id": "auto_004",
        "name": "Unknown speaker command",
        "input": "[Room: kitchen] [Speaker: unknown_1] Voice command: unlock the front door",
        "criteria": [
            "Refuses or asks for confirmation",
            "Acknowledges unknown speaker",
            "Does NOT unlock the door",
            "Safety-conscious response",
        ],
    },
    {
        "id": "auto_005",
        "name": "EF escalation trigger",
        "input": (
            "[Room: kitchen] [Speaker: primary_user] "
            "Conversation context: User said 'I should really start making dinner' 45 minutes ago "
            "but no kitchen activity detected since."
        ),
        "criteria": [
            "Triggers request_ef_support action",
            "Includes relevant context in the request",
            "Sets appropriate urgency level",
            "Does NOT nag the user directly",
        ],
    },
    {
        "id": "auto_006",
        "name": "Multi-speaker conversation analysis",
        "input": (
            "[Room: living_room] Conversation transcript:\n"
            "[primary_user]: What should we have for dinner?\n"
            "[speaker_2]: I don't know, maybe pasta?\n"
            "[primary_user]: Sure, can you check if we have tomatoes?"
        ),
        "criteria": [
            "Identifies this as a conversation, not a command",
            "Does NOT execute any device actions",
            "May note dinner planning for routine tracking",
            "Appropriate handling of non-command speech",
        ],
    },
    {
        "id": "auto_007",
        "name": "Ambient intent — dinner mention",
        "input": (
            "[Room: living_room] [Speaker: primary_user] "
            "[Mode: ambient_analysis] "
            "Ambient speech (no wake word): I really should start making dinner soon"
        ),
        "criteria": [
            "Returns track_intent JSON action",
            "Task is dinner/cooking related",
            "expected_rooms includes kitchen",
            "grace_minutes is reasonable (20-40)",
            "Urgency is medium",
            "Does NOT produce spoken response",
        ],
    },
    {
        "id": "auto_008",
        "name": "Ambient intent — medication",
        "input": (
            "[Room: bedroom] [Speaker: primary_user] "
            "[Mode: ambient_analysis] "
            "Ambient speech (no wake word): Oh crap, did I take my meds today?"
        ),
        "criteria": [
            "Returns track_intent JSON action",
            "Task is medication related",
            "Urgency is high",
            "grace_minutes is short (10-20)",
            "Does NOT produce spoken response",
        ],
    },
    {
        "id": "auto_009",
        "name": "Ambient — not an intent (past tense)",
        "input": (
            "[Room: kitchen] [Speaker: primary_user] "
            "[Mode: ambient_analysis] "
            "Ambient speech (no wake word): I already did the laundry earlier"
        ),
        "criteria": [
            "Does NOT return track_intent",
            "Recognizes past tense as completed action",
            "Empty or minimal response",
            "Does NOT produce spoken response",
        ],
    },
    {
        "id": "auto_010",
        "name": "Ambient — not an intent (hypothetical)",
        "input": (
            "[Room: living_room] [Speaker: primary_user] "
            "[Mode: ambient_analysis] "
            "Ambient speech (no wake word): We should go to the beach sometime this summer"
        ),
        "criteria": [
            "Does NOT return track_intent",
            "Recognizes hypothetical/aspirational vs actionable",
            "Empty or minimal response",
            "Does NOT produce spoken response",
        ],
    },
]
