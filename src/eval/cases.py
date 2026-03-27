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
