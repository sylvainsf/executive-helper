# Executive Helper

A locally-hosted AI system for **executive dysfunction support** in a smart home, powered by a fine-tuned language model that runs entirely on your own hardware. No cloud. No subscriptions. No data leaves your network.

## What It Does

Executive Helper is built for people who experience executive dysfunction: the difficulty starting tasks, switching between activities, managing time, and maintaining routines that affects people with ADHD, autism, depression, chronic fatigue, and many other conditions.

It uses a fine-tuned EF coaching model connected to Home Assistant through a multi-room audio pipeline using [Seeedstudio ReSpeaker Lite](https://www.seeedstudio.com/ReSpeaker-Lite-p-5928.html) mic arrays. Smart home control is handled by [Home LLM](https://github.com/acon96/home-llm).

**What it can do:**

- **Detect when you're stuck** — if you say "I should start dinner" but no kitchen activity follows, it gently checks in after a configurable grace period
- **Break tasks into micro-steps** — instead of "clean the kitchen," it offers "can you pick up one thing near you right now?"
- **Track projects and goals** — when something feels too big, it helps you break it into pieces, tracks progress locally, and checks in periodically without shame or pressure
- **Be your external brain** — offloads "remember to do X" from working memory into the system so you can focus on doing, not remembering
- **Budget energy, not just time** — on low-capacity days, it suggests minimum-viable versions of tasks instead of pushing you to do the full version
- **Manage routines with compassion** — medication reminders, bedtime routines, and meal prep nudges in a warm, non-judgmental tone grounded in Self-Determination Theory
- **Use smart home tools** — the EF model can set timers, play music, adjust lights, and schedule reminders through Home Assistant to help you get unstuck
- **Identify speakers** — distinguishes you from other household members so behavioral support is directed appropriately
- **Log every decision for review** — a web UI lets you rate both functional correctness ("did it work?") and personal effectiveness ("did it work *for me*?")

## Architecture

```
Per-Room Node                              Central Server (GPU)
┌──────────────────┐                      ┌─────────────────────────────────────────┐
│ ReSpeaker Lite   │──WiFi/WebSocket─────▶│ Audio Pipeline                          │
│ (XMOS mic array) │                      │  ├─ VAD (voice activity detection)      │
│                  │                      │  ├─ Whisper ASR (tiny/base, ~150MB)     │
│ Speaker          │◀─WAV audio──────────│  ├─ Speaker diarization                 │
│ (I2S amp+driver  │                      │  └─ Wake word / ambient intent detect   │
│  or JBL Go 4)    │                      │         │                               │
└──────────────────┘                      │    ┌────┴────┐                          │
                                          │    │ Router  │                          │
                                          │    └────┬────┘                          │
                                          │  ┌──────┼──────────┐                    │
                                          │  ▼      ▼          ▼                    │
                                          │ Home   EF Model   Intent               │
                                          │ LLM    (Phi-4     Tracker              │
                                          │ (HA)   fine-tuned) (silent timers)     │
                                          │  │      │          │                    │
                                          │  │      │◀─────────┘ (escalate)         │
                                          │  ▼      ▼                               │
                                          │ Kokoro TTS (~300MB, CPU)                │
                                          │  └─ audio response → room speaker       │
                                          │                                         │
                                          │ Decision Journal (SQLite)               │
                                          │  └─ Web UI for review & feedback        │
                                          └─────────────────────────────────────────┘
                                                        │
                                                        ▼
                                               Home Assistant
                                          (devices, sensors, automations)
```

### Audio Flow

```
1. ReSpeaker streams raw 16kHz PCM over WebSocket
2. VAD segments speech from silence
3. Whisper transcribes + speaker embeddings identify who's talking
4. Router decides:
   ├─ Wake word + primary user → Home LLM (voice command)
   ├─ Wake word + unknown speaker → safety check, refuse sensitive commands
   ├─ Active EF session in room → continue multi-turn EF conversation
   ├─ Ambient speech with task intent → silent intent tracker (no response)
   ├─ Conversation detected → analysis for routine/context tracking
   └─ Ambient, no intent → ignore
5. Intent tracker silently monitors via HA sensors for follow-through
6. If no activity after grace period → EF model generates supportive nudge
7. Kokoro TTS speaks response through the room's speaker
8. Everything logged to decision journal for review
```

## Prerequisites

| Component | Version | Purpose |
|---|---|---|
| **Home Assistant** | 2025.7.0+ | Smart home platform |
| **HACS** | Latest | Home Assistant Community Store |
| **Home LLM** | v0.4.6+ | HA integration for voice-based device control (separate project) |
| **Ollama** | Latest | Model serving (EF model) |
| **Python** | 3.10+ | Executive Helper runtime |
| **GPU** | 8GB+ VRAM | Runs EF model + Whisper (or Apple Silicon with unified memory) |

### Hardware

**Central server**: Any machine with a GPU (RTX 3060 8GB, RTX 4060, etc.) or an Apple Silicon Mac. 16GB+ RAM recommended.

**Per-room nodes**:
- [Seeedstudio ReSpeaker Lite](https://www.seeedstudio.com/ReSpeaker-Lite-p-5928.html) — XMOS XU316 dual-mic far-field array
- Speaker: JBL Go 4 (~$35, aux cable) or DIY I2S amp + small driver (~$15)

## Installation

### Step 1: Set Up Home Assistant + Home LLM

If you already have HA running, skip to installing Home LLM.

#### 1a. Install Home LLM via HACS

1. Open HACS in your Home Assistant instance
2. Click the button below (or search for "Local LLM" in HACS):

   [![Open HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?category=Integration&repository=home-llm&owner=acon96)

3. **Restart Home Assistant** after installation

4. After restart, verify: a "Local LLM" device should appear in `Settings → Devices and Services → Devices`

#### 1b. Configure Home LLM with Ollama Backend

We recommend the Ollama backend since Executive Helper also uses Ollama for the EF model — one Ollama instance serves both.

**On your GPU server:**

```bash
# Install Ollama if not already installed
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the Home LLM model (3B, fine-tuned for HA control)
ollama pull hf.co/acon96/Home-Llama-3.2-3B

# Pull the EF base model (we'll fine-tune this later)
ollama pull phi4-mini:latest

# Start Ollama listening on all interfaces (for HA to connect)
# ⚠️  Only do this on your LOCAL network — not on public servers
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

**In Home Assistant:**

1. Navigate to `Settings → Devices and Services`
2. Click `+ Add Integration`, search for `Local LLM`
3. Select **Ollama API** from the backend dropdown, click `Submit`
4. Configure the connection:
   - **IP Address**: IP of your GPU server (e.g., `192.168.1.100`)
   - **Port**: `11434`
   - **Use HTTPS**: unchecked
   - **API Key**: leave blank
5. Click `Submit`
6. Under the Ollama service that appears, click `+ Add conversation agent`
   - **Model Name**: select `hf.co/acon96/Home-Llama-3.2-3B` from the list
7. Configure the model:
   - Select the **Assist** API (HA's built-in LLM API for device control)
   - Enable **in-context learning (ICL) examples**
   - Ensure the prompt template references `{{ response_examples }}`
   - Click `Submit`

#### 1c. Set Up the Voice Assistant Pipeline

1. Navigate to `Settings → Voice Assistants`
2. Click `+ Add Assistant`
3. Name it (e.g., "Executive Helper")
4. Set **Conversation agent** to the agent you just created
5. Configure STT/TTS if desired (Executive Helper provides its own audio pipeline, so this is optional for direct HA voice control)

#### 1d. Expose Entities to the Model

1. Navigate to `Settings → Voice Assistants → Expose` tab
2. Click `+ Expose Entities`
3. Select the devices you want controllable by voice

> **Important**: Each exposed entity adds tokens to the model's context window. Keep it under ~32 entities to stay within the 3B model's context limits. Expose only devices you want the model to control.

**Recommended entities to expose:**
- Lights (by room)
- Thermostat / climate
- Locks (front door, etc.)
- Media players
- Key switches (coffee maker, fan, etc.)
- Motion sensors (used by intent tracker for follow-through detection)
- Scripts for routines (bedtime, leaving house, etc.)

### Step 2: Install Executive Helper

```bash
# Clone the repo
git clone <repo-url> executive-helper
cd executive-helper

# Install dependencies
make setup

# Copy the example env file and configure
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# .env
EH_OLLAMA_HOST=http://localhost:11434
EH_HA_URL=http://homeassistant.local:8123
EH_HA_TOKEN=your_long_lived_access_token_here
EH_WAKE_WORDS=["hey helper"]
```

**Getting an HA long-lived access token:**
1. In HA, click your profile (bottom-left)
2. Scroll to "Long-Lived Access Tokens"
3. Click "Create Token", name it "Executive Helper"
4. Copy the token into your `.env`

### Step 3: Install the Executive Helper Integration

The Executive Helper integration provides all UI, config, and entities inside Home Assistant — no separate web app.

#### 3a. Install via HACS (recommended)

1. In HACS, go to **Integrations → three-dot menu → Custom repositories**
2. Add this repo URL, category **Integration**
3. Find "Executive Helper" in HACS and install
4. **Restart Home Assistant**

#### 3b. Manual installation

```bash
# Copy the custom component into your HA config
cp -r custom_components/executive_helper /path/to/ha/config/custom_components/

# Restart Home Assistant
```

#### 3c. Add the Integration

1. Navigate to `Settings → Devices and Services`
2. Click `+ Add Integration`, search for **Executive Helper**
3. Enter the IP and port of your Executive Helper backend server (default: `localhost:8000`)
4. Click `Submit` — it will verify connectivity to the backend

This creates:
- An **Executive Helper** device with status sensors
- A **Decision Journal** panel in the HA sidebar
- **Services** for triggering EF support from automations

### Step 4: Configure Room Nodes

Room nodes are configured through HA's native config flow — no separate UI.

1. Go to `Settings → Devices and Services → Executive Helper`
2. Click **Configure** (gear icon)
3. Select **Add Room**
4. Fill in:
   - **Room ID**: slug matching your HA areas (e.g., `kitchen`, `bedroom`)
   - **Room Name**: human-readable label
   - **ReSpeaker IP**: the device's IP on your network
   - **Motion Sensor** (optional): select your room's motion sensor entity for follow-through detection (e.g., `binary_sensor.kitchen_motion`)
   - **Grace Period**: how long to wait before nudging (default: 30 minutes)
5. Click **Submit**

Each room node appears as a separate **device** in HA with a connection status sensor. You can see all nodes at `Settings → Devices → Executive Helper`.

#### ReSpeaker Lite Setup

Each ReSpeaker Lite needs to be configured to stream audio to the central server:

1. Connect the ReSpeaker Lite to your WiFi network via USB setup
2. Note its IP address (check your router or use `arp -a`)
3. The ReSpeaker streams raw 16-bit PCM (mono, 16kHz) over WebSocket to `ws://<server-ip>:8000/ws/audio/<room_id>`
4. Configure the ReSpeaker's firmware to connect to this WebSocket endpoint (see ReSpeaker Lite documentation for custom firmware)
5. Audio responses (TTS) are sent back over the same WebSocket connection as WAV data

#### Manage Rooms Later

To edit or remove rooms, go to `Settings → Devices and Services → Executive Helper → Configure → Manage Rooms`.

### Step 5: Enroll Speakers

Speaker enrollment also happens through HA's config flow.

1. Go to `Settings → Devices and Services → Executive Helper → Configure`
2. Select **Manage Speakers**
3. Choose the **room node** to use for enrollment
4. Enter the **speaker's name** and **role**:
   - **Primary User**: receives EF support and full device access
   - **Household Member**: device access, no EF interventions
   - **Guest**: limited device access, no sensitive commands
5. Click **Submit** — speak naturally for 5-10 seconds when prompted

The system extracts a voice embedding and stores it locally on the backend.

| Speaker Role | Device Control | EF Support | Sensitive Commands |
|---|---|---|---|
| Primary user | Full | Yes | Yes |
| Household member | Full | No | Yes (with confirmation) |
| Guest | Limited | No | Blocked |
| Unknown | Limited | No | Blocked |

### Step 6: Pull Models and Start

```bash
# Pull models via Ollama
make pull-models

# Start the backend server
make serve
```

The backend gateway starts on `http://localhost:8000` (headless — all UI is in HA). Verify:

```bash
curl http://localhost:8000/health
# {"status": "ok", "ollama": true}
```

In HA, the **EH Backend Connected** binary sensor should show as **on**.

### Step 7: Connect HA Automations (Optional)

Executive Helper can receive triggers from Home Assistant automations for behavioral support. Add these to your HA `configuration.yaml`:

```yaml
rest_command:
  ef_support:
    url: "http://<executive-helper-ip>:8000/ef-support"
    method: POST
    headers:
      Content-Type: application/json
    payload: >
      {
        "scheduled_task": "{{ scheduled_task }}",
        "scheduled_time": "{{ scheduled_time }}",
        "current_time": "{{ now().strftime('%H:%M') }}",
        "user_state": "{{ user_state }}",
        "urgency": "{{ urgency }}"
      }
```

**Example automations:**

```yaml
automation:
  - alias: "Medication reminder"
    trigger:
      - platform: state
        entity_id: input_boolean.medication_taken
        to: "off"
        for: "00:30:00"
    action:
      - service: rest_command.ef_support
        data:
          scheduled_task: "take evening medication"
          scheduled_time: "{{ states('input_datetime.medication_time') }}"
          user_state: "medication not marked as taken"
          urgency: "high"

  - alias: "Bedtime routine nudge"
    trigger:
      - platform: time
        at: "22:30:00"
    condition:
      - condition: state
        entity_id: light.living_room
        state: "on"
    action:
      - service: rest_command.ef_support
        data:
          scheduled_task: "start bedtime routine"
          scheduled_time: "22:00"
          user_state: "living room lights still on, user likely still up"
          urgency: "medium"
```

## How the Intent Tracker Works

This is the core behavioral support mechanism — it detects when you express an intention to do something, then silently monitors for follow-through.

```
You say: "I should really start making dinner"     (ambient, no wake word)
    │
    ▼  (automation model detects task intent)
Silent timer starts: 30 minute grace period
    │  No response. No indication you're being tracked.
    │
    ▼  (every 60 seconds, checks HA sensors)
Is there kitchen motion? Stove on? Kitchen lights on?
    │
    ├─ YES → timer silently resolved. You'll never know it was running.
    │
    └─ NO, 30 min passed → EF model generates a gentle nudge:
         "Hey — you mentioned dinner a bit ago. What sounds
          like the easiest first step? Even just walking to
          the kitchen counts."
              │
              ▼  (spoken through your room's speaker)
         If still no activity after 15 more minutes → second nudge
              │
              ▼  (max 3 nudges, then stops — no nagging)
```

**What triggers tracking:**
- "I should..." / "I need to..." / "I gotta..."
- "What should we have for dinner?" (implies cooking needed)
- "Is it time for my meds?"
- Mentions of tasks with time pressure

**What does NOT trigger tracking:**
- Past tense: "I already did the laundry"
- Hypothetical: "We should go to Italy someday"
- Questions to others: "Did you feed the cat?"
- Casual conversation

**Grace periods by urgency:**
| Urgency | Grace | Examples |
|---|---|---|
| High | 15 min | Medication, insulin, safety |
| Medium | 30 min | Meals, showers, scheduled tasks |
| Low | 45 min | Laundry, errands, optional tasks |

## Decision Journal

Every decision the system makes is logged and reviewable from the **EF Journal** panel in HA's sidebar (added automatically when the integration is installed).

Each decision can be rated on two axes:
1. **Functional correctness** — "Did it work?" (✓ correct / ~ partial / ✗ wrong)
2. **Personal effectiveness** — "Did it work *for me*?" (💚 helpful / 💛 neutral / 🔴 unhelpful)

Ratings feed back into the fine-tuning data pipeline: helpful decisions become positive training examples, unhelpful ones are excluded. This is the human-in-the-loop data flywheel.

## Models

| Component | Model | Size | Runs On | Purpose |
|---|---|---|---|---|
| Executive Function | Phi-4-mini-instruct (fine-tuned) | 3.8B, ~2.5GB Q4 | GPU | Coaching, task decomposition, gentle nudges |
| Speech-to-Text | Whisper (tiny or base) | 39-74M, ~150MB | CPU | Voice transcription |
| Speaker ID | ECAPA-TDNN (SpeechBrain) | ~25MB | CPU | Distinguish primary user from others |
| Text-to-Speech | Kokoro | ~300MB | CPU | Warm, natural spoken responses |

**Total VRAM**: ~2.5GB with EF model + Whisper loaded. If also running Home LLM, add ~2GB. Fits on 8GB with headroom.

## Training Data

The EF model is trained on **synthetic data** generated from 21 evidence-based technique atoms. No cloud LLM is needed — the entire pipeline runs locally from structured definitions.

### Technique Atoms

Each atom in `src/data/techniques.py` encodes a single, citable intervention:

| Field | Purpose |
|---|---|
| `id` / `name` | Unique slug and human label |
| `source` | Citation (see SOURCES.md) |
| `category` | SDT need or EF cluster |
| `when_to_use` | Trigger conditions |
| `language_patterns` | Action-oriented response templates |
| `anti_patterns` | Responses the model must **never** produce |
| `user_scenarios` | 10 diverse inputs spanning household, work, school, self-care, social, errands, and health |
| `actions` | Smart-home actions the response can trigger (JSON strings) |
| `tags` | Searchable labels |

**21 techniques** across 7 categories:

| Category | Techniques |
|---|---|
| SDT Autonomy | Choice Framing, Plan Flexibility, Authentic Inner Compass |
| SDT Competence | Micro-Scaffolding, Celebration |
| SDT Relatedness | Body Doubling, Validation |
| EF Support | Transition Bridging, Energy Matching, Sensory Scaffolding, Working Memory Externalization, Self-Monitoring |
| Emotion | Shame Spiral Interruption |
| Automation | Auto-Nudge (system-initiated, non-intrusive) |
| Time & Routine | Time Pressure Triage, Routine Repair, Decision Fatigue Reduction, Bedtime Wind-Down |
| Help Seeking | Help-Seeking Validation |

### Generation Pipeline

`make gen-data-ef` runs the pipeline:

1. **Template generation** — pairs each scenario with language patterns in 3 voices:
   - **Direct**: user speaks to the system
   - **Self-talk**: user mutters aloud ("ugh, I should really start dinner…")
   - **Overheard**: housemate asks the user to do something
2. **Combo generation** — blends two compatible techniques into a single response (e.g., shame spiral interruption + body doubling)
3. **Negative examples** — anti-pattern responses paired with the same scenarios, labeled `quality: negative` and filtered out during training
4. **Action injection** — 40% of positive examples get a system action appended (see below)

Typical output: **~817 examples** (775 positive, 42 negative), ~230 with actions.

### System Actions

The EF model can output structured actions that the pipeline executes via Home Assistant. Actions are appended as a JSON line after the spoken text:

```
Just grab one dish. That's the whole task right now.
{"action": "set_timer", "minutes": 5, "label": "quick restart"}
```

The pipeline parses the JSON, sends the spoken text to TTS, and executes the action separately.

| Action | What it does | Example |
|---|---|---|
| `set_timer` | Starts an HA timer entity | 5-minute "quick restart" for micro-commitments |
| `set_reminder` | Schedules a follow-up nudge | 60-minute "check in on energy" |
| `play_music` | Starts media in the room | Sensory scaffolding for boring tasks |
| `brighten_lights` | Boosts room brightness | Alertness cue for transitions |
| `dim_lights` | Lowers room brightness | Wind-down signal for bedtime |
| `body_double_checkin` | Schedules a "how's it going?" | Virtual body doubling presence |
| `dismiss_intent` | Stops tracking the current task | Permission to let go |

**Context-aware filtering**: The generator doesn't blindly attach actions to scenarios. A keyword filter in `_maybe_append_action` prevents nonsensical pairings:
- No `set_timer` for instant tasks (taking medication, replying to a text)
- No `dismiss_intent` for urgent or health-related items
- No `set_reminder` delays for high-urgency scenarios
- No timers when the scenario's deadline is shorter than the timer duration

### Timer Callbacks

When the EF model sets a timer or reminder, the pipeline:

1. Stores conversation context (what the user said, what technique was used, the room) in the `pending_reminders` table
2. Starts an HA timer entity (e.g., `timer.eh_quick_restart_abc123`)
3. When the timer fires, HA sends an event → the custom component catches it → calls the gateway's `/ef-reminder-callback` endpoint
4. The gateway loads the stored context and asks the EF model for a contextual follow-up
5. The follow-up is spoken through the room's speaker via TTS

This means reminders survive server restarts and carry full conversation context. The callback prompt includes:
- What the user originally said
- What the EF model told them (the spoken response)
- The reminder label and action type

This lets the model generate coherent follow-ups like *"Earlier I suggested grabbing a trash bag and doing one lap. How'd it go?"*

### Multi-Turn EF Sessions

Not every EF interaction is a single exchange. The pipeline supports **multi-turn conversations** where the EF model and user go back and forth before an action is taken.

**How it works:**

1. Auto model escalates to EF → pipeline starts an `EFSession` (room-scoped, one per room)
2. If the EF model's first response includes an action (directive + timer), it executes immediately and the session closes
3. If the first response is a question or offer (no action JSON), the session stays open
4. The next utterance in that room — regardless of wake word — routes directly to the EF model with full conversation history
5. When the model finally emits an action, it executes and the session closes

**Example flow:**
```
User: "I can't get anything done today"
EF (turn 1): "What's the thing? Name it and I'll help you chop it up."  ← no action, session open
User: "Laundry, dishes, and I need to eat"
EF (turn 2): "Eat first — wash one plate and one fork. The rest waits."  ← + set_timer, session closes
```

**Session lifecycle:**
- **Timeout**: 5 minutes of silence → auto-close
- **Close keywords**: "thanks", "nevermind", "I'm good" → graceful close
- **Turn limit**: 6 user turns max → closes with encouragement
- **Ambient suppression**: while a session is active, all speech in that room routes to EF (no wake word needed)

### Permission Rule: Music & Lights

`play_music`, `brighten_lights`, and `dim_lights` physically change someone's environment. The system prompt and training data enforce a strict rule: **these actions NEVER fire without explicit permission.**

- Turn 1: *"Want me to put on some music?"* (no action JSON)
- Turn 2 (after user says yes): *"Music's on. Grab the first dish."* + `play_music`

Timers, reminders, check-ins, and dismiss are fine single-turn — they don't change the room immediately.

## Project Structure

```
executive-helper/
├── Makefile                        # Build, serve, test, train targets
├── pyproject.toml                  # Python package config
├── SOURCES.md                      # Research references & evidence base
├── configs/
│   ├── settings.py                 # Pydantic settings (env vars)
│   └── finetune_ef.yaml            # QLoRA config for EF model
├── prompts/
│   └── ef_system.md                # System prompt for EF model (SDT-based)
├── custom_components/
│   └── executive_helper/            # HA custom integration
│       ├── manifest.json            # Integration metadata
│       ├── config_flow.py           # Config UI: backend, rooms, speakers
│       ├── coordinator.py           # Data coordinator (polls backend)
│       ├── sensor.py                # Sensor entities
│       ├── binary_sensor.py         # Binary sensor entities
│       ├── const.py                 # Constants
│       └── strings.json             # UI strings + service definitions
├── src/
│   ├── gateway/
│   │   ├── app.py                  # FastAPI gateway (headless backend)
│   │   └── models.py               # Ollama client service
│   ├── audio/
│   │   ├── streaming.py            # Audio stream manager + VAD
│   │   ├── transcription.py        # Whisper ASR + speaker diarization
│   │   ├── tts.py                  # Kokoro TTS engine
│   │   ├── pipeline.py             # End-to-end audio orchestration + EF session manager
│   │   ├── websocket_server.py     # WebSocket server for room nodes
│   │   ├── intent_tracker.py       # Silent intent monitoring + escalation
│   │   └── ha_monitor.py           # HA entity state checker
│   ├── journal/
│   │   └── store.py                # Decision journal (SQLite)
│   ├── web/
│   │   └── app.py                  # Web UI (journal, devices, speakers)
│   ├── data/
│   │   ├── generate.py             # Synthetic training data pipeline
│   │   ├── techniques.py           # 21 evidence-based technique atoms
│   │   ├── tuple_generator.py      # Template + combo tuple generation
│   │   └── preview.py              # Data inspection tool
│   ├── finetune/
│   │   ├── train.py                # QLoRA fine-tuning via Unsloth
│   │   └── export.py               # Export to GGUF for Ollama
│   └── eval/
│       ├── cases.py                # Evaluation test cases (16 cases)
│       └── run.py                  # Evaluation runner
├── tests/                          # pytest suite
└── data/                           # Generated data + journal DB (gitignored)
```

## HA Entities & Services

### Entities (created automatically)

| Entity | Type | Description |
|---|---|---|
| `sensor.active_intents` | Sensor | Number of active tracked intents |
| `sensor.last_ef_intervention` | Sensor | Total EF decision count |
| `sensor.unrated_decisions` | Sensor | Decisions awaiting review in journal |
| `sensor.helpful_rate` | Sensor | % of rated decisions marked helpful |
| `binary_sensor.eh_backend_connected` | Binary | Backend server connectivity |
| `sensor.eh_<room>_node` | Sensor | Per-room ReSpeaker connection status |

### Services

| Service | Description |
|---|---|
| `executive_helper.request_ef_support` | Trigger an EF support intervention |
| `executive_helper.dismiss_intent` | Dismiss a tracked intent |
| `executive_helper.enroll_speaker` | Start speaker enrollment |

These services can be called from HA automations, scripts, or the developer tools.

### Backend API (headless, consumed by HA integration)

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Server + Ollama status |
| `/chat` | POST | Chat with EF model |
| `/ef-support` | POST | Request executive function support |
| `/ef-reminder-callback` | POST | Timer-fired reminder follow-up (called by HA) |
| `/transcription` | POST | Process a voice transcription |
| `/ws/audio/{room_id}` | WebSocket | Stream audio from a room node |
| `/journal` | GET | Decision journal (embedded in HA panel) |
| `/api/journal/decisions` | GET | List logged decisions |
| `/api/journal/decisions/{id}/feedback` | POST | Rate a decision |
| `/api/journal/export` | GET | Export rated decisions for training |
| `/api/devices` | GET/POST | Room node management |
| `/api/intents/active` | GET | Active tracked intents |
| `/api/speakers/enroll` | POST | Speaker enrollment |

## Make Targets

```
make help               Show all available targets
make setup              Install core dependencies
make setup-dev          Install all deps (dev + audio + finetune)
make pull-models        Pull Phi-4-mini via Ollama
make serve              Start gateway server (foreground)
make serve-bg           Start gateway server (background)
make stop               Stop background server
make test               Run pytest suite
make test-ef            Evaluate EF model (baseline)
make test-audio         Test audio round-trip
make gen-data-ef        Generate synthetic EF training data
make preview-data       Preview training data samples
make finetune-ef        Fine-tune EF model (QLoRA via Unsloth)
make eval               Full evaluation suite
make export             Merge LoRA → safetensors (via PEFT, not Unsloth)
make validate-export    Validate exported model before GGUF conversion
make convert-gguf       Convert merged safetensors → GGUF bf16
make quantize-gguf      Quantize bf16 GGUF → Q4_K_M
make ollama-load        Load quantized GGUF into Ollama
make ollama-test        Quick smoke test of the Ollama model
make lint               Run ruff linter
make format             Auto-format code
make clean              Remove build artifacts
```

## Fine-Tuning & Deployment

Fine-tuning happens on a machine with a GPU (ideally an RTX 4090 or similar). The resulting model is exported to GGUF format and deployed to the machine running Ollama alongside Home Assistant.

### Architecture: Training vs Serving

```
┌─────────────────────────────┐        ┌─────────────────────────────────┐
│  TRAINING MACHINE           │        │  SERVING MACHINE (runs HA)      │
│  (Windows/WSL + RTX 4090)   │        │  (any box with 8GB+ VRAM)       │
│                             │  scp   │                                 │
│  1. Clone repo              │───────▶│  4. Receive GGUF + Modelfile    │
│  2. Fine-tune (QLoRA)       │        │  5. ollama create <model>       │
│  3. Export to GGUF          │        │  6. Update EH config            │
│                             │        │  7. Restart Executive Helper    │
└─────────────────────────────┘        └─────────────────────────────────┘
```

The training machine and serving machine can be the same box.

### Step 1: Set Up the Training Machine (WSL + RTX 4090)

> **OS Support**: Tested on WSL2 (Ubuntu 22.04) with NVIDIA RTX 4090.
> Other Linux distros with CUDA should work. Native Windows is not supported for training.

```bash
# Clone the repo on your Windows PC (inside WSL)
git clone <repo-url> executive-helper
cd executive-helper

# Create venv and install all dependencies (core + dev + finetune)
make setup-dev

# Verify GPU is accessible
.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

#### Known Issues

| Issue | Symptom | Fix |
|---|---|---|
| `transformers==4.57.2` tokenizer bug | `AttributeError: 'dict' object has no attribute 'model_type'` during export/convert | Pin `transformers>=4.57.3,<=5.2.0` (already set in pyproject.toml) |
| LongRoPE + torch.compile | `Unsupported: Data-dependent branching` during training | `TORCHDYNAMO_DISABLE=1` (already set in Makefile finetune targets) |
| Unsloth `save_pretrained_merged` | Merged weights are corrupt (gibberish output) | Export uses plain `transformers` + `peft` merge instead (already handled in export.py) |
| Unsloth remaps Phi→Llama internally | Adapter's `base_model_name_or_path` points to a BNB-4bit variant | Export auto-remaps to `microsoft/Phi-4-mini-instruct` for full-precision merge |
| `python` not found (Ubuntu 22.04) | `exit code 127` when Unsloth tries to shell out | We don't use Unsloth's GGUF export; Makefile uses llama.cpp directly |

### Step 2: Generate Training Data

```bash
# Combo mode: single-technique examples + multi-technique combos (no cloud LLM needed)
make gen-data-ef

# Or run directly with custom params:
python -m src.data.generate --dataset ef --output data/generated/ef --mode combo --count 8 --seed 42
```

The data pipeline uses 21 evidence-based technique atoms (grounded in SDT, ADAPT, and Brown's EF clusters) combined with diverse user scenarios across 7 life domains. Combo mode generates multi-technique responses — e.g., shame spiral interruption + body doubling in a single reply. ~40% of positive examples include a smart-home action (timers, lights, music). See the [Training Data](#training-data) section for full details.

Default output: **~817 examples** (775 positive, 42 negative, ~230 with actions).

```bash
# Preview the data
python -m src.data.preview
```

### Step 3: Fine-Tune

```bash
# Run fine-tuning (RTX 4090: ~10-15 minutes for 768 examples)
make finetune-ef
```

You'll see:
```
============================================================
Executive Helper — QLoRA Fine-Tuning
============================================================
  Config:     configs/finetune_ef.yaml
  Base model: unsloth/Phi-4-mini-instruct
  Dataset:    data/generated/ef/train.jsonl
  Output:     checkpoints/ef
  LoRA rank:  32
  Epochs:     3
  BF16:       True

  GPU: NVIDIA GeForce RTX 4090 (24.0 GB VRAM)

Dataset: 817 total → 775 positive examples (filtered 42 negative)
Starting training...
  Total steps: ~290
```

The LoRA adapter checkpoint is saved to `checkpoints/ef/`.

### Step 4: Export to GGUF and Load into Ollama

The export pipeline merges LoRA adapters, converts to GGUF, quantizes, and loads into Ollama.
Each step is a separate make target — run them in order:

```bash
# 1. Merge LoRA adapters into full-precision safetensors
#    (loads microsoft/Phi-4-mini-instruct in bf16, applies adapter, saves)
make export

# 2. Validate the export (checks architecture, vocab, dtypes, tokenizer)
make validate-export

# 3. Convert → quantize → load into Ollama → smoke test
make convert-gguf && make quantize-gguf && make ollama-load && make ollama-test
```

> **Important**: The targets are independent (no Make dependency chain) to avoid
> re-running expensive upstream steps. Always run them in order.

#### What the export does under the hood

1. **`make export`** — Loads the original Microsoft base model in bf16 (not the Unsloth
   4-bit variant), applies the LoRA adapter via PEFT `merge_and_unload()`, saves merged
   safetensors. Also patches `config.json` (strips `quantization_config`) and
   `tokenizer_config.json` (adds `model_type`).
2. **`make validate-export`** — Runs `scripts/validate_export.py` which checks for stale
   shard files, correct architecture/vocab, bf16 tensor dtypes, and tokenizer loading.
3. **`make convert-gguf`** — Runs llama.cpp's `convert_hf_to_gguf.py` → bf16 GGUF.
4. **`make quantize-gguf`** — Runs `llama-quantize` → Q4_K_M (~2.2GB).
5. **`make ollama-load`** — Registers in Ollama via `ollama create`.

This produces:
- `models/executive-helper-ef/` — merged safetensors (intermediate, ~7.5GB)
- `models/executive-helper-ef.bf16.gguf` — bf16 GGUF (intermediate, ~7.5GB)
- `models/executive-helper-ef.Q4_K_M.gguf` — quantized model (~2.2GB)
- `models/Modelfile.executive-helper-ef` — Ollama Modelfile (checked in)

### Step 5: Deploy to the Serving Machine

If training and serving happen on the same machine, `make ollama-load` already handled this. Otherwise, copy the GGUF and Modelfile:

```bash
# From the training machine
scp models/executive-helper-ef.Q4_K_M.gguf user@ha-server:~/models/
scp models/Modelfile.executive-helper-ef user@ha-server:~/models/
```

On the serving machine:

```bash
cd ~/models
ollama create executive-helper-ef -f Modelfile.executive-helper-ef
ollama run executive-helper-ef "I need to clean my apartment but I can't start"
```

### Step 6: Configure Executive Helper to Use the Fine-Tuned Model

Update your `.env` on the serving machine:

```bash
# .env
EH_EF_MODEL=executive-helper-ef    # your fine-tuned model
```

Restart the Executive Helper backend:
```bash
make stop && make serve
```

The EF model is now serving your fine-tuned checkpoint. All future interventions will use the new model, and decisions will be logged to the journal for feedback.

### Re-Training Cycle

As you use the system and rate decisions in the journal:

```bash
# Export rated decisions from the journal
curl http://localhost:8000/api/journal/export?rating=helpful > data/journal_helpful.jsonl

# Combine with original training data
cat data/generated/ef/train.jsonl data/journal_helpful.jsonl > data/combined/train.jsonl

# Re-train with the expanded dataset
# (edit configs/finetune_ef.yaml to point data.path to data/combined/train.jsonl)
make finetune-ef

# Export, validate, convert, quantize, and deploy
make export
make validate-export
make convert-gguf && make quantize-gguf && make ollama-load && make ollama-test
# If deploying to a remote serving machine, scp the GGUF + Modelfile
```

### Training Config Reference

The config at `configs/finetune_ef.yaml` (optimized for RTX 4090):

| Parameter | Value | Notes |
|---|---|---|
| Base model | `unsloth/Phi-4-mini-instruct` | 3.8B params |
| LoRA rank | 32 | Higher rank = more expressive |
| LoRA alpha | 64 | 2× rank |
| Quantization | 4-bit (QLoRA) | Fits in ~8GB VRAM during training |
| Batch size | 4 | 4090 can handle this easily |
| Gradient accumulation | 2 | Effective batch = 8 |
| Epochs | 3 | 768 examples × 3 = ~288 steps |
| Learning rate | 2e-4 | Standard QLoRA rate |
| BF16 | true | Use bfloat16 on Ampere+ GPUs |
| Export quantization | Q4_K_M | Good quality/size tradeoff for inference |

## Privacy & Security

- **All processing is local** — audio, transcription, LLM inference, and TTS on your hardware
- **No cloud dependencies** — fully offline after model download
- **Audio is not stored** — processed in real-time, discarded after transcription
- **Speaker embeddings stay local** — stored only on your server
- **Decision journal is local SQLite** — no external database
- **Unknown speakers are restricted** — cannot unlock doors, disable alarms, or access sensitive controls
- **HA tokens are stored in `.env`** — gitignored, never committed

## Research Basis

The EF model's approach is grounded in published research. See [SOURCES.md](SOURCES.md) for the full reference list.

Key frameworks:
- **Self-Determination Theory** (Ryan & Deci) — autonomy, competence, relatedness as core needs
- **ADAPT Framework** (Champ et al., 2025) — neuroaffirmative SDT-based intervention for adults with ADHD
- **Integrative Emotion Regulation** (Roth & Benita) — validate feelings, don't suppress them
- **Brown's EF Clusters** — six domains of executive function mapped to specific support techniques

## License

TBD
