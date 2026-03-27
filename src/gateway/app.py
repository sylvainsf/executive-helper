"""FastAPI gateway — routes requests between the two Phi-4-mini models."""

import json
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.gateway.models import chat, check_health, request_ef_support
from src.web.app import router as journal_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Executive Helper Gateway",
    description="Executive function support gateway with smart home integration",
    version="0.1.0",
)

app.include_router(journal_router)


# ── Request / Response schemas ───────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    model: str = Field("auto", pattern=r"^(ef|auto)$")
    conversation: list[dict[str, str]] = Field(default_factory=list)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=2048)


class ChatResponse(BaseModel):
    response: str
    model: str
    ef_intervention: str | None = None


class EFSupportRequest(BaseModel):
    scheduled_task: str
    scheduled_time: str = ""
    current_time: str = ""
    user_state: str = ""
    urgency: str = Field("low", pattern=r"^(low|medium|high)$")


class TranscriptionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    room: str = "unknown"
    speaker: str = "unknown"


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    ollama_ok = await check_health()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": ollama_ok,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Direct chat with either model."""
    messages = [*req.conversation, {"role": "user", "content": req.message}]

    try:
        response = await chat(
            req.model, messages, temperature=req.temperature, max_tokens=req.max_tokens
        )
    except Exception as e:
        logger.error("Model chat failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Model unavailable: {e}")

    # Check if the automation model is requesting EF support
    ef_intervention = None
    if req.model == "auto":
        ef_intervention = await _check_for_ef_escalation(response)

    return ChatResponse(response=response, model=req.model, ef_intervention=ef_intervention)


@app.post("/ef-support", response_model=ChatResponse)
async def ef_support_endpoint(req: EFSupportRequest):
    """Directly request executive function support."""
    try:
        response = await request_ef_support(req.model_dump())
    except Exception as e:
        logger.error("EF support request failed: %s", e)
        raise HTTPException(status_code=502, detail=f"EF model unavailable: {e}")

    return ChatResponse(response=response, model="ef")


class ReminderCallbackRequest(BaseModel):
    """Fired by HA automation when a timer entity finishes."""
    timer_entity_id: str = Field(..., min_length=1)


@app.post("/ef-reminder-callback", response_model=ChatResponse)
async def ef_reminder_callback(req: ReminderCallbackRequest):
    """Callback endpoint for HA timer-fired reminders and check-ins.

    Flow:
    1. EF model response included a set_reminder or body_double_checkin action
    2. Pipeline stored context in journal DB + started a HA timer
    3. HA timer fires → HA automation calls this endpoint
    4. We look up the stored context and ask the EF model for a follow-up
    5. Response is spoken via TTS in the original room
    """
    from src.journal.store import get_pending_reminder_by_timer, complete_reminder

    reminder = get_pending_reminder_by_timer(req.timer_entity_id)
    if not reminder:
        raise HTTPException(status_code=404, detail="No pending reminder for this timer")

    # Build a contextual prompt for the EF model
    action_type = reminder["action_type"]
    original_input = reminder.get("original_user_input", "")
    label = reminder["label"]
    room = reminder["room"]

    prior_response = reminder.get("conversation_context", "")

    if action_type == "body_double_checkin":
        prompt = (
            f"You set a check-in timer for the user. They were working on: {original_input}\n"
        )
        if prior_response:
            prompt += f"You told them: \"{prior_response}\"\n"
        prompt += "Check in on them — are they still going? Be warm and brief."
    else:
        prompt = (
            f"You set a reminder for the user about: {label}\n"
            f"The original context was: {original_input}\n"
        )
        if prior_response:
            prompt += f"You told them: \"{prior_response}\"\n"
        prompt += "Deliver this reminder naturally — be warm, action-oriented, and brief."

    try:
        response = await chat("ef", [{"role": "user", "content": prompt}], temperature=0.8)
    except Exception as e:
        logger.error("EF reminder callback failed: %s", e)
        raise HTTPException(status_code=502, detail=f"EF model unavailable: {e}")

    # Mark reminder as completed
    complete_reminder(reminder["id"])
    logger.info("Reminder #%d fired in %s: %s", reminder["id"], room, response[:100])

    return ChatResponse(response=response, model="ef")


@app.post("/transcription", response_model=ChatResponse)
async def transcription_endpoint(req: TranscriptionRequest):
    """Process a voice transcription through the automation model.

    This is the primary entry point for the audio pipeline:
    ReSpeaker → Whisper → this endpoint → automation model → (optional EF) → TTS
    """
    messages = [
        {
            "role": "user",
            "content": f"[Room: {req.room}] [Speaker: {req.speaker}] Voice command: {req.text}",
        }
    ]

    try:
        response = await chat("auto", messages)
    except Exception as e:
        logger.error("Transcription processing failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Model unavailable: {e}")

    ef_intervention = await _check_for_ef_escalation(response)

    return ChatResponse(response=response, model="auto", ef_intervention=ef_intervention)


async def _check_for_ef_escalation(auto_response: str) -> str | None:
    """Parse the automation model's response for EF support requests."""
    try:
        # Look for JSON blocks requesting EF support
        if "request_ef_support" not in auto_response:
            return None

        # Try to extract the JSON — the model may embed it in mixed text
        start = auto_response.index("{")
        depth = 0
        end = start
        for i, c in enumerate(auto_response[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        parsed = json.loads(auto_response[start:end])
        if parsed.get("action") == "request_ef_support":
            context = parsed.get("context", parsed)
            return await request_ef_support(context)
    except (ValueError, json.JSONDecodeError, KeyError):
        pass
    return None
