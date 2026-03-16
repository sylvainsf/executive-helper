"""FastAPI gateway — routes requests between the two Phi-3.5-mini models."""

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
    description="Routes between executive function and home automation models",
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
