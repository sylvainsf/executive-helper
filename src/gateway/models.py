"""Ollama client for interacting with both Phi-4-mini model instances."""

from pathlib import Path

import httpx

from configs.settings import settings

_PROMPT_CACHE: dict[str, str] = {}


def _load_prompt(name: str) -> str:
    if name not in _PROMPT_CACHE:
        path = Path(settings.prompts_dir) / f"{name}_system.md"
        _PROMPT_CACHE[name] = path.read_text()
    return _PROMPT_CACHE[name]


async def chat(
    model_role: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Send a chat completion request to Ollama.

    Args:
        model_role: Either "ef" (executive function) or "auto" (automation).
        messages: Chat messages (user/assistant turns). System prompt is prepended automatically.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.

    Returns:
        The assistant's response text.
    """
    model_name = settings.ef_model
    system_prompt = _load_prompt(model_role)

    full_messages = [{"role": "system", "content": system_prompt}, *messages]

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": model_name,
                "messages": full_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


async def request_ef_support(context: dict) -> str:
    """Ask the executive function model for supportive guidance.

    Called when the system detects a need for behavioral support.
    """
    prompt = (
        f"The smart home system is requesting your help. Here is the context:\n\n"
        f"Scheduled task: {context.get('scheduled_task', 'unknown')}\n"
        f"Scheduled time: {context.get('scheduled_time', 'unknown')}\n"
        f"Current time: {context.get('current_time', 'unknown')}\n"
        f"User state: {context.get('user_state', 'unknown')}\n"
        f"Urgency: {context.get('urgency', 'low')}\n\n"
        f"Please provide a brief, warm, supportive message to help the user get started. "
        f"This will be spoken aloud, so keep it conversational and under 4 sentences."
    )
    return await chat("ef", [{"role": "user", "content": prompt}], temperature=0.8)


async def check_health() -> bool:
    """Check if Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_host}/api/tags")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False
