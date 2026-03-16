from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ef_model: str = "phi3.5:3.8b-mini-instruct-q4_K_M"
    auto_model: str = "phi3.5:3.8b-mini-instruct-q4_K_M"

    # Gateway
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000

    # Audio
    whisper_model: str = "tiny"
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    audio_sample_rate: int = 16000
    wake_words: list[str] = ["hey helper"]

    # Home Assistant
    ha_url: str = "http://localhost:8123"
    ha_token: str = ""

    # Paths
    prompts_dir: str = "prompts"
    data_dir: str = "data"
    models_dir: str = "models"
    journal_db: str = "data/journal.db"

    model_config = {"env_prefix": "EH_", "env_file": ".env"}


settings = Settings()
