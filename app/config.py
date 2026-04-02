from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application configuration from environment variables."""
    LLM_API_KEY: str
    LLM_API_URL: str = "https://api.openai.com/v1/chat/completions"
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large-instruct"
    SIMILARITY_THRESHOLD: float = 0.80   # raw cosine noise floor (e5-large: noise ~0.74, signal ~0.83+)
    Z_SCORE_THRESHOLD: float = 0.7       # std deviations above per-sentence mean
    COMPETITOR_MARGIN: float = 0.02      # secondary values within this margin of best are also forwarded to LLM
    SPACY_MODEL: str = "xx_sent_ud_sm"
    POS_MODEL: str = "en_core_web_sm"
    RU_POS_MODEL: str = "ru_core_news_sm"

    PROMETHEUS_MODEL: str = "ollama/vicgalle/prometheus-7b-v2.0"
    CONFIRMATION_THRESHOLD: int = 3   # score >= this → confirmed

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

config = Settings()
