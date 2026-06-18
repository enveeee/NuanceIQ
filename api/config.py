from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    model_path: str = "model/artifacts/distilbert-imdb"
    model_max_length: int = 512
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit: str = "100/minute"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()