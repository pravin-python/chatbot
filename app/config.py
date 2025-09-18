from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    cache_prefix: str = "fastapi-cache"
    cache_expire_time: int = 1800
    max_messages_per_chat: int = 5
    max_products_per_user: int = 5
    
    class Config:
        env_file = ".env"

settings = Settings()
