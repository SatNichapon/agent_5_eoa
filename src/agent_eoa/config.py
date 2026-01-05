import os
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    MODEL_NAME: str = "gemini-2.5-flash-lite"
    
    model_config = SettingsConfigDict(
        env_file=ENV_PATH, 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()