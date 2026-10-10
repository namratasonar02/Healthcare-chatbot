import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    ollama_model: str = "phi3:mini"
    chroma_path: str = "./vector_store"
    data_path: str = "./data"
    gemini_api_key: Optional[str] = None
    llm_source: str = "ollama"
    embedding_source: str = "ollama"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# Auto-detect Gemini key to override backends
if settings.gemini_api_key:
    settings.llm_source = "gemini"
    settings.embedding_source = "gemini"

