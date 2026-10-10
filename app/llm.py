from langchain_community.llms import Ollama
from langchain.llms.base import LLM
from typing import Optional, List, Any
from app.config import settings
import httpx
import os

class GeminiLLM(LLM):
    api_key: str
    model: str = "gemini-1.5-flash"

    @property
    def _llm_type(self) -> str:
        return "gemini-api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        try:
            response = httpx.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            res_data = response.json()
            return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise ValueError(f"Gemini API request failed: {e}")

def get_llm():
    """Returns the LLM instance (Gemini API or Ollama)."""
    if settings.llm_source == "gemini" and settings.gemini_api_key:
        return GeminiLLM(api_key=settings.gemini_api_key)
        
    # Default to local Ollama
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return Ollama(model=settings.ollama_model, base_url=ollama_base_url)

