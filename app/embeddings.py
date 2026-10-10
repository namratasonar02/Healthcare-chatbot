from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List
from app.config import settings
import httpx

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent?key={self.api_key}"
        self.batch_url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:batchEmbedContents?key={self.api_key}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            payload = {
                "requests": [
                    {
                        "model": self.model,
                        "content": {
                            "parts": [{"text": text}]
                        }
                    }
                    for text in chunk
                ]
            }
            try:
                response = httpx.post(self.batch_url, json=payload, timeout=60.0)
                response.raise_for_status()
                res_data = response.json()
                for item in res_data.get("embeddings", []):
                    embeddings.append(item["values"])
            except Exception as e:
                raise ValueError(f"Gemini Batch Embeddings request failed: {e}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        payload = {
            "content": {
                "parts": [{"text": text}]
            }
        }
        try:
            response = httpx.post(self.url, json=payload, timeout=60.0)
            response.raise_for_status()
            res_data = response.json()
            return res_data["embedding"]["values"]
        except Exception as e:
            raise ValueError(f"Gemini Embeddings query request failed: {e}")

def get_embeddings():
    """Returns the embeddings model (Gemini API or Ollama nomic-embed-text)."""
    if settings.embedding_source == "gemini" and settings.gemini_api_key:
        return GeminiEmbeddings(api_key=settings.gemini_api_key)
        
    return OllamaEmbeddings(model="nomic-embed-text")

