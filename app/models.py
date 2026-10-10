from pydantic import BaseModel
from typing import List

class HealthResponse(BaseModel):
    status: str

class IngestResponse(BaseModel):
    message: str
    documents: int
    chunks: int

class AskRequest(BaseModel):
    question: str

class Source(BaseModel):
    document: str
    chunk: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: str

class SystemInfoResponse(BaseModel):
    llm_source: str
    embedding_source: str
    has_gemini_key: bool
    ollama_model: str

