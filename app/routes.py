from fastapi import APIRouter, HTTPException
from app.models import AskRequest, AskResponse, IngestResponse, HealthResponse, SystemInfoResponse
from app.config import settings
from app.agent import get_agent
from app.rag import get_rag_service
from app.utils import logger

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy")

@router.get("/api/info", response_model=SystemInfoResponse)
def get_system_info():
    return SystemInfoResponse(
        llm_source=settings.llm_source,
        embedding_source=settings.embedding_source,
        has_gemini_key=bool(settings.gemini_api_key),
        ollama_model=settings.ollama_model
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents():
    logger.info("Received request to ingest documents.")
    try:
        result = get_rag_service().ingest_documents()
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Document ingestion failed.")

@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    logger.info(f"Received question: {request.question}")
    try:
        result = get_agent().process_query(request.question)
        return AskResponse(**result)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the question.")
