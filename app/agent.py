from app.rag import get_rag_service
from app.utils import logger

def check_available_slots() -> str:
    return "Cardiology:\n- Monday 10:00 AM\n- Monday 2:00 PM\n\nNeurology:\n- Tuesday 11:00 AM"

class Agent:
    def __init__(self):
        self.appointment_keywords = ["appointment", "booking", "slot", "schedule"]

    def process_query(self, question: str) -> dict:
        logger.info(f"Agent processing query: {question}")
        
        question_lower = question.lower()
        is_appointment = any(keyword in question_lower for keyword in self.appointment_keywords)
        
        if is_appointment:
            logger.info("Routing to appointment tool.")
            answer = check_available_slots()
            return {
                "answer": answer,
                "sources": [],
                "confidence": "high"
            }
        else:
            logger.info("Routing to RAG service.")
            return get_rag_service().ask_question(question)

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent
