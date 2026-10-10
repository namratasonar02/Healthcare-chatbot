from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.routes import router
from app.utils import logger
import os

app = FastAPI(
    title="Healthcare AI Assistant",
    description="Mindbowser AI Engineer Hackathon Assignment",
    version="1.0.0"
)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Healthcare AI Assistant is running.")

@app.get("/", include_in_schema=False)
def serve_index():
    # Fallback to current directory if not running from root
    path = "templates/index.html"
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "..", "templates", "index.html")
    return FileResponse(path)
