from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.redis_manager import redis_manager
from app.routers import chat
from app.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    await redis_manager.initialize()
    yield
    # Shutdown
    logger.info("Shutting down...")
    await redis_manager.close()

app = FastAPI(
    title="Chat Cache API",
    description="FastAPI application for caching chat messages with Redis",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(chat.router)

@app.get("/")
async def read_root():
    return {
        "message": "Chat Cache API is running!",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "redis": "connected"}
