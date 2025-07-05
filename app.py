#!/usr/bin/env python3
"""
MRIA - Medical Research Intelligence Assistant
FastAPI web service for processing research queries via POST requests
"""

import asyncio
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.agent import process_single_query
from src.utils import setup_logging, print_startup_banner, handle_error

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    conversation_id: str
    user_id: str

class QueryResponse(BaseModel):
    result: str

# Global logger
logger = None

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global logger

    # Setup logging
    logger = setup_logging("INFO")

    # Windows compatibility
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Startup logs
    print_startup_banner()
    logger.info("MRIA FastAPI server started successfully")

    yield  # App runs here

    # Shutdown logs
    if logger:
        logger.info("MRIA FastAPI server shutting down")

# Initialize FastAPI app
app = FastAPI(
    title="MRIA - Medical Research Intelligence Assistant",
    description="API interface for Medical Research Intelligence Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MRIA - Medical Research Intelligence Assistant",
        "status": "healthy",
        "version": "1.0.0",
        "query_endpoint": "/query (POST)"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "MRIA",
        "version": "1.0.0",
        "description": "Medical Research Intelligence Assistant"
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a medical research query via MRIA"""
    try:
        logger.info(f"Processing query: {request.query[:100]}... (conversation: {request.conversation_id}, user: {request.user_id})")

        # Validate input
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not request.conversation_id or not request.conversation_id.strip():
            raise HTTPException(status_code=400, detail="Conversation ID cannot be empty")
            
        if not request.user_id or not request.user_id.strip():
            raise HTTPException(status_code=400, detail="User ID cannot be empty")

        # Run processing with all 3 required parameters
        result = await process_single_query(
            request.query.strip(), 
            request.conversation_id,
            request.user_id
        )

        logger.info(f"Query processed successfully for conversation: {request.conversation_id}")
        return QueryResponse(result=result)

    except HTTPException:
        raise  # Re-raise as is
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing query: {error_msg}")
        handle_error(e, "query processing")
        raise HTTPException(status_code=500, detail=f"Processing error: {error_msg}")

#uvicorn app:app --host 0.0.0.0 --port 8000 --reload