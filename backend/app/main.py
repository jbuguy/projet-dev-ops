import time
import os
import uuid
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from app.core.rag import rag_service

# --- 1. Lifespan Management (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Application starting up...")
    # Load the RAG artifacts (FAISS, Models) only once
    rag_service.load_artifacts()
    yield
    print("[INFO] Application shutting down...")

app = FastAPI(title="DevOps Chatbot API", lifespan=lifespan)

# --- 2. MLflow & Middleware Setup ---
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
# Ensure this matches the experiment name used in ingest.py
mlflow.set_experiment("DevOps_RAG_Context")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. In-Memory Storage ---
# Stores conversation context. In production, use Redis.
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# --- 4. Data Models ---
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str
    request_id: str  # Unique ID for tracking feedback

class FeedbackRequest(BaseModel):
    request_id: str
    score: int  # 1-5 scale
    comment: Optional[str] = None

# --- 5. Routes ---

@app.get("/")
def root():
    return {"message": "Chatbot API is running"}

@app.get("/health")
def health_check():
    """Used by Docker/Kubernetes to check status"""
    status = "healthy" if rag_service.is_ready else "initializing"
    return {
        "status": status, 
        "rag_ready": rag_service.is_ready
    }

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Utility to clear conversation history for a user"""
    if session_id in chat_histories:
        del chat_histories[session_id]
    return {"message": f"History cleared for {session_id}"}

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    """
    Main Chat Endpoint.
    Handles: Context retrieval -> Vector Search -> LLM Generation -> MLflow Logging
    """
    if not rag_service.is_ready:
        raise HTTPException(
            status_code=503, 
            detail="System is still initializing or no data found. Please wait or check logs."
        )

    # Generate a unique ID for this specific interaction
    request_id = str(uuid.uuid4())

    # Start MLflow Run
    with mlflow.start_run(run_name="chat_request", nested=True):
        mlflow.set_tag("request_id", request_id)
        mlflow.log_param("session_id", request.session_id)
        
        start_time = time.time()
        
        # 1. Retrieve History
        history = chat_