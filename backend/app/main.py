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
@app.on_event("startup")
async def startup_event():
    rag_service.load_artifacts()

# --- Routes ---

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    if not rag_service.is_ready:
        raise HTTPException(status_code=503, detail="System not ready.")

    # Generate a unique ID for this specific interaction
    request_id = str(uuid.uuid4())

    with mlflow.start_run(run_name="chat_request", nested=True) as run:
        # Log the request_id as a tag so we can find it later for feedback
        mlflow.set_tag("request_id", request_id)
        mlflow.log_param("session_id", request.session_id)
        
        start_time = time.time()
        
        # 1. Retrieve & Generate
        # (Assuming you updated rag.py for history as discussed previously)
        history = [] # Retrieve from your history storage
        relevant_chunks = rag_service.search(request.query, history)
        response_data = rag_service.generate_answer(request.query, relevant_chunks)

        duration = time.time() - start_time
        
        # --- 7.1 Metric: Latency ---
        mlflow.log_metric("latency_seconds", duration)
        
        # Log inputs/outputs for manual review later
        mlflow.log_text(request.query, "question.txt")
        mlflow.log_text(response_data["answer"], "answer.txt")

        return {
            "answer": response_data["answer"],
            "sources": response_data["sources"],
            "session_id": request.session_id,
            "request_id": request_id
        }

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    7.2 User Testing: Collect satisfaction scores.
    """
    # In MLflow, you can't easily 're-open' a run to add metrics later without the run_id.
    # For simplicity in this assignment, we log feedback as a NEW run linked by tag.
    with mlflow.start_run(run_name="user_feedback"):
        mlflow.set_tag("related_request_id", feedback.request_id)
        
        # --- 7.2 Metric: Satisfaction Score ---
        mlflow.log_metric("user_satisfaction_score", feedback.score)
        
        if feedback.comment:
            mlflow.log_text(feedback.comment, "user_comment.txt")
            
    return {"status": "feedback_received"}

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

