import time
import os
import uuid
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.core.rag import rag_service

app = FastAPI(title="DevOps Chatbot API")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("DevOps_RAG_Context")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Memory Storage (In-Memory for Demo, use Redis for Prod) ---
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# --- Data Models ---
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    request_id: str  # Unique ID for this specific Q&A pair

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
        relevant_chunks, _ = rag_service.search(request.query, history)
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


@app.get("/")
def root():
    return {"message": "Chatbot API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "rag_ready": rag_service.is_ready}

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Utility to clear conversation history"""
    if session_id in chat_histories:
        del chat_histories[session_id]
    return {"message": f"History cleared for {session_id}"}

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    if not rag_service.is_ready:
        raise HTTPException(status_code=503, detail="System not ready.")

    with mlflow.start_run(run_name="chat_request", nested=True):
        start_time = time.time()
        
        # 1. Retrieve History
        history = chat_histories.get(request.session_id, [])
        
        # 2. Search with Context (Pass history to service)
        relevant_chunks, actual_query_used = rag_service.search(request.query, history)
        
        mlflow.log_param("original_query", request.query)
        mlflow.log_param("contextual_query", actual_query_used)
        mlflow.log_metric("retrieved_chunks", len(relevant_chunks))

        # 3. Generate Answer
        response_data = rag_service.generate_answer(request.query, relevant_chunks)

        # 4. Update History
        # Append User Query
        history.append({"role": "user", "content": request.query})
        # Append Bot Answer
        history.append({"role": "assistant", "content": response_data["answer"]})
        
        # Save back to global store
        # Limit history length to prevent context window explosion (e.g., last 10 turns)
        chat_histories[request.session_id] = history[-10:]

        duration = time.time() - start_time
        mlflow.log_metric("latency", duration)

        return {
            "answer": response_data["answer"],
            "sources": response_data["sources"],
            "session_id": request.session_id
        }
