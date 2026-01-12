import time
import os
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
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
# Format: { "session_id": [ {"role": "user", "content": "x"}, {"role": "assistant", "content": "y"} ] }
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# --- Data Models ---


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session" # Frontend should generate a UUID


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str

# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    rag_service.load_artifacts()

# --- Routes ---

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