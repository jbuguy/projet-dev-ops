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
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- 1. Lifespan Management (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Application starting - loading models in background...")
    # Load models in background to not block startup
    # They will be loaded on first request if not ready
    print("[INFO] Application ready (models will load on first request)")
    yield
    print("[INFO] Application shutting down...")

app = FastAPI(title="DevOps Chatbot API", lifespan=lifespan)

# --- 2. MLflow & Middleware Setup ---
# Initialize MLflow lazily to avoid blocking startup
_mlflow_initialized = False

def _init_mlflow():
    global _mlflow_initialized
    if not _mlflow_initialized:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("DevOps_RAG_Context")
        _mlflow_initialized = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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


# --- Routes ---

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    # Initialize MLflow on first request
    _init_mlflow()
    
    # Load artifacts if not ready
    if not rag_service.is_ready:
        print("[INFO] RAG Service not ready - loading artifacts...")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, rag_service.load_artifacts)
            print("[INFO] RAG Service loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load RAG artifacts: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG service: {str(e)}")
    
    if not rag_service.is_ready:
        raise HTTPException(status_code=503, detail="System failed to initialize. Check server logs.")

    # Generate a unique ID for this specific interaction
    request_id = str(uuid.uuid4())

    try:
        start_time = time.time()
        
        # Run blocking operations in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=4)
        
        # 1. Retrieve relevant chunks
        history = []  # Retrieve from your history storage
        relevant_chunks = await loop.run_in_executor(
            executor,
            lambda: rag_service.search(request.query, history)
        )
        
        # 2. Generate answer with timeout
        try:
            response_data = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: rag_service.generate_answer(request.query, relevant_chunks[0])
                ),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="API request timed out. Please try again.")

        duration = time.time() - start_time
        
        # Log metrics
        with mlflow.start_run(run_name="chat_request", nested=True) as run:
            mlflow.set_tag("request_id", request_id)
            mlflow.log_param("session_id", request.session_id)
            mlflow.log_metric("latency_seconds", duration)
            mlflow.log_text(request.query, "question.txt")
            mlflow.log_text(response_data["answer"], "answer.txt")
        
        return QueryResponse(
            answer=response_data["answer"],
            sources=response_data["sources"],
            session_id=request.session_id,
            request_id=request_id
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Chat endpoint failed: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    7.2 User Testing: Collect satisfaction scores.
    """
    # Initialize MLflow on first request
    _init_mlflow()
    
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

