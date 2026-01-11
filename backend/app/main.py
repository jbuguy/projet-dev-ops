import time
import os
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.core.rag import rag_service

app = FastAPI(title="DevOps Chatbot API")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("DevOps_RAG_...")
# Allow Frontend to talk to Backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change to specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# --- Startup Event ---


@app.on_event("startup")
async def startup_event():
    # Load the Brain into memory when server starts
    rag_service.load_artifacts()

# --- Routes ---


@app.get("/")
def root():
    return {"message": "Chatbot API is running"}


@app.get("/health")
def health_check():
    """Used by Docker/Kubernetes to check status"""
    return {
        "status": "healthy",
        "rag_ready": rag_service.is_ready
    }


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    if not rag_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="System is still initializing or no data found. Please wait or check logs."
        )

    with mlflow.start_run(run_name="chat_request", nested=True):

        # 1. Log Input
        mlflow.log_param("query", request.query)

        start_time = time.time()

        # 2. Retrieve
        relevant_chunks = rag_service.search(request.query)
        mlflow.log_metric("retrieved_chunks_count", len(relevant_chunks))

        # 3. Generate
        response_data = rag_service.generate_answer(
            request.query, relevant_chunks)

        duration = time.time() - start_time
        mlflow.log_metric("latency_seconds", duration)

        # 4. Log Output (The Answer)
        # We log this as text or a "tag" for easy reading in the UI
        mlflow.log_text(response_data["answer"], "answer.txt")

        # Optional: Log the retrieved sources to see what the model actually found
        sources_str = "\n".join(response_data["sources"])
        mlflow.log_text(sources_str, "retrieved_sources.txt")

        return response_data
