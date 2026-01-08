from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.core.rag import rag_service

app = FastAPI(title="DevOps Chatbot API")

# --- Pydantic Models (Data Contracts) ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# --- Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    # Load the heavy ML models when the server starts
    rag_service.load_artifacts()

# --- Endpoints ---
@app.get("/health")
def health_check():
    """Kubernetes/Docker uses this to check if app is alive."""
    return {"status": "healthy", "rag_ready": rag_service.is_ready}

@app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    if not rag_service.is_ready:
        raise HTTPException(status_code=503, detail="Search index not ready. Please run ingestion.")

    # 1. Retrieve
    relevant_chunks = rag_service.search(request.query)
    
    # 2. Synthesize
    result = rag_service.generate_answer(request.query, relevant_chunks)
    
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }