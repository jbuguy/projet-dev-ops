from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.core.rag import rag_service

app = FastAPI(title="DevOps Chatbot API")

# Allow Frontend to talk to Backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change to specific frontend URL
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

    # 1. Retrieve
    relevant_chunks = rag_service.search(request.query)
    
    # 2. Generate
    response_data = rag_service.generate_answer(request.query, relevant_chunks)
    
    return response_data