from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.core.rag import rag_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Application starting up...")
    rag_service.load_artifacts()
    yield
    print("[INFO] Application shutting down...")

app = FastAPI(title="DevOps Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

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

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    if not rag_service.is_ready:
        raise HTTPException(
            status_code=503, 
            detail="System is still initializing or no data found. Please wait or check logs."
        )
    relevant_chunks = rag_service.search(request.query)
    response_data = rag_service.generate_answer(request.query, relevant_chunks)
    
    return response_data