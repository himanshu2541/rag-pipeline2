from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

# --- Chat Models ---
class ChatQuery(BaseModel):
    query: str = Field(..., description="The question to ask the RAG system")

class DocumentContext(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str
    context: List[DocumentContext]

# --- Ingestion Models ---
class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_ingested: int