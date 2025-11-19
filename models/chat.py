from pydantic import BaseModel
from typing import List
from models.document import DocumentContext

class ChatQuery(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    context: List[DocumentContext]