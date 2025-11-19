import logging
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends

from models.chat import ChatQuery, ChatResponse
from models.document import DocumentContext
from models.upload import UploadResponse

from config import config, setup_logging
from rag_system import RAGSystem

setup_logging()
logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
os.makedirs(config.DATA_DIR, exist_ok=True)

app = FastAPI(
    title="RAG System",
    description="A RAG system with BM25 and Pinecone, built with FastAPI.",
    version="1.0.0"
)

try:
    rag_system = RAGSystem(config)
except Exception as e:
    logger.critical(f"Failed to initialize RAGSystem: {e}", exc_info=True)
    rag_system = None 

# Dependency to check if RAG system is healthy
def get_rag_system():
    if rag_system is None:
        logger.error("RAG system is not initialized. Endpoint cannot serve.")
        raise HTTPException(
            status_code=503, 
            detail="RAG system is not available. Check logs."
        )
    return rag_system

# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"status": "ok", "message": "RAG System API is running."}

@app.post("/upload", response_model=UploadResponse, tags=["RAG"])
async def upload_document(
    file: UploadFile = File(...),
    system: RAGSystem = Depends(get_rag_system)
):
    """
    Uploads a .txt file, ingests it into the RAG system.
    """

    filename = file.filename or ""
    if not filename.lower().endswith(".txt"):
        logger.warning(f"Upload rejected: Non-txt file {filename}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only .txt files are allowed."
        )

    file_path = os.path.join(config.DATA_DIR, filename)
    
    try:
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to save file: {e}"
        )
    finally:
        file.file.close()

    try:
        # Ingest the document
        logger.info(f"Starting ingestion for {filename}...")
        num_chunks = system.upload_document(file_path)
        logger.info(f"Ingestion complete for {filename}.")

        return UploadResponse(
            message="File uploaded and ingested successfully.",
            filename=filename,
            chunks_ingested=num_chunks
        )
    except Exception as e:
        logger.error(f"Failed to ingest file {filename}: {e}", exc_info=True)
        os.remove(file_path) # removes the saved file on failure
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process and ingest file: {e}"
        )

@app.post("/chat", response_model=ChatResponse, tags=["RAG"])
async def chat(
    query: ChatQuery,
    system: RAGSystem = Depends(get_rag_system)
):
    """
    Asks a question to the RAG system based on ingested documents.
    """
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        logger.info(f"Received chat query: {query.query}")
        response = system.ask_question(query.query)
        
        # Format context for Pydantic model
        formatted_context = [
            DocumentContext(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) 
            for doc in response.get("context", [])
        ]
        
        return ChatResponse(
            answer=response.get("answer", "No answer found."),
            context=formatted_context
        )
    except Exception as e:
        logger.error(f"Error processing chat query '{query.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing chat query: {e}"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)