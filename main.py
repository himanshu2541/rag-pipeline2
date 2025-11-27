import logging
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, system: RAGSystem = Depends(get_rag_system)):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    temp_filename = f"ws_audio_{os.urandom(4).hex()}.webm"
    temp_file_path = os.path.join(config.DATA_DIR, temp_filename)
    
    file_handle = open(temp_file_path, "wb")
    
    try:
        while True:
            data = await websocket.receive()
            
            if data["type"] == "websocket.disconnect":
                logger.info("WebSocket disconnected")
                break

            if "bytes" in data:
                file_handle.write(data["bytes"])
                
            elif "text" in data:
                text_data = data["text"]
                
                if text_data == "END":
                    file_handle.close()
                    logger.info(f"Audio received. Processing {temp_file_path}...")
                    
                    try:
                        # 1. Transcribe
                        transcribed_text = system.transcribe_audio(temp_file_path)
                        
                        if not transcribed_text:
                            # Handle silence
                            await websocket.send_json({
                                "answer": "I could not hear any audio.", 
                                "context": []
                            })
                        else:
                            # 2. Send Query to Frontend IMMEDIATELY
                            await websocket.send_json({
                                "query": transcribed_text
                            })
                            
                            # 3. Then Process the Answer
                            response = system.ask_question(transcribed_text)
                            
                            # 4. Send Answer
                            await websocket.send_json({
                                "answer": response.get("answer"),
                                "context": [
                                    {"page_content": doc.page_content, "metadata": doc.metadata}
                                    for doc in response.get("context", [])
                                ]
                            })
                            
                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                        if websocket.client_state.name == "CONNECTED":
                            await websocket.send_json({"error": str(e)})
                    
                    # Cleanup
                    os.remove(temp_file_path)
                    file_handle = open(temp_file_path, "wb") 

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if not file_handle.closed:
            file_handle.close()
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)