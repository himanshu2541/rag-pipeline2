import logging
import grpc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from shared.config import config, setup_logging
from shared.models import ChatQuery, ChatResponse, DocumentContext

from protos import service_pb2, service_pb2_grpc
from query_service.service import QueryService

setup_logging()
logger = logging.getLogger("QueryService")

app = FastAPI(
    title="RAG Query Service",
    description="User-facing gateway. Handles Chat (RAG) and Audio (gRPC to STT).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    query_system = QueryService(config)
except Exception as e:
    logger.critical(f"Failed to initialize Query System: {e}", exc_info=True)
    query_system = None


def get_query_system():
    if query_system is None:
        logger.error("Query System is not initialized. Endpoint cannot serve.")
        raise HTTPException(
            status_code=503, detail="Query System is not available. Check logs."
        )
    return query_system


# --- gRPC Connection Setup ---
# Connects to the 'stt-service' container defined in docker-compose
STT_TARGET = f"{config.STT_SERVICE_HOST}:{config.STT_SERVICE_PORT}"
logger.info(f"Initializing gRPC connection to STT Service at {STT_TARGET}...")

# Keep the channel open. In high-load apps, we might want to manage this connection better.
stt_channel = grpc.insecure_channel(STT_TARGET)
stt_stub = service_pb2_grpc.TranscriberStub(stt_channel)


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Query Service is running."}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    query: ChatQuery, query_system: QueryService = Depends(get_query_system)
):
    """
    Takes a user text query and returns an answer using the RAG system.
    """
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        logger.info(f"Recieved text query: {query.query}")
        response = query_system.process_query(query.query)

        formatted_context = [
            DocumentContext(page_content=doc.page_content, metadata=doc.metadata)
            for doc in response.get("context", [])
        ]

        return ChatResponse(
            answer=response.get("answer", "No answer found."), context=formatted_context
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat query: {e}")


@app.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket, query_system: QueryService = Depends(get_query_system)
):
    """
    WebSocket endpoint for Audio Chat.
    Stream logic: Audio Bytes -> gRPC -> STT Service -> Text -> RAG -> Answer
    """
    await websocket.accept()
    logger.info("WebSocket connection established.")

    try:
        while True:
            data = await websocket.receive()

            if data["type"] == "websocket.disconnect":
                logger.info("WebSocket disconnected")
                break

            # Handle Audio Bytes
            if "bytes" in data:
                audio_chunk = data["bytes"]
                logger.info(
                    f"Received audio chunk: {len(audio_chunk)} bytes. Forwarding to STT Service..."
                )

                try:
                    # 1. Send to STT Service (gRPC)
                    # Resolve the correct request message class from the generated protos.
                    # Some proto files may name this message differently (e.g., TranscribeRequest).
                    request_cls = None
                    for candidate in (
                        "AudioRequest",
                        "TranscribeRequest",
                        "AudioChunk",
                        "StreamAudioRequest",
                        "StreamRequest",
                    ):
                        if hasattr(service_pb2, candidate):
                            request_cls = getattr(service_pb2, candidate)
                            break

                    if request_cls is None:
                        logger.error("No suitable request message found in protos.service_pb2")
                        await websocket.send_json({"error": "Voice service request type unsupported."})
                        continue

                    grpc_request = request_cls(
                        audio_content=audio_chunk,
                        filename="stream.webm",  # Metadata only
                    )

                    # Call the remote service
                    grpc_response = stt_stub.Transcribe(grpc_request)

                    if not grpc_response.success:
                        logger.error(
                            f"STT Service Error: {grpc_response.error_message}"
                        )
                        await websocket.send_json(
                            {"error": grpc_response.error_message}
                        )
                        continue

                    user_text = grpc_response.text
                    logger.info(f"Transcribed Text: {user_text}")

                    # 2. Send Transcribed Query back to Frontend
                    await websocket.send_json({"query": user_text})

                    if not user_text.strip():
                        continue

                    # 3. Process RAG with the transcribed text
                    rag_response = query_system.process_query(user_text)

                    # 4. Send Answer to Frontend
                    await websocket.send_json(
                        {
                            "answer": rag_response.get("answer"),
                            "context": [
                                {
                                    "page_content": doc.page_content,
                                    "metadata": doc.metadata,
                                }
                                for doc in rag_response.get("context", [])
                            ],
                        }
                    )

                except grpc.RpcError as e:
                    logger.critical(f"gRPC Communication Failed: {e}")
                    await websocket.send_json(
                        {"error": "Voice service is currently unavailable."}
                    )
                except Exception as e:
                    logger.error(f"Error in RAG pipeline: {e}")
                    await websocket.send_json(
                        {"error": "Error processing your question."}
                    )

    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket.")
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}", exc_info=True)
