import logging
import sys
from pydantic_settings import BaseSettings
from typing import Optional

class Config(BaseSettings):
    """
    Configuration class to load settings from environment variables.
    """
    OPENAI_API_KEY: str = ""
    
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    LLM_MODEL: str = "phi-3-mini-4k-instruct"
    LLM_BASE_URL: str = "http://localhost:1234/v1"

    STT_MODEL_SIZE: str = "small"  # Options: tiny, base, small, medium, large-v3
    STT_DEVICE: str = "cpu"      # "cuda" if GPU available, else "cpu"
    STT_COMPUTE_TYPE: str = "int8" # "int8" for CPU speed, "float16" for GPU precision
    STT_SERVICE_HOST: str = "stt-service"
    STT_SERVICE_PORT: int = 50051
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = ""
    
    # Text Splitting Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retriever Configuration
    BM25_WEIGHT: float = 0.4
    PINECONE_WEIGHT: float = 0.6
    
    # Data Directory
    DATA_DIR: str = "../data"

    class Config:
        # This allows loading from a .env file
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

def setup_logging():
    """
    Configures the root logger for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

config = Config()