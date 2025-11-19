import logging
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from config import config

logger = logging.getLogger(__name__)

def get_pinecone_vector_store(embeddings: Embeddings) -> PineconeVectorStore:
    """
    Initializes and returns a PineconeVectorStore instance.
    This assumes the index already exists.

    Args:
        embeddings: The embedding model to use.

    Returns:
        An initialized PineconeVectorStore client.
    """
    try:
        vector_store = PineconeVectorStore(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
        )
        logger.info(f"Connected to Pinecone index: {config.PINECONE_INDEX_NAME}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone index: {e}")
        raise

def add_documents_to_pinecone(
    vector_store: PineconeVectorStore, 
    documents: list[Document]
):
    """
    Adds documents to the provided Pinecone vector store.
    
    Args:
        vector_store: The PineconeVectorStore instance.
        documents: The list of documents to add.
    """
    try:
        vector_store.add_documents(documents)
        logger.info(f"Successfully added {len(documents)} chunks to Pinecone.")
    except Exception as e:
        logger.error(f"Failed to add documents to Pinecone: {e}")
        raise