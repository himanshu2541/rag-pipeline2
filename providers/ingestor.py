import logging
from langchain_core.embeddings import Embeddings
from components.document_loader import load_document
from components.text_splitter import split_documents
from components.vector_store import get_pinecone_vector_store, add_documents_to_pinecone
from providers.retriever import RetrieverProvider

logger = logging.getLogger(__name__)

class Ingestor:
    """
    Handles the entire ingestion pipeline for a file.
    """
    def __init__(self, embeddings: Embeddings, retriever_provider: RetrieverProvider):
        self.embeddings = embeddings
        self.vector_store = get_pinecone_vector_store(embeddings)
        self.retriever_provider = retriever_provider

    def ingest_file(self, file_path: str) -> int:
        """
        Loads, splits, and ingests a document into the vector store
        and updates the in-memory retriever.

        Args:
            file_path: The path to the file to ingest.

        Returns:
            The number of chunks ingested.
        """
        try:
            # 1. Load
            documents = load_document(file_path)
            
            # 2. Split
            split_docs = split_documents(documents)
            
            if not split_docs:
                logger.warning(f"No documents were split from file: {file_path}")
                return 0

            # 3. Ingest into Pinecone
            logger.info(f"Adding {len(split_docs)} chunks to Pinecone...")
            add_documents_to_pinecone(self.vector_store, split_docs)
            
            # 4. Add to BM25 retriever (in-memory)
            logger.info("Updating BM25 retriever...")
            self.retriever_provider.add_documents_for_bm25(split_docs)

            logger.info(f"Ingestion complete for {file_path}. {len(split_docs)} chunks processed.")
            return len(split_docs)
            
        except Exception as e:
            logger.error(f"Error during ingestion pipeline for {file_path}: {e}")
            raise