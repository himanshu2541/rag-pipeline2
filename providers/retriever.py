import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from components.vector_store import get_pinecone_vector_store
from config import config

logger = logging.getLogger(__name__)

class RetrieverProvider:
    """
    Manages the creation and state of the retrievers (BM25, Pinecone, and Ensemble).
    
    Note: This implementation holds the BM25 documents in memory.
    For a production system, Persist the BM25 index
    or rebuild it from a persistent document store on startup.
    """
    def __init__(self, config_instance=config, embeddings: Embeddings = None):
        self.config = config_instance
        self.pinecone_store = get_pinecone_vector_store(embeddings)
        self.pinecone_retriever = self.pinecone_store.as_retriever()
        
        # BM25 components
        self.all_docs_in_memory: List[Document] = []
        self.bm25_retriever: BaseRetriever = None
        
        # Ensemble retriever
        self.ensemble_retriever: BaseRetriever = self.pinecone_retriever
        self._update_ensemble() # Initially, it's just Pinecone

    def add_documents_for_bm25(self, documents: List[Document]):
        """
        Adds new documents to the in-memory list and rebuilds the BM25 retriever.
        """
        self.all_docs_in_memory.extend(documents)
        logger.info(f"Total documents for BM25 in memory: {len(self.all_docs_in_memory)}")
        
        if self.all_docs_in_memory:
            try:
                self.bm25_retriever = BM25Retriever.from_documents(
                    self.all_docs_in_memory
                )
                logger.info("BM25 retriever has been updated.")
                self._update_ensemble()
            except Exception as e:
                logger.error(f"Failed to create BM25 retriever: {e}")
                # Don't raise, just log. The ensemble will fallback.
        else:
            logger.warning("No documents in memory, BM25 retriever not created.")

    def _update_ensemble(self):
        """
        Re-creates the ensemble retriever based on available retrievers.
        """
        retrievers = []
        weights = []
        
        if self.bm25_retriever:
            retrievers.append(self.bm25_retriever)
            weights.append(self.config.BM25_WEIGHT)
            
        retrievers.append(self.pinecone_retriever)
        weights.append(self.config.PINECONE_WEIGHT)

        if len(retrievers) > 1:
            logger.info(f"Creating ensemble retriever with weights: {weights}")
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers, 
                weights=weights
            )
        elif retrievers:
            logger.info("Creating ensemble with only one retriever (Pinecone).")
            self.ensemble_retriever = self.pinecone_retriever
        else:
            logger.error("No retrievers available to create an ensemble.")
            # This shouldn't happen, but as a fallback:
            self.ensemble_retriever = self.pinecone_retriever

    def get_retriever(self) -> BaseRetriever:
        """
        Returns the current ensemble retriever.
        """
        return self.ensemble_retriever