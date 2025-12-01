import logging

from langchain_core.retrievers import BaseRetriever
from shared.database import VectorDatabase
from shared.config import config

logger = logging.getLogger(__name__)

class RetrieverProvider:
    
    def __init__(self, vector_store: VectorDatabase, config_instance = config):
        self.config = config_instance
        self.vector_store = vector_store
        self.vector_retriever = self.vector_store.pinecone_store.as_retriever()
        self.ensemble_retriever: BaseRetriever = self.vector_retriever

    # Add any additional methods needed for retrieval here

    def get_retriever(self) -> BaseRetriever:
        """
        Returns the ensemble retriever.
        """
        return self.ensemble_retriever