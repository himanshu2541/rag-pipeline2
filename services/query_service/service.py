import logging

logger = logging.getLogger(__name__)
from shared.config import config
from shared.embeddings import EmbeddingModel
from shared.database import VectorDatabase
from query_service.providers.llm import LLMProvider
from query_service.providers.retriever import RetrieverProvider
from query_service.providers.chain import ChainProvider

class QueryService:
    """
    Handles all the components related to query service. (Retrieval)
    """

    def __init__(self, config_instance=config):
        logger.info("Initializing Query Service components...")

        self.config = config_instance

        # 1. Init core components
        self.embeddings = EmbeddingModel(self.config)

        # 2. Init LLM Provider
        self.llm_provider = LLMProvider(self.config)
        self.llm = self.llm_provider.get_llm()

        # 3. Init vector database
        self.vector_store = VectorDatabase(self.config)

        # 4. Init retriever provider
        self.retriever_provider = RetrieverProvider(self.vector_store, self.config)
        self.retriever = self.retriever_provider.get_retriever()

        # 5. Init chain provider
        self.chain_provider = ChainProvider(self.llm, self.retriever)
        self.chain = self.chain_provider.get_chain()

        logger.info("Query Service components initialized successfully.")

    def process_query(self, query: str) -> dict:
        """
        Processes a user query using the RAG chain.

        Args:
            query: The user query string.

        Returns:
            The response from the LLM.
        """
        logger.info(f"Processing query: {query}")

        try:
            response = self.chain.invoke({"input": query})
            logger.info("Query processed successfully.")
            return response
        except Exception as e:
            logger.error(f"Error during chain invocation: {e}")
            raise