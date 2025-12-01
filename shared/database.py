import logging
from langchain_pinecone import PineconeVectorStore
from shared.config import config
from shared.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Manages the initialization and connection to the vector store.
    Currently supports Pinecone as the vector store backend.
    """

    def __init__(self, config_instance=config):
        self.config = config_instance
        self.embeddings = EmbeddingModel(self.config).embeddings
        self.pinecone_store = self._get_pinecone_store()

    def _get_pinecone_store(self) -> PineconeVectorStore:
        """Returns the connected Pinecone Vector Store."""
        try:
            vector_store = PineconeVectorStore(
                index_name=self.config.PINECONE_INDEX_NAME,
                embedding=self.embeddings,
                pinecone_api_key=self.config.PINECONE_API_KEY,
            )
            logger.info(
                f"Connected to Pinecone index: {self.config.PINECONE_INDEX_NAME}"
            )
            return vector_store
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise
