import logging
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from shared.config import config

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Provides methods to initialize and retrieve embedding models
    based on the configured provider.
    """
    def __init__(self, config_instance=config):
        self.config = config_instance
        self.embeddings = self._get_embedding_model()

    def _get_embedding_model(self) -> Embeddings:
        """
        Initializes and returns the embedding model based on the config.
        Supports "openai" and "local" (HuggingFace) providers.

        Returns:
            An instance of the configured embedding model.
        """
        provider = self.config.EMBEDDING_PROVIDER.lower()
        
        try:
            if provider == "openai":
                embeddings = OpenAIEmbeddings(
                    model=self.config.EMBEDDING_MODEL,
                    api_key= lambda: self.config.OPENAI_API_KEY
                )
                logger.info(f"Initialized OpenAI embedding model: {self.config.EMBEDDING_MODEL}")
                
            elif provider == "local":
                # Uses sentence-transformers to download and run the model locally.
                # The first time this runs, it will download the model.
                embeddings = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'} # Use 'cuda' if using a GPU
                )
                logger.info(f"Initialized local embedding model: {self.config.EMBEDDING_MODEL}")
                
            else:
                raise ValueError(f"Unknown embedding provider: {provider}")
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model (provider: {provider}): {e}")
            raise