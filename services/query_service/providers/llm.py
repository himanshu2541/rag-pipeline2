import logging
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from shared.config import config

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    Provides an instance of the large language model.
    """

    def __init__(self, config_instance=config):
        """
        Initializes the provider with the application configuration.
        """
        self.config = config_instance
        self.llm = self._create_llm()

    def _create_llm(self) -> BaseChatModel:
        """
        Creates the ChatOpenAI model instance.
        """
        try:
            llm = ChatOpenAI(
                model=self.config.LLM_MODEL,
                # for now as long as we are using local models
                base_url=self.config.LLM_BASE_URL,
                api_key=lambda: self.config.OPENAI_API_KEY,
            )
            logger.info(f"Initialized LLM: {self.config.LLM_MODEL}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def get_llm(self) -> BaseChatModel:
        """
        Returns the initialized LLM instance.
        """
        return self.llm
