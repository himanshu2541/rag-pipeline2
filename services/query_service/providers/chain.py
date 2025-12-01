from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from services.query_service.components.prompt import PromptTemplate

import logging

logger = logging.getLogger(__name__)


class ChainProvider:
    """
    Provides a Retrieval-Augmented Generation (RAG) chain.
    """

    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever
        self.prompt = PromptTemplate().get_prompt()
        self.chain = self._build_chain()

    def _build_chain(self):
        """
        Creates the RAG chain using the provided LLM and retriever.
        """
        logger.info("Creating RAG chain...")
        doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, doc_chain)
        logger.info("RAG chain created successfully.")
        return retrieval_chain

    def get_chain(self):
        """
        Returns the current RAG chain.
        """
        return self.chain
