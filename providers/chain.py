import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

logger = logging.getLogger(__name__)

class ChainProvider:
    """
    Creates and manages the RAG retrieval chain.
    """
    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever
        self.prompt = self._create_prompt()
        self.chain = self._build_chain()

    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Creates the prompt template for the RAG chain.
        """
        template = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer concise.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
        return ChatPromptTemplate.from_template(template)

    def _build_chain(self) -> Runnable:
        """
        Builds the complete retrieval-augmented generation chain.
        """
        logger.info("Building the RAG chain...")
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
        logger.info("RAG chain built successfully.")
        return retrieval_chain

    def update_retriever(self, new_retriever: BaseRetriever):
        """
        Updates the retriever and rebuilds the chain.
        This is crucial for when new documents are ingested.
        """
        logger.info("Retriever has been updated. Rebuilding RAG chain...")
        self.retriever = new_retriever
        self.chain = self._build_chain()

    def get_chain(self) -> Runnable:
        """
        Returns the runnable RAG chain.
        """
        return self.chain