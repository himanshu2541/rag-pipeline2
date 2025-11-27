import logging
from config import config
from components.embedding_model import get_embedding_model
from providers.llm import LLMProvider
from providers.retriever import RetrieverProvider
from providers.ingestor import Ingestor
from providers.chain import ChainProvider
from providers.stt import STTProvider

logger = logging.getLogger(__name__)

class RAGSystem:
    """
    The main coordinator class that wires all the components together.
    """
    def __init__(self, config_instance=config):
        logger.info("Initializing RAG System...")
        self.config = config_instance
        
        # 1. Init core components
        self.embeddings = get_embedding_model()
        self.llm_provider = LLMProvider(self.config)
        self.llm = self.llm_provider.get_llm()
        
        # 2. Init retriever provider (starts with just Pinecone)
        self.retriever_provider = RetrieverProvider(
            self.config, 
            self.embeddings
        )
        
        # 3. Init ingestor (which needs the retriever provider to update it)
        self.ingestor = Ingestor(
            self.embeddings,
            self.retriever_provider
        )
        
        # 4. Init chain provider (which needs the initial retriever)
        initial_retriever = self.retriever_provider.get_retriever()
        self.chain_provider = ChainProvider(
            self.llm,
            initial_retriever
        )

        # 5. Init STT Provider
        self.stt_provider = STTProvider(self.config)

        logger.info("RAG System initialized successfully.")

    def upload_document(self, file_path: str) -> int:
        """
        Orchestrates the ingestion of a document.
        
        Args:
            file_path: Path to the .txt file.
            
        Returns:
            Number of chunks ingested.
        """
        logger.info(f"RAGSystem: Starting document upload for {file_path}")
        # Ingest the file
        num_chunks = self.ingestor.ingest_file(file_path)
        
        # After ingestion, the ingestor tells the retriever_provider
        # to update itself (BM25 + Ensemble).
        # Now, we get the *new* ensemble retriever...
        new_retriever = self.retriever_provider.get_retriever()
        
        # ...and tell the chain_provider to rebuild the chain with it.
        self.chain_provider.update_retriever(new_retriever)
        
        logger.info(f"RAGSystem: Document upload finished for {file_path}")
        return num_chunks

    def ask_question(self, query: str) -> dict:
        """
        Orchestrates asking a question to the RAG chain.
        
        Args:
            query: The user's question.
            
        Returns:
            The response dictionary from the chain (includes "answer").
        """
        logger.info(f"RAGSystem: Received query: {query}")
        chain = self.chain_provider.get_chain()
        
        try:
            response = chain.invoke({"input": query})
            logger.info(f"RAGSystem: Generated answer.")
            return response
        except Exception as e:
            logger.error(f"Error during chain invocation: {e}")
            raise
    
    def ask_question_from_audio(self, audio_file_path: str) -> dict:
        """
        Transcribes audio and then asks the question to the RAG chain.
        """
        logger.info(f"RAGSystem: Processing audio query from {audio_file_path}")
        
        transcribed_text = self.stt_provider.transcribe(audio_file_path)
        logger.info(f"Transcribed Text: {transcribed_text}")
        
        if not transcribed_text:
            return {"answer": "I could not hear any audio.", "context": []}

        return self.ask_question(transcribed_text)