from langchain_core.prompts import ChatPromptTemplate

class PromptTemplate:
    """
    Holds prompt templates for the RAG chain.
    """

    def __init__(self):
        self.prompt = self._create_rag_prompt()

    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """
        Creates the prompt template for the RAG chain.
        """
        return ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer elaborative and easy to understand.

        Context:
        {context}

        Question:
        {input}
        
        Answer the question based on the context above."""
        )
    
    def get_prompt(self) -> ChatPromptTemplate:
        """
        Returns the RAG prompt template.
        """
        return self.prompt
