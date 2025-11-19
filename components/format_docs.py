from typing import List
from langchain_core.documents import Document

def format_docs(docs: List[Document]) -> str:
    """
    A simple utility to format a list of Document objects into a single string.
    
    Note: This is often handled automatically by chains like 
    `create_stuff_documents_chain`, but is included per your structure.

    Args:
        docs: A list of Document objects.

    Returns:
        A single string with document contents separated by newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)