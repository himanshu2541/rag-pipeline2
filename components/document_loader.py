import logging
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_document(file_path: str) -> List[Document]:
    """
    Loads a .txt file from the given file path.

    Args:
        file_path: The path to the .txt file.

    Returns:
        A list of Document objects, where each object represents the file.
    """
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        logger.info(f"Successfully loaded document: {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load document {file_path}: {e}")
        # Depending on requirements, you might want to re-raise or return empty
        raise