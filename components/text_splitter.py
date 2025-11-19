import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import config

logger = logging.getLogger(__name__)

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of documents into smaller chunks.

    Args:
        documents: The list of Document objects to split.

    Returns:
        A list of smaller Document chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} document(s) into {len(split_docs)} chunks.")
        return split_docs
    except Exception as e:
        logger.error(f"Failed to split documents: {e}")
        raise