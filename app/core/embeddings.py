"""Embeddings generation module using OpenAI embeddings."""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """Get cached OpenAI embeddings instance.

    Returns:
        Configured OpenAIEmbeddings instance
    """
    settings = get_settings()
    logger.info(f"Initializing embeddings model: {settings.embedding_model}")
    
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    logger.info("Embeddings model initialized successfully")

    return embeddings

class EmbeddingsService:
    """Service for generating embeddings."""

    def __init__(self):
        """Initialize the embeddings service."""
        settings = get_settings()
        self.embeddings = get_embeddings()
        self.embeddings_model = settings.embedding_model

    def embed_query(self, text: str) -> list[float]:
        """Generate embeddings for a given text."""
        logger.debug(f"Generating embedding for query: {text[:50]}...")
        return self.embeddings.embed_query(text)

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents."""
        logger.debug(f"Generating embeddings for {len(docs)} documents...")
        return self.embeddings.embed_documents(docs)
