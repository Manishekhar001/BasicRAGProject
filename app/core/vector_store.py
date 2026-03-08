"""Vector Store module for Qdrant operations."""

from typing import Any, List
from functools import lru_cache
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Dimension map for known OpenAI embedding models.
# Extend this dict when adding new models.
_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def _get_embedding_dimension() -> int:
    """Return the vector dimension for the configured embedding model."""
    model = settings.embedding_model
    dim = _EMBEDDING_DIMENSIONS.get(model)
    if dim is None:
        logger.warning(
            f"Unknown embedding model '{model}'; defaulting dimension to 1536. "
            "Add the model to _EMBEDDING_DIMENSIONS in vector_store.py if this is wrong."
        )
        dim = 1536
    return dim

@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Get cached Qdrant Client instance.

    Returns:
        QdrantClient: Qdrant Client instance.
    """
    logger.info(f"Connecting to Qdrant at: {settings.qdrant_url}")

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    logger.info("Qdrant client connected successfully")
    return client


class VectorStoreService:
    """Service for managing vector store operations."""

    def __init__(self, collection_name: str = None):
        """Initialize vector store service.

        Args:
            collection_name: Name of the Qdrant collection (default from settings)
        """
        self.client = get_qdrant_client()
        self.collection_name = collection_name or settings.collection_name

        self.embeddings = get_embeddings()

        self._embedding_dimension = _get_embedding_dimension()
        self._ensure_collection()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        logger.info(f"VectorStoreService initialized for collection: {self.collection_name}")

    def _ensure_collection(self) -> None:
        """Ensure collection exists in Qdrant."""
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            logger.info(
                f"Collection '{self.collection_name}' exists with "
                f"{collection_info.points_count} points"
            )
        except Exception as e:
            # UnexpectedResponse for older qdrant-client; newer versions may raise
            # ValueError or a generic Exception for a missing collection.
            # Re-raise if it is not a "not found" situation.
            err_str = str(e).lower()
            is_not_found = (
                isinstance(e, UnexpectedResponse)
                or "not found" in err_str
                or "doesn't exist" in err_str
                or "does not exist" in err_str
            )
            if is_not_found:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self._embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents provided")
            return []

        logger.info(f"Adding {len(documents)} documents to collection")

        ids = [str(uuid4()) for _ in documents]

        self.vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

        logger.info(f"Added {len(documents)} documents to collection")
        return ids

    def search(self, query: str, k: int | None = None) -> List[Document]:
        """Search for similar documents in the vector store."""
        k = k or settings.retrieval_k

        if not query:
            logger.warning("No query provided")
            return []

        logger.debug(f"Searching for: {query[:50]}... (k={k})")

        results = self.vector_store.similarity_search(
            query=query,
            k=k,
        )

        logger.debug(f"Found {len(results)} results")
        return results

    def search_with_score(self, query: str, k: int | None = None) -> List[tuple[Document, float]]:
        """Search for similar documents in the vector store with scores."""
        k = k or settings.retrieval_k

        if not query:
            logger.warning("No query provided")
            return []

        logger.debug(f"Searching for: {query[:50]}... (k={k})")

        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
        )

        logger.debug(f"Found {len(results)} results")
        return results

    def get_retriever(self, k: int | None = None) -> Any:
        """Get a retriever for the vector store."""
        k = k or settings.retrieval_k

        logger.debug(f"Creating retriever with k={k}")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k
            }
        )

    def delete_collection(self) -> None:
        """Delete a collection from Qdrant."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted successfully")

    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
            }
        except UnexpectedResponse:
            return {
                "name": self.collection_name,
                "points_count": 0,
                "indexed_vectors_count": 0,
                "status": "not_found",
            }

    def health_check(self) -> bool:
        """Check if vector store is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False