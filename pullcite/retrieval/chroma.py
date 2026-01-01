"""
ChromaDB retriever implementation.

Uses ChromaDB for persistent vector storage with efficient similarity search.
Supports both in-memory and persistent collections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import uuid

from .base import Retriever, SearchResult, SearchResults, RetrieverError

if TYPE_CHECKING:
    from ..core.chunk import Chunk
    from ..core.document import Document
    from ..embeddings.base import Embedder


@dataclass
class ChromaRetriever(Retriever):
    """
    ChromaDB-based retriever with optional persistence.

    Uses ChromaDB for efficient vector similarity search. Supports both
    in-memory and persistent storage modes.

    Attributes:
        _embedder: Embedder for converting text to vectors.
        collection_name: Name of the ChromaDB collection.
        persist_directory: Path for persistent storage (None for in-memory).
        distance_metric: Distance metric ('cosine', 'l2', or 'ip').

    Example:
        >>> from pullcite.embeddings.openai import OpenAIEmbedder
        >>> embedder = OpenAIEmbedder()
        >>> retriever = ChromaRetriever(embedder, persist_directory="./chroma_db")
        >>> retriever.index(document)
        >>> results = retriever.search("deductible amount", k=3)
    """

    _embedder: "Embedder"
    collection_name: str = "pullcite_documents"
    persist_directory: str | None = None
    distance_metric: str = "cosine"
    _client: Any = field(default=None, repr=False, compare=False)
    _collection: Any = field(default=None, repr=False, compare=False)
    _chunks: list["Chunk"] = field(default_factory=list, repr=False)
    _document_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.distance_metric not in ("cosine", "l2", "ip"):
            raise RetrieverError(
                f"Invalid distance_metric: {self.distance_metric}. "
                "Must be 'cosine', 'l2', or 'ip'."
            )

    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is not None:
            return self._client

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise RetrieverError(
                "chromadb package required. Install with: pip install chromadb"
            )

        try:
            if self.persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                self._client = chromadb.Client(
                    settings=Settings(anonymized_telemetry=False)
                )
            return self._client
        except Exception as e:
            raise RetrieverError(f"Failed to create ChromaDB client: {e}", cause=e)

    def _get_collection(self):
        """Get or create the collection."""
        if self._collection is not None:
            return self._collection

        client = self._get_client()

        try:
            # Map distance metric to ChromaDB's space parameter
            space_map = {
                "cosine": "cosine",
                "l2": "l2",
                "ip": "ip",
            }

            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": space_map[self.distance_metric]},
            )
            return self._collection
        except Exception as e:
            raise RetrieverError(f"Failed to get/create collection: {e}", cause=e)

    @property
    def embedder(self) -> "Embedder":
        """Return the embedder."""
        return self._embedder

    @property
    def is_indexed(self) -> bool:
        """Check if a document has been indexed."""
        return len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        """Return number of indexed chunks."""
        return len(self._chunks)

    @property
    def document_id(self) -> str | None:
        """Return ID of indexed document."""
        return self._document_id

    def index(self, document: "Document") -> None:
        """
        Index a document for retrieval.

        Embeds all chunks and stores in ChromaDB. Replaces any existing index.

        Args:
            document: Document to index.

        Raises:
            RetrieverError: If indexing fails.
        """
        # Clear existing data first
        self.clear()

        chunks = document.chunks

        if not chunks:
            self._document_id = document.id
            return

        try:
            collection = self._get_collection()

            # Get texts and embed
            texts = [chunk.text for chunk in chunks]
            vectors = self._embedder.embed_many(texts)

            # Generate unique IDs for each chunk
            ids = [f"{document.id}_{i}" for i in range(len(chunks))]

            # Build metadata for each chunk
            metadatas = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "document_id": document.id,
                    "chunk_index": chunk.index,
                    "text_preview": chunk.text[:200],  # First 200 chars
                }
                if chunk.page is not None:
                    metadata["page"] = chunk.page
                metadatas.append(metadata)

            # Add to collection
            collection.add(
                ids=ids,
                embeddings=[list(v) for v in vectors],
                documents=texts,
                metadatas=metadatas,
            )

            # Store chunks locally for retrieval
            self._chunks = list(chunks)
            self._document_id = document.id

        except RetrieverError:
            raise
        except Exception as e:
            raise RetrieverError(f"Failed to index document: {e}", cause=e)

    def search(self, query: str, k: int = 5) -> SearchResults:
        """
        Search for relevant chunks using similarity search.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            SearchResults with top k matches.

        Raises:
            RetrieverError: If no document indexed or search fails.
        """
        if not self.is_indexed:
            raise RetrieverError("No document indexed. Call index() first.")

        if k < 1:
            raise RetrieverError("k must be >= 1")

        try:
            collection = self._get_collection()

            # Embed query
            query_result = self._embedder.embed(query)
            query_vec = list(query_result.vector)

            # Search
            k_actual = min(k, len(self._chunks))
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=k_actual,
                include=["distances", "documents", "metadatas"],
            )

            # Build SearchResults
            search_results = []
            if results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                distances = results["distances"][0] if results["distances"] else []
                metadatas = results["metadatas"][0] if results["metadatas"] else []

                for rank, (id_, distance, metadata) in enumerate(
                    zip(ids, distances, metadatas)
                ):
                    # Get chunk index from ID or metadata
                    chunk_idx = metadata.get("chunk_index", 0)
                    if chunk_idx < len(self._chunks):
                        chunk = self._chunks[chunk_idx]

                        # Convert distance to similarity score
                        # For cosine: score = 1 - distance (ChromaDB returns distance)
                        # For l2: score = 1 / (1 + distance)
                        # For ip: score = -distance (inner product, higher is better)
                        if self.distance_metric == "cosine":
                            score = 1.0 - distance
                        elif self.distance_metric == "l2":
                            score = 1.0 / (1.0 + distance)
                        else:  # ip
                            score = -distance

                        search_results.append(
                            SearchResult(chunk=chunk, score=score, rank=rank)
                        )

            return SearchResults(
                results=tuple(search_results),
                query=query,
                total_chunks=len(self._chunks),
            )

        except RetrieverError:
            raise
        except Exception as e:
            raise RetrieverError(f"Search failed: {e}", cause=e)

    def clear(self) -> None:
        """Clear the index."""
        self._chunks = []
        self._document_id = None

        if self._collection is not None:
            try:
                client = self._get_client()
                client.delete_collection(self.collection_name)
                self._collection = None
            except Exception:
                pass  # Collection might not exist


__all__ = ["ChromaRetriever"]
