"""
PostgreSQL pgvector retriever implementation.

Uses PostgreSQL with the pgvector extension for scalable vector similarity search.
Suitable for production deployments with large document collections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .base import Retriever, SearchResult, SearchResults, RetrieverError

if TYPE_CHECKING:
    from ..core.chunk import Chunk
    from ..core.document import Document
    from ..embeddings.base import Embedder


@dataclass
class PgVectorRetriever(Retriever):
    """
    PostgreSQL pgvector-based retriever for scalable vector search.

    Uses PostgreSQL with the pgvector extension for efficient similarity search.
    Supports cosine distance, L2 distance, and inner product.

    Requirements:
        - PostgreSQL with pgvector extension
        - psycopg2 or psycopg package

    Attributes:
        _embedder: Embedder for converting text to vectors.
        connection_string: PostgreSQL connection string.
        table_name: Name of the table to store vectors.
        distance_metric: Distance metric ('cosine', 'l2', or 'ip').

    Example:
        >>> from pullcite.embeddings.openai import OpenAIEmbedder
        >>> embedder = OpenAIEmbedder()
        >>> retriever = PgVectorRetriever(
        ...     embedder,
        ...     connection_string="postgresql://user:pass@localhost/db",
        ... )
        >>> retriever.index(document)
        >>> results = retriever.search("deductible amount", k=3)
    """

    _embedder: "Embedder"
    connection_string: str = "postgresql://localhost/pullcite"
    table_name: str = "pullcite_chunks"
    distance_metric: str = "cosine"
    _connection: Any = field(default=None, repr=False, compare=False)
    _chunks: list["Chunk"] = field(default_factory=list, repr=False)
    _document_id: str | None = field(default=None, repr=False)
    _table_initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.distance_metric not in ("cosine", "l2", "ip"):
            raise RetrieverError(
                f"Invalid distance_metric: {self.distance_metric}. "
                "Must be 'cosine', 'l2', or 'ip'."
            )

    def _get_connection(self):
        """Get or create database connection."""
        if self._connection is not None:
            return self._connection

        try:
            import psycopg2
        except ImportError:
            try:
                import psycopg as psycopg2
            except ImportError:
                raise RetrieverError(
                    "psycopg2 or psycopg package required. "
                    "Install with: pip install psycopg2-binary or pip install psycopg"
                )

        try:
            self._connection = psycopg2.connect(self.connection_string)
            return self._connection
        except Exception as e:
            raise RetrieverError(f"Failed to connect to PostgreSQL: {e}", cause=e)

    def _ensure_table(self) -> None:
        """Create table and extension if they don't exist."""
        if self._table_initialized:
            return

        conn = self._get_connection()
        dimensions = self._embedder.dimensions

        try:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create table for chunks
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        page INTEGER,
                        text TEXT NOT NULL,
                        embedding vector({dimensions}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                # Create index based on distance metric
                index_name = f"{self.table_name}_embedding_idx"

                # Drop existing index if it exists
                cur.execute(f"DROP INDEX IF EXISTS {index_name}")

                # Create appropriate index
                if self.distance_metric == "cosine":
                    cur.execute(
                        f"""
                        CREATE INDEX {index_name} ON {self.table_name}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                        """
                    )
                elif self.distance_metric == "l2":
                    cur.execute(
                        f"""
                        CREATE INDEX {index_name} ON {self.table_name}
                        USING ivfflat (embedding vector_l2_ops)
                        WITH (lists = 100)
                        """
                    )
                else:  # ip (inner product)
                    cur.execute(
                        f"""
                        CREATE INDEX {index_name} ON {self.table_name}
                        USING ivfflat (embedding vector_ip_ops)
                        WITH (lists = 100)
                        """
                    )

                conn.commit()
                self._table_initialized = True

        except Exception as e:
            conn.rollback()
            raise RetrieverError(f"Failed to initialize table: {e}", cause=e)

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

        Embeds all chunks and stores in PostgreSQL. Replaces any existing index
        for this document.

        Args:
            document: Document to index.

        Raises:
            RetrieverError: If indexing fails.
        """
        self._ensure_table()

        # Clear existing data for this document
        self.clear()

        chunks = document.chunks

        if not chunks:
            self._document_id = document.id
            return

        try:
            conn = self._get_connection()

            # Get texts and embed
            texts = [chunk.text for chunk in chunks]
            vectors = self._embedder.embed_many(texts)

            # Insert chunks
            with conn.cursor() as cur:
                for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                    cur.execute(
                        f"""
                        INSERT INTO {self.table_name}
                        (document_id, chunk_index, page, text, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            document.id,
                            chunk.index,
                            chunk.page,
                            chunk.text,
                            list(vector),
                        ),
                    )

                conn.commit()

            # Store chunks locally for retrieval
            self._chunks = list(chunks)
            self._document_id = document.id

        except RetrieverError:
            raise
        except Exception as e:
            if self._connection:
                self._connection.rollback()
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

        self._ensure_table()

        try:
            conn = self._get_connection()

            # Embed query
            query_result = self._embedder.embed(query)
            query_vec = list(query_result.vector)

            # Build query based on distance metric
            if self.distance_metric == "cosine":
                distance_op = "<=>"
                score_expr = "1 - (embedding <=> %s::vector)"
            elif self.distance_metric == "l2":
                distance_op = "<->"
                score_expr = "1 / (1 + (embedding <-> %s::vector))"
            else:  # ip
                distance_op = "<#>"
                score_expr = "-(embedding <#> %s::vector)"

            k_actual = min(k, len(self._chunks))

            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT chunk_index, {score_expr} as score
                    FROM {self.table_name}
                    WHERE document_id = %s
                    ORDER BY embedding {distance_op} %s::vector
                    LIMIT %s
                    """,
                    (query_vec, self._document_id, query_vec, k_actual),
                )

                rows = cur.fetchall()

            # Build SearchResults
            search_results = []
            for rank, (chunk_idx, score) in enumerate(rows):
                if chunk_idx < len(self._chunks):
                    chunk = self._chunks[chunk_idx]
                    search_results.append(
                        SearchResult(chunk=chunk, score=float(score), rank=rank)
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
        """Clear the index for the current document."""
        if self._document_id and self._connection:
            try:
                conn = self._get_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self.table_name} WHERE document_id = %s",
                        (self._document_id,),
                    )
                    conn.commit()
            except Exception:
                pass  # Table might not exist yet

        self._chunks = []
        self._document_id = None

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __del__(self) -> None:
        """Clean up on destruction."""
        self.close()


__all__ = ["PgVectorRetriever"]
