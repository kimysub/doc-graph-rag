"""ChromaDB vector store for semantic search."""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from document_agent.config import settings
from document_agent.graph import Chunk, KnowledgeGraph
from document_agent.llm import LLMClient
from document_agent.parsers import ImageInfo


class VectorStore:
    """Vector store using ChromaDB for semantic search."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        collection_name: str = "document_chunks",
    ):
        """Initialize ChromaDB vector store.

        Args:
            db_path: Path to the database directory.
            collection_name: Name of the collection.
        """
        self.db_path = Path(db_path or settings.chroma_db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Image descriptions collection
        self.image_collection = self.client.get_or_create_collection(
            name=f"{collection_name}_images",
            metadata={"hnsw:space": "cosine"},
        )

        # LLM client for embeddings
        self._llm_client: Optional[LLMClient] = None

    @property
    def llm_client(self) -> LLMClient:
        """Lazy-load LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def store_knowledge_graph(self, graph: KnowledgeGraph) -> None:
        """Store a knowledge graph in the vector store.

        Args:
            graph: KnowledgeGraph to store.
        """
        # Store chunks
        if graph.chunks:
            self.store_chunks(graph.chunks)

        # Store image descriptions
        if graph.images:
            self.store_images(graph.images)

    def store_chunks(self, chunks: list[Chunk]) -> None:
        """Store chunks with their embeddings.

        Args:
            chunks: List of chunks to store.
        """
        if not chunks:
            return

        # Prepare data
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "page_num": chunk.page_num or 0,
                "position": chunk.position,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        # Generate embeddings
        embeddings = self.llm_client.embed(documents)

        # Upsert to collection
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def store_images(self, images: list[ImageInfo]) -> None:
        """Store image descriptions with their embeddings.

        Args:
            images: List of images to store.
        """
        # Filter images with descriptions
        images_with_desc = [img for img in images if img.description]

        if not images_with_desc:
            return

        ids = [img.id for img in images_with_desc]
        documents = [img.description for img in images_with_desc]
        metadatas = [
            {
                "path": str(img.path),
                "page_num": img.page_num or 0,
            }
            for img in images_with_desc
        ]

        # Generate embeddings
        embeddings = self.llm_client.embed(documents)

        # Upsert to collection
        self.image_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_doc_id: Optional[str] = None,
    ) -> list[dict]:
        """Search for similar chunks.

        Args:
            query: Search query.
            n_results: Number of results to return.
            filter_doc_id: Optional document ID to filter by.

        Returns:
            List of matching chunks with scores.
        """
        # Generate query embedding
        query_embedding = self.llm_client.embed([query])[0]

        # Build filter
        where_filter = None
        if filter_doc_id:
            where_filter = {"doc_id": filter_doc_id}

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                chunks.append({
                    "id": chunk_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0,
                })

        return chunks

    def search_images(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Search for similar image descriptions.

        Args:
            query: Search query.
            n_results: Number of results to return.

        Returns:
            List of matching images with scores.
        """
        # Generate query embedding
        query_embedding = self.llm_client.embed([query])[0]

        # Search
        results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        images = []
        if results["ids"] and results["ids"][0]:
            for i, image_id in enumerate(results["ids"][0]):
                images.append({
                    "id": image_id,
                    "description": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0,
                })

        return images

    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        include_images: bool = True,
    ) -> dict:
        """Perform hybrid search across chunks and images.

        Args:
            query: Search query.
            n_results: Number of results per type.
            include_images: Whether to include image results.

        Returns:
            Dictionary with 'chunks' and optionally 'images' results.
        """
        results = {
            "chunks": self.search(query, n_results),
        }

        if include_images:
            results["images"] = self.search_images(query, n_results // 2)

        return results

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks for a document.

        Args:
            doc_id: Document ID to delete.
        """
        # Get chunk IDs for document
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=[],
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def get_stats(self) -> dict:
        """Get collection statistics.

        Returns:
            Dictionary with collection stats.
        """
        return {
            "chunks_count": self.collection.count(),
            "images_count": self.image_collection.count(),
        }
