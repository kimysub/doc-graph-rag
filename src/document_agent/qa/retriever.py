"""Hybrid retriever combining graph and vector search."""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from document_agent.graph import EntityExtractor
from document_agent.storage import KuzuStore, VectorStore


@dataclass
class RetrievalResult:
    """Result from retrieval."""

    chunks: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    subgraph: dict = field(default_factory=dict)

    def to_context(self) -> str:
        """Convert retrieval result to context string for LLM.

        Returns:
            Formatted context string.
        """
        context_parts = []

        # Add relevant chunks
        if self.chunks:
            context_parts.append("## 관련 문서 내용\n")
            for i, chunk in enumerate(self.chunks[:5], 1):
                content = chunk.get("content", "")
                score = chunk.get("score", 0)
                context_parts.append(f"### 발췌 {i} (관련도: {score:.2f})\n{content}\n")

        # Add entity information
        if self.entities:
            context_parts.append("\n## 관련 엔티티\n")
            for entity in self.entities[:10]:
                name = entity.get("name", "")
                etype = entity.get("type", "")
                props = entity.get("properties", "")
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except json.JSONDecodeError:
                        props = {}
                desc = props.get("description", "") if isinstance(props, dict) else ""
                context_parts.append(f"- **{name}** ({etype}): {desc}")

        # Add relationships
        if self.relations:
            context_parts.append("\n## 관련 관계\n")
            for rel in self.relations[:10]:
                subj = rel.get("subject", "")
                pred = rel.get("predicate", "")
                obj = rel.get("object", "")
                context_parts.append(f"- {subj} --[{pred}]--> {obj}")

        # Add image descriptions
        if self.images:
            context_parts.append("\n## 관련 이미지\n")
            for img in self.images[:3]:
                desc = img.get("description", "")
                context_parts.append(f"- {desc}")

        return "\n".join(context_parts)


class GraphRetriever:
    """Hybrid retriever combining graph and vector search."""

    def __init__(
        self,
        kuzu_store: Optional[KuzuStore] = None,
        vector_store: Optional[VectorStore] = None,
        entity_extractor: Optional[EntityExtractor] = None,
    ):
        """Initialize the retriever.

        Args:
            kuzu_store: Kuzu graph store.
            vector_store: ChromaDB vector store.
            entity_extractor: Entity extractor for query analysis.
        """
        self.kuzu_store = kuzu_store or KuzuStore()
        self.vector_store = vector_store or VectorStore()
        self.entity_extractor = entity_extractor or EntityExtractor()

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        include_images: bool = True,
        use_graph: bool = True,
    ) -> RetrievalResult:
        """Retrieve relevant context for a query.

        Args:
            query: User's question.
            n_results: Number of results to retrieve.
            include_images: Whether to include image results.
            use_graph: Whether to use graph-based retrieval.

        Returns:
            RetrievalResult containing all retrieved information.
        """
        result = RetrievalResult()

        # Step 1: Vector search for relevant chunks
        vector_results = self.vector_store.hybrid_search(
            query,
            n_results=n_results,
            include_images=include_images,
        )
        result.chunks = vector_results.get("chunks", [])
        result.images = vector_results.get("images", [])

        # Step 2: Extract entities from query
        if use_graph:
            query_entities = self._extract_query_entities(query)

            if query_entities:
                # Step 3: Search for entities in graph
                entity_names = [e.name for e in query_entities]

                # Get entity information
                for name in entity_names:
                    entities = self.kuzu_store.search_entities(name)
                    result.entities.extend(entities)

                # Get subgraph around entities
                subgraph = self.kuzu_store.get_entity_subgraph(entity_names)
                result.subgraph = subgraph
                result.relations = subgraph.get("relations", [])

                # Get additional chunks that mention these entities
                graph_chunks = self.kuzu_store.get_chunks_for_entities(
                    entity_names,
                    limit=n_results // 2,
                )

                # Merge with vector search results
                existing_ids = {c["id"] for c in result.chunks}
                for chunk in graph_chunks:
                    if chunk["id"] not in existing_ids:
                        chunk["score"] = 0.5  # Default score for graph results
                        result.chunks.append(chunk)

        # Step 4: Re-rank and deduplicate
        result.chunks = self._deduplicate_chunks(result.chunks)
        result.entities = self._deduplicate_entities(result.entities)

        return result

    def _extract_query_entities(self, query: str) -> list:
        """Extract potential entities from the query.

        Args:
            query: User's query.

        Returns:
            List of extracted entities.
        """
        try:
            entities = self.entity_extractor.extract_entities(query)
            return entities
        except Exception:
            return []

    def _deduplicate_chunks(self, chunks: list[dict]) -> list[dict]:
        """Deduplicate chunks by ID and sort by score.

        Args:
            chunks: List of chunks.

        Returns:
            Deduplicated and sorted list.
        """
        seen = {}
        for chunk in chunks:
            chunk_id = chunk.get("id")
            if chunk_id not in seen:
                seen[chunk_id] = chunk
            else:
                # Keep the one with higher score
                if chunk.get("score", 0) > seen[chunk_id].get("score", 0):
                    seen[chunk_id] = chunk

        # Sort by score descending
        sorted_chunks = sorted(
            seen.values(),
            key=lambda x: x.get("score", 0),
            reverse=True,
        )

        return sorted_chunks

    def _deduplicate_entities(self, entities: list[dict]) -> list[dict]:
        """Deduplicate entities by name.

        Args:
            entities: List of entities.

        Returns:
            Deduplicated list.
        """
        seen = {}
        for entity in entities:
            name = entity.get("name", "").lower()
            if name not in seen:
                seen[name] = entity

        return list(seen.values())

    def get_context_for_question(
        self,
        question: str,
        max_tokens: int = 4000,
    ) -> str:
        """Get formatted context for a question.

        Args:
            question: User's question.
            max_tokens: Maximum tokens for context.

        Returns:
            Formatted context string.
        """
        result = self.retrieve(question)
        context = result.to_context()

        # Truncate if too long (rough estimate: 4 chars per token)
        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[컨텍스트가 잘림...]"

        return context
