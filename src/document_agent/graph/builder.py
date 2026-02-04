"""Graph builder for constructing knowledge graph from documents."""

from dataclasses import dataclass, field
from typing import Any, Optional

from document_agent.parsers import ImageInfo, ParsedDocument

from .chunker import Chunk, TextChunker
from .entity import Entity, EntityExtractor
from .relation import Relation, RelationExtractor


@dataclass
class KnowledgeGraph:
    """A knowledge graph built from documents."""

    documents: list[dict] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert knowledge graph to dictionary.

        Returns:
            Dictionary representation of the graph.
        """
        return {
            "documents": self.documents,
            "chunks": [
                {
                    "id": c.id,
                    "content": c.content,
                    "doc_id": c.doc_id,
                    "page_num": c.page_num,
                    "position": c.position,
                    "token_count": c.token_count,
                }
                for c in self.chunks
            ],
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "properties": e.properties,
                }
                for e in self.entities
            ],
            "relations": [
                {
                    "id": r.id,
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "subject_type": r.subject_type,
                    "object_type": r.object_type,
                }
                for r in self.relations
            ],
            "images": [
                {
                    "id": img.id,
                    "path": str(img.path),
                    "description": img.description,
                    "page_num": img.page_num,
                }
                for img in self.images
            ],
        }


class GraphBuilder:
    """Builder for constructing knowledge graphs from parsed documents."""

    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        relation_extractor: Optional[RelationExtractor] = None,
    ):
        """Initialize the graph builder.

        Args:
            chunker: Text chunker instance.
            entity_extractor: Entity extractor instance.
            relation_extractor: Relation extractor instance.
        """
        self.chunker = chunker or TextChunker()
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.relation_extractor = relation_extractor or RelationExtractor()

    def build_from_document(
        self,
        parsed_doc: ParsedDocument,
        extract_relations: bool = True,
    ) -> KnowledgeGraph:
        """Build a knowledge graph from a parsed document.

        Args:
            parsed_doc: Parsed document.
            extract_relations: Whether to extract relations.

        Returns:
            KnowledgeGraph containing all extracted information.
        """
        # Create document entry
        doc_entry = {
            "id": parsed_doc.doc_id,
            "name": parsed_doc.name,
            "path": str(parsed_doc.path),
            "type": parsed_doc.doc_type,
            "metadata": parsed_doc.metadata,
        }

        # Chunk the document
        chunks = self.chunker.chunk_document(
            parsed_doc.content,
            parsed_doc.doc_id,
            parsed_doc.pages if parsed_doc.pages else None,
        )

        # Extract entities from each chunk
        chunk_entities: dict[str, list[Entity]] = {}
        all_entities = []

        for chunk in chunks:
            entities = self.entity_extractor.extract_entities(
                chunk.content,
                chunk.id,
            )
            chunk_entities[chunk.id] = entities
            all_entities.extend(entities)

        # Deduplicate entities
        unique_entities = self.entity_extractor.deduplicate_entities(all_entities)

        # Extract relations if requested
        relations = []
        if extract_relations and len(unique_entities) >= 2:
            relations = self.relation_extractor.extract_from_chunks(
                chunks,
                chunk_entities,
            )
            relations = self.relation_extractor.deduplicate_relations(relations)

        return KnowledgeGraph(
            documents=[doc_entry],
            chunks=chunks,
            entities=unique_entities,
            relations=relations,
            images=parsed_doc.images,
        )

    def build_from_documents(
        self,
        parsed_docs: list[ParsedDocument],
        extract_relations: bool = True,
    ) -> KnowledgeGraph:
        """Build a knowledge graph from multiple documents.

        Args:
            parsed_docs: List of parsed documents.
            extract_relations: Whether to extract relations.

        Returns:
            Combined KnowledgeGraph.
        """
        combined = KnowledgeGraph()

        for doc in parsed_docs:
            graph = self.build_from_document(doc, extract_relations)

            combined.documents.extend(graph.documents)
            combined.chunks.extend(graph.chunks)
            combined.entities.extend(graph.entities)
            combined.relations.extend(graph.relations)
            combined.images.extend(graph.images)

        # Final deduplication of entities across documents
        combined.entities = self.entity_extractor.deduplicate_entities(
            combined.entities
        )

        return combined

    def merge_graphs(
        self,
        *graphs: KnowledgeGraph,
    ) -> KnowledgeGraph:
        """Merge multiple knowledge graphs.

        Args:
            *graphs: Knowledge graphs to merge.

        Returns:
            Merged KnowledgeGraph.
        """
        merged = KnowledgeGraph()

        for graph in graphs:
            merged.documents.extend(graph.documents)
            merged.chunks.extend(graph.chunks)
            merged.entities.extend(graph.entities)
            merged.relations.extend(graph.relations)
            merged.images.extend(graph.images)

        # Deduplicate
        merged.entities = self.entity_extractor.deduplicate_entities(merged.entities)
        merged.relations = self.relation_extractor.deduplicate_relations(merged.relations)

        return merged
