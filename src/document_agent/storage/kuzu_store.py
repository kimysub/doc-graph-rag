"""Kuzu graph database storage."""

from pathlib import Path
from typing import Any, Optional

import kuzu

from document_agent.config import settings
from document_agent.graph import Chunk, Entity, KnowledgeGraph, Relation
from document_agent.parsers import ImageInfo


class KuzuStore:
    """Storage backend using Kuzu graph database."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize Kuzu store.

        Args:
            db_path: Path to the database directory.
        """
        self.db_path = Path(db_path or settings.kuzu_db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)

        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create the graph schema if it doesn't exist."""
        # Node tables
        node_schemas = [
            """
            CREATE NODE TABLE IF NOT EXISTS Document(
                id STRING PRIMARY KEY,
                name STRING,
                path STRING,
                type STRING,
                metadata STRING
            )
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS Chunk(
                id STRING PRIMARY KEY,
                content STRING,
                doc_id STRING,
                page_num INT64,
                position INT64,
                token_count INT64
            )
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS Entity(
                id STRING PRIMARY KEY,
                name STRING,
                type STRING,
                properties STRING
            )
            """,
            """
            CREATE NODE TABLE IF NOT EXISTS Image(
                id STRING PRIMARY KEY,
                path STRING,
                description STRING,
                doc_id STRING,
                page_num INT64
            )
            """,
        ]

        # Relationship tables
        rel_schemas = [
            """
            CREATE REL TABLE IF NOT EXISTS CONTAINS(
                FROM Document TO Chunk,
                MANY_MANY
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS HAS_IMAGE(
                FROM Document TO Image,
                MANY_MANY
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS MENTIONS(
                FROM Chunk TO Entity,
                MANY_MANY
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS RELATED_TO(
                FROM Entity TO Entity,
                predicate STRING,
                source_chunk_id STRING,
                MANY_MANY
            )
            """,
            """
            CREATE REL TABLE IF NOT EXISTS DEPICTS(
                FROM Image TO Entity,
                MANY_MANY
            )
            """,
        ]

        for schema in node_schemas + rel_schemas:
            try:
                self.conn.execute(schema)
            except Exception:
                # Schema might already exist
                pass

    def store_knowledge_graph(self, graph: KnowledgeGraph) -> None:
        """Store a knowledge graph in the database.

        Args:
            graph: KnowledgeGraph to store.
        """
        import json

        # Store documents
        for doc in graph.documents:
            self._upsert_document(doc)

        # Store chunks and create CONTAINS relationships
        for chunk in graph.chunks:
            self._upsert_chunk(chunk)
            self._create_contains_relationship(chunk.doc_id, chunk.id)

        # Store entities
        for entity in graph.entities:
            self._upsert_entity(entity)

        # Create MENTIONS relationships (chunk -> entity)
        chunk_entity_map = self._build_chunk_entity_map(graph.chunks, graph.entities)
        for chunk_id, entity_ids in chunk_entity_map.items():
            for entity_id in entity_ids:
                self._create_mentions_relationship(chunk_id, entity_id)

        # Store relations as RELATED_TO relationships
        for relation in graph.relations:
            self._create_related_to_relationship(relation)

        # Store images and create HAS_IMAGE relationships
        for image in graph.images:
            self._upsert_image(image, graph.documents[0]["id"] if graph.documents else "")

    def _upsert_document(self, doc: dict) -> None:
        """Insert or update a document."""
        import json

        query = """
        MERGE (d:Document {id: $id})
        SET d.name = $name, d.path = $path, d.type = $type, d.metadata = $metadata
        """
        self.conn.execute(
            query,
            {
                "id": doc["id"],
                "name": doc.get("name", ""),
                "path": doc.get("path", ""),
                "type": doc.get("type", ""),
                "metadata": json.dumps(doc.get("metadata", {})),
            },
        )

    def _upsert_chunk(self, chunk: Chunk) -> None:
        """Insert or update a chunk."""
        query = """
        MERGE (c:Chunk {id: $id})
        SET c.content = $content, c.doc_id = $doc_id, 
            c.page_num = $page_num, c.position = $position, c.token_count = $token_count
        """
        self.conn.execute(
            query,
            {
                "id": chunk.id,
                "content": chunk.content,
                "doc_id": chunk.doc_id,
                "page_num": chunk.page_num or 0,
                "position": chunk.position,
                "token_count": chunk.token_count,
            },
        )

    def _upsert_entity(self, entity: Entity) -> None:
        """Insert or update an entity."""
        import json

        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name, e.type = $type, e.properties = $properties
        """
        self.conn.execute(
            query,
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "properties": json.dumps(entity.properties),
            },
        )

    def _upsert_image(self, image: ImageInfo, doc_id: str) -> None:
        """Insert or update an image."""
        query = """
        MERGE (i:Image {id: $id})
        SET i.path = $path, i.description = $description, 
            i.doc_id = $doc_id, i.page_num = $page_num
        """
        self.conn.execute(
            query,
            {
                "id": image.id,
                "path": str(image.path),
                "description": image.description or "",
                "doc_id": doc_id,
                "page_num": image.page_num or 0,
            },
        )

        # Create HAS_IMAGE relationship
        if doc_id:
            rel_query = """
            MATCH (d:Document {id: $doc_id}), (i:Image {id: $image_id})
            MERGE (d)-[:HAS_IMAGE]->(i)
            """
            try:
                self.conn.execute(
                    rel_query,
                    {"doc_id": doc_id, "image_id": image.id},
                )
            except Exception:
                pass

    def _create_contains_relationship(self, doc_id: str, chunk_id: str) -> None:
        """Create a CONTAINS relationship between document and chunk."""
        query = """
        MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id})
        MERGE (d)-[:CONTAINS]->(c)
        """
        try:
            self.conn.execute(query, {"doc_id": doc_id, "chunk_id": chunk_id})
        except Exception:
            pass

    def _create_mentions_relationship(self, chunk_id: str, entity_id: str) -> None:
        """Create a MENTIONS relationship between chunk and entity."""
        query = """
        MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $entity_id})
        MERGE (c)-[:MENTIONS]->(e)
        """
        try:
            self.conn.execute(query, {"chunk_id": chunk_id, "entity_id": entity_id})
        except Exception:
            pass

    def _create_related_to_relationship(self, relation: Relation) -> None:
        """Create a RELATED_TO relationship between entities."""
        query = """
        MATCH (e1:Entity), (e2:Entity)
        WHERE e1.name = $subject AND e2.name = $object
        MERGE (e1)-[r:RELATED_TO]->(e2)
        SET r.predicate = $predicate, r.source_chunk_id = $source_chunk_id
        """
        try:
            self.conn.execute(
                query,
                {
                    "subject": relation.subject,
                    "object": relation.object,
                    "predicate": relation.predicate,
                    "source_chunk_id": relation.source_chunk_id or "",
                },
            )
        except Exception:
            pass

    def _build_chunk_entity_map(
        self,
        chunks: list[Chunk],
        entities: list[Entity],
    ) -> dict[str, list[str]]:
        """Build a map of chunk IDs to entity IDs mentioned in them."""
        result = {}

        for chunk in chunks:
            chunk_text = chunk.content.lower()
            entity_ids = []

            for entity in entities:
                if entity.name.lower() in chunk_text:
                    entity_ids.append(entity.id)

            if entity_ids:
                result[chunk.id] = entity_ids

        return result

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search for entities by name.

        Args:
            query: Search query.
            entity_type: Optional entity type filter.
            limit: Maximum number of results.

        Returns:
            List of entity dictionaries.
        """
        if entity_type:
            cypher = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $query AND e.type = $type
            RETURN e.id, e.name, e.type, e.properties
            LIMIT $limit
            """
            result = self.conn.execute(
                cypher,
                {"query": query, "type": entity_type, "limit": limit},
            )
        else:
            cypher = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $query
            RETURN e.id, e.name, e.type, e.properties
            LIMIT $limit
            """
            result = self.conn.execute(cypher, {"query": query, "limit": limit})

        entities = []
        while result.has_next():
            row = result.get_next()
            entities.append({
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "properties": row[3],
            })

        return entities

    def get_entity_subgraph(
        self,
        entity_names: list[str],
        hops: int = 2,
    ) -> dict[str, Any]:
        """Get a subgraph around specified entities.

        Args:
            entity_names: Names of entities to start from.
            hops: Number of relationship hops to traverse.

        Returns:
            Subgraph containing entities and relations.
        """
        entities = []
        relations = []

        for name in entity_names:
            # Get the entity and its neighbors
            cypher = """
            MATCH (e:Entity)
            WHERE e.name = $name
            OPTIONAL MATCH (e)-[r:RELATED_TO]-(other:Entity)
            RETURN e.id, e.name, e.type, e.properties,
                   r.predicate, other.id, other.name, other.type
            """
            result = self.conn.execute(cypher, {"name": name})

            while result.has_next():
                row = result.get_next()
                entities.append({
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "properties": row[3],
                })

                if row[4]:  # Has relation
                    relations.append({
                        "subject": row[1],
                        "predicate": row[4],
                        "object": row[6],
                    })

                    if row[5]:  # Has other entity
                        entities.append({
                            "id": row[5],
                            "name": row[6],
                            "type": row[7],
                        })

        # Deduplicate
        seen_entities = {}
        for e in entities:
            seen_entities[e["id"]] = e

        seen_relations = set()
        unique_relations = []
        for r in relations:
            key = (r["subject"], r["predicate"], r["object"])
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(r)

        return {
            "entities": list(seen_entities.values()),
            "relations": unique_relations,
        }

    def get_chunks_for_entities(
        self,
        entity_names: list[str],
        limit: int = 10,
    ) -> list[dict]:
        """Get chunks that mention specified entities.

        Args:
            entity_names: Entity names to search for.
            limit: Maximum number of chunks.

        Returns:
            List of chunk dictionaries.
        """
        chunks = []

        for name in entity_names:
            cypher = """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.name = $name
            RETURN c.id, c.content, c.doc_id, c.page_num
            LIMIT $limit
            """
            result = self.conn.execute(cypher, {"name": name, "limit": limit})

            while result.has_next():
                row = result.get_next()
                chunks.append({
                    "id": row[0],
                    "content": row[1],
                    "doc_id": row[2],
                    "page_num": row[3],
                })

        return chunks

    def close(self) -> None:
        """Close the database connection."""
        # Kuzu automatically handles connection cleanup
        pass
