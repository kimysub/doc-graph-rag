"""Relation extraction module using LLM."""

import json
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from document_agent.config import settings

from .entity import Entity


@dataclass
class Relation:
    """A relation between two entities."""

    id: str
    subject: str  # Entity name
    predicate: str  # Relation type
    object: str  # Entity name
    subject_type: Optional[str] = None
    object_type: Optional[str] = None
    source_chunk_id: Optional[str] = None
    confidence: float = 1.0


class RelationExtractor:
    """LLM-based relation extractor."""

    EXTRACTION_PROMPT = """다음 텍스트와 엔티티 목록을 기반으로 엔티티 간의 관계를 추출해주세요.

엔티티 목록:
{entities}

관계 유형 예시:
- WORKS_FOR: A가 B에서 일함
- LOCATED_IN: A가 B에 위치함
- PART_OF: A가 B의 일부
- CREATED_BY: A가 B에 의해 생성됨
- RELATED_TO: A가 B와 관련됨
- OWNS: A가 B를 소유함
- USES: A가 B를 사용함
- PRODUCES: A가 B를 생산함
- AFFECTS: A가 B에 영향을 줌
- DEPENDS_ON: A가 B에 의존함

다음 JSON 형식으로 응답해주세요:
{{
    "relations": [
        {{
            "subject": "주체 엔티티 이름",
            "predicate": "관계 유형",
            "object": "대상 엔티티 이름"
        }}
    ]
}}

중요: 반드시 유효한 JSON만 응답하세요. 엔티티 목록에 있는 이름만 사용하세요.

텍스트:
{text}
"""

    def __init__(
        self,
        llm_client: Optional[OpenAI] = None,
        model: Optional[str] = None,
    ):
        """Initialize the relation extractor.

        Args:
            llm_client: OpenAI client.
            model: Model to use for extraction.
        """
        self.model = model or settings.llm_model

        if llm_client:
            self.client = llm_client
        else:
            self.client = OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
            )

    def extract_relations(
        self,
        text: str,
        entities: list[Entity],
        chunk_id: Optional[str] = None,
    ) -> list[Relation]:
        """Extract relations between entities from text.

        Args:
            text: Text containing the entities.
            entities: List of entities in the text.
            chunk_id: Optional chunk ID for tracking.

        Returns:
            List of extracted Relation objects.
        """
        if not text.strip() or len(entities) < 2:
            return []

        # Format entities for prompt
        entity_list = "\n".join(
            f"- {e.name} ({e.type})" for e in entities
        )

        prompt = self.EXTRACTION_PROMPT.format(
            entities=entity_list,
            text=text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a relation extraction assistant. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            result_text = response.choices[0].message.content

            # Parse JSON response
            relations = self._parse_response(result_text, entities, chunk_id)
            return relations

        except Exception as e:
            print(f"Relation extraction error: {e}")
            return []

    def extract_from_chunks(
        self,
        chunks: list,
        chunk_entities: dict[str, list[Entity]],
    ) -> list[Relation]:
        """Extract relations from multiple chunks.

        Args:
            chunks: List of Chunk objects.
            chunk_entities: Dict mapping chunk IDs to their entities.

        Returns:
            List of all extracted relations.
        """
        all_relations = []

        for chunk in chunks:
            entities = chunk_entities.get(chunk.id, [])
            if len(entities) >= 2:
                relations = self.extract_relations(
                    chunk.content,
                    entities,
                    chunk.id,
                )
                all_relations.extend(relations)

        return all_relations

    def deduplicate_relations(
        self,
        relations: list[Relation],
    ) -> list[Relation]:
        """Deduplicate relations.

        Args:
            relations: List of relations.

        Returns:
            Deduplicated list of relations.
        """
        seen = {}

        for relation in relations:
            key = (
                relation.subject.lower(),
                relation.predicate.upper(),
                relation.object.lower(),
            )

            if key not in seen:
                seen[key] = relation

        return list(seen.values())

    def _parse_response(
        self,
        response_text: str,
        entities: list[Entity],
        chunk_id: Optional[str],
    ) -> list[Relation]:
        """Parse LLM response into Relation objects.

        Args:
            response_text: Raw LLM response.
            entities: List of entities for validation.
            chunk_id: Chunk ID for tracking.

        Returns:
            List of Relation objects.
        """
        relations = []

        # Create entity lookup
        entity_lookup = {e.name.lower(): e for e in entities}

        try:
            # Try to extract JSON from response
            response_text = response_text.strip()

            # Handle markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            data = json.loads(response_text)

            for rel_data in data.get("relations", []):
                subject = rel_data.get("subject", "").strip()
                predicate = rel_data.get("predicate", "").strip().upper()
                obj = rel_data.get("object", "").strip()

                if not subject or not predicate or not obj:
                    continue

                # Get entity types if available
                subject_entity = entity_lookup.get(subject.lower())
                object_entity = entity_lookup.get(obj.lower())

                relation_id = f"rel_{hash(subject + predicate + obj) % 10000000:07d}"

                relations.append(
                    Relation(
                        id=relation_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        subject_type=subject_entity.type if subject_entity else None,
                        object_type=object_entity.type if object_entity else None,
                        source_chunk_id=chunk_id,
                    )
                )

        except json.JSONDecodeError:
            pass

        return relations
