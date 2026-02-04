"""Entity extraction module using LLM."""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from document_agent.config import settings


@dataclass
class Entity:
    """An extracted entity."""

    id: str
    name: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    source_chunk_id: Optional[str] = None


class EntityExtractor:
    """LLM-based entity extractor."""

    EXTRACTION_PROMPT = """다음 텍스트에서 주요 엔티티(개체)를 추출해주세요.

엔티티 유형:
- PERSON: 인물, 사람 이름
- ORGANIZATION: 조직, 회사, 기관
- LOCATION: 장소, 지역, 국가
- DATE: 날짜, 시간, 기간
- CONCEPT: 주요 개념, 용어, 기술
- PRODUCT: 제품, 서비스
- EVENT: 이벤트, 사건

다음 JSON 형식으로 응답해주세요:
{
    "entities": [
        {
            "name": "엔티티 이름",
            "type": "엔티티 유형",
            "properties": {"description": "간단한 설명"}
        }
    ]
}

중요: 반드시 유효한 JSON만 응답하세요. 다른 텍스트는 포함하지 마세요.

텍스트:
"""

    def __init__(
        self,
        llm_client: Optional[OpenAI] = None,
        model: Optional[str] = None,
    ):
        """Initialize the entity extractor.

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

    def extract_entities(
        self,
        text: str,
        chunk_id: Optional[str] = None,
    ) -> list[Entity]:
        """Extract entities from text using LLM.

        Args:
            text: Text to extract entities from.
            chunk_id: Optional chunk ID for tracking.

        Returns:
            List of extracted Entity objects.
        """
        if not text.strip():
            return []

        prompt = self.EXTRACTION_PROMPT + text

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an entity extraction assistant. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            result_text = response.choices[0].message.content

            # Parse JSON response
            entities = self._parse_response(result_text, chunk_id)
            return entities

        except Exception as e:
            # Return empty list on error
            print(f"Entity extraction error: {e}")
            return []

    def extract_from_chunks(
        self,
        chunks: list,
    ) -> list[Entity]:
        """Extract entities from multiple chunks.

        Args:
            chunks: List of Chunk objects.

        Returns:
            List of all extracted entities (may contain duplicates).
        """
        all_entities = []

        for chunk in chunks:
            entities = self.extract_entities(chunk.content, chunk.id)
            all_entities.extend(entities)

        return all_entities

    def deduplicate_entities(
        self,
        entities: list[Entity],
    ) -> list[Entity]:
        """Deduplicate entities by name and type.

        Args:
            entities: List of entities.

        Returns:
            Deduplicated list of entities.
        """
        seen = {}

        for entity in entities:
            key = (entity.name.lower(), entity.type)

            if key not in seen:
                seen[key] = entity
            else:
                # Merge properties
                existing = seen[key]
                existing.properties.update(entity.properties)

        return list(seen.values())

    def _parse_response(
        self,
        response_text: str,
        chunk_id: Optional[str],
    ) -> list[Entity]:
        """Parse LLM response into Entity objects.

        Args:
            response_text: Raw LLM response.
            chunk_id: Chunk ID for tracking.

        Returns:
            List of Entity objects.
        """
        entities = []

        try:
            # Try to extract JSON from response
            response_text = response_text.strip()

            # Handle markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            data = json.loads(response_text)

            for i, entity_data in enumerate(data.get("entities", [])):
                name = entity_data.get("name", "").strip()
                entity_type = entity_data.get("type", "CONCEPT").upper()
                properties = entity_data.get("properties", {})

                if not name:
                    continue

                entity_id = f"entity_{hash(name + entity_type) % 10000000:07d}"

                entities.append(
                    Entity(
                        id=entity_id,
                        name=name,
                        type=entity_type,
                        properties=properties,
                        source_chunk_id=chunk_id,
                    )
                )

        except json.JSONDecodeError:
            # Try to extract entities manually if JSON parsing fails
            pass

        return entities
