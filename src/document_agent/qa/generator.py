"""Response generator using LLM."""

from typing import Optional

from openai import OpenAI

from document_agent.config import settings

from .retriever import GraphRetriever, RetrievalResult


class ResponseGenerator:
    """Generate responses using LLM and retrieved context."""

    SYSTEM_PROMPT = """당신은 문서 기반 질의응답 어시스턴트입니다.
주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요.

지침:
1. 컨텍스트에 있는 정보만 사용하여 답변하세요.
2. 컨텍스트에서 답을 찾을 수 없으면 솔직하게 "주어진 문서에서 해당 정보를 찾을 수 없습니다"라고 말하세요.
3. 답변은 명확하고 구조적으로 작성하세요.
4. 가능하면 관련 엔티티와 관계를 언급하여 답변의 근거를 제시하세요.
5. 이미지 설명이 관련 있으면 포함하세요.
"""

    def __init__(
        self,
        retriever: Optional[GraphRetriever] = None,
        llm_client: Optional[OpenAI] = None,
        model: Optional[str] = None,
    ):
        """Initialize the response generator.

        Args:
            retriever: Retriever for getting context.
            llm_client: OpenAI client.
            model: Model to use.
        """
        self.retriever = retriever or GraphRetriever()
        self.model = model or settings.llm_model

        if llm_client:
            self.client = llm_client
        else:
            self.client = OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
            )

    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        retrieval_result: Optional[RetrievalResult] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response to a question.

        Args:
            question: User's question.
            context: Optional pre-computed context. If not provided, retrieval will be done.
            retrieval_result: Optional pre-computed retrieval result.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            Generated response text.
        """
        # Get context if not provided
        if context is None:
            if retrieval_result:
                context = retrieval_result.to_context()
            else:
                context = self.retriever.get_context_for_question(question)

        # Build prompt
        user_message = f"""## 컨텍스트
{context}

## 질문
{question}

위 컨텍스트를 바탕으로 질문에 답변해주세요."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

    def ask(
        self,
        question: str,
        include_sources: bool = False,
    ) -> dict:
        """Ask a question and get a response with optional sources.

        Args:
            question: User's question.
            include_sources: Whether to include source information.

        Returns:
            Dictionary with 'answer' and optionally 'sources'.
        """
        # Retrieve context
        retrieval_result = self.retriever.retrieve(question)

        # Generate response
        answer = self.generate(
            question,
            retrieval_result=retrieval_result,
        )

        result = {"answer": answer}

        if include_sources:
            result["sources"] = {
                "chunks": [
                    {
                        "id": c.get("id"),
                        "content": c.get("content", "")[:200] + "..."
                        if len(c.get("content", "")) > 200
                        else c.get("content", ""),
                        "score": c.get("score", 0),
                    }
                    for c in retrieval_result.chunks[:5]
                ],
                "entities": [
                    {
                        "name": e.get("name"),
                        "type": e.get("type"),
                    }
                    for e in retrieval_result.entities[:10]
                ],
                "relations": retrieval_result.relations[:5],
            }

        return result

    def stream_generate(
        self,
        question: str,
        context: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        """Generate a streaming response.

        Args:
            question: User's question.
            context: Optional pre-computed context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Yields:
            Response chunks.
        """
        # Get context if not provided
        if context is None:
            context = self.retriever.get_context_for_question(question)

        user_message = f"""## 컨텍스트
{context}

## 질문
{question}

위 컨텍스트를 바탕으로 질문에 답변해주세요."""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"응답 생성 중 오류가 발생했습니다: {str(e)}"
