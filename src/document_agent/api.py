"""REST API server using FastAPI."""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from document_agent import __version__
from document_agent.config import settings

app = FastAPI(
    title="Document Agent API",
    description="Graph RAG 기반 문서 질의응답 API",
    version=__version__,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QuestionRequest(BaseModel):
    """Request model for asking questions."""

    question: str = Field(..., description="질문 내용")
    include_sources: bool = Field(False, description="출처 포함 여부")
    stream: bool = Field(False, description="스트리밍 응답 여부")


class QuestionResponse(BaseModel):
    """Response model for questions."""

    answer: str = Field(..., description="답변")
    sources: Optional[dict] = Field(None, description="출처 정보")


class IndexResponse(BaseModel):
    """Response model for indexing."""

    success: bool
    message: str
    doc_id: Optional[str] = None
    chunks_count: int = 0
    entities_count: int = 0
    images_count: int = 0


class SearchRequest(BaseModel):
    """Request model for entity search."""

    query: str = Field(..., description="검색어")
    entity_type: Optional[str] = Field(None, description="엔티티 유형 필터")
    limit: int = Field(10, description="결과 수")


class StatsResponse(BaseModel):
    """Response model for stats."""

    chunks_count: int
    images_count: int


# Lazy-loaded components
_kuzu_store = None
_vector_store = None
_graph_builder = None
_image_processor = None
_generator = None


def get_kuzu_store():
    """Get Kuzu store instance."""
    global _kuzu_store
    if _kuzu_store is None:
        from document_agent.storage import KuzuStore

        settings.ensure_directories()
        _kuzu_store = KuzuStore()
    return _kuzu_store


def get_vector_store():
    """Get vector store instance."""
    global _vector_store
    if _vector_store is None:
        from document_agent.storage import VectorStore

        settings.ensure_directories()
        _vector_store = VectorStore()
    return _vector_store


def get_graph_builder():
    """Get graph builder instance."""
    global _graph_builder
    if _graph_builder is None:
        from document_agent.graph import GraphBuilder

        _graph_builder = GraphBuilder()
    return _graph_builder


def get_image_processor():
    """Get image processor instance."""
    global _image_processor
    if _image_processor is None:
        from document_agent.parsers import ImageProcessor

        _image_processor = ImageProcessor()
    return _image_processor


def get_generator():
    """Get response generator instance."""
    global _generator
    if _generator is None:
        from document_agent.qa import ResponseGenerator

        _generator = ResponseGenerator()
    return _generator


def get_parser_for_file(file_path: Path):
    """Get appropriate parser for a file."""
    from document_agent.parsers import OfficeParser, PDFParser, TextParser

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PDFParser()
    elif suffix in [".docx", ".pptx", ".xlsx", ".xls", ".doc"]:
        return OfficeParser()
    elif suffix in [".md", ".txt", ".markdown", ".text"]:
        return TextParser()
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Document Agent API",
        "version": __version__,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/index", response_model=IndexResponse)
async def index_document(
    file: UploadFile = File(...),
    extract_relations: bool = True,
    process_images: bool = True,
):
    """Index a document file.

    Supports: PDF, DOCX, PPTX, XLSX, MD, TXT
    """
    # Validate file type
    allowed_extensions = [".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".doc", ".md", ".txt"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}",
        )

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Parse document
        parser = get_parser_for_file(tmp_path)
        parsed_doc = parser.parse(tmp_path)

        # Process images if enabled
        if process_images and parsed_doc.images:
            image_processor = get_image_processor()
            parsed_doc.images = image_processor.process_images(parsed_doc.images)

        # Build knowledge graph
        graph_builder = get_graph_builder()
        graph = graph_builder.build_from_document(
            parsed_doc,
            extract_relations=extract_relations,
        )

        # Store in databases
        kuzu_store = get_kuzu_store()
        vector_store = get_vector_store()

        kuzu_store.store_knowledge_graph(graph)
        vector_store.store_knowledge_graph(graph)

        return IndexResponse(
            success=True,
            message=f"문서 '{file.filename}'이(가) 성공적으로 인덱싱되었습니다.",
            doc_id=parsed_doc.doc_id,
            chunks_count=len(graph.chunks),
            entities_count=len(graph.entities),
            images_count=len(graph.images),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"인덱싱 실패: {str(e)}")

    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about indexed documents."""
    generator = get_generator()

    if request.stream:
        # Return streaming response
        async def generate():
            for chunk in generator.stream_generate(request.question):
                yield chunk

        return StreamingResponse(generate(), media_type="text/plain")

    # Regular response
    result = generator.ask(request.question, include_sources=request.include_sources)

    return QuestionResponse(
        answer=result["answer"],
        sources=result.get("sources"),
    )


@app.post("/search")
async def search_entities(request: SearchRequest):
    """Search for entities in the knowledge graph."""
    kuzu_store = get_kuzu_store()

    entities = kuzu_store.search_entities(
        request.query,
        entity_type=request.entity_type,
        limit=request.limit,
    )

    return {"entities": entities}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get index statistics."""
    vector_store = get_vector_store()
    stats = vector_store.get_stats()

    return StatsResponse(
        chunks_count=stats["chunks_count"],
        images_count=stats["images_count"],
    )


@app.get("/entity/{entity_name}/subgraph")
async def get_entity_subgraph(entity_name: str, hops: int = 2):
    """Get subgraph around an entity."""
    kuzu_store = get_kuzu_store()

    subgraph = kuzu_store.get_entity_subgraph([entity_name], hops=hops)

    return subgraph


@app.get("/entity/{entity_name}/chunks")
async def get_entity_chunks(entity_name: str, limit: int = 10):
    """Get chunks that mention an entity."""
    kuzu_store = get_kuzu_store()

    chunks = kuzu_store.get_chunks_for_entities([entity_name], limit=limit)

    return {"chunks": chunks}
