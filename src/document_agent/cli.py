"""CLI interface for Document Agent using Typer."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from document_agent import __version__
from document_agent.config import settings

app = typer.Typer(
    name="docagent",
    help="Document Agent - Graph RAG 기반 문서 질의응답 시스템",
    add_completion=False,
)
console = Console()


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


@app.command()
def index(
    path: Path = typer.Argument(..., help="인덱싱할 파일 또는 디렉토리 경로"),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="디렉토리 재귀 탐색"),
    extract_relations: bool = typer.Option(True, "--relations", help="관계 추출 여부"),
    process_images: bool = typer.Option(True, "--images", help="이미지 처리 여부"),
):
    """문서를 파싱하고 지식 그래프를 구축합니다."""
    from document_agent.graph import GraphBuilder
    from document_agent.parsers import ImageProcessor
    from document_agent.storage import KuzuStore, VectorStore

    # Ensure directories exist
    settings.ensure_directories()

    # Collect files to process
    files = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        patterns = ["*.pdf", "*.docx", "*.pptx", "*.xlsx", "*.xls", "*.doc", "*.md", "*.txt"]
        for pattern in patterns:
            if recursive:
                files.extend(path.rglob(pattern))
            else:
                files.extend(path.glob(pattern))
    else:
        console.print(f"[red]경로를 찾을 수 없습니다: {path}[/red]")
        raise typer.Exit(1)

    if not files:
        console.print("[yellow]인덱싱할 파일이 없습니다.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]총 {len(files)}개 파일을 인덱싱합니다.[/bold]\n")

    # Initialize components
    graph_builder = GraphBuilder()
    kuzu_store = KuzuStore()
    vector_store = VectorStore()
    image_processor = ImageProcessor() if process_images else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for file_path in files:
            task = progress.add_task(f"처리 중: {file_path.name}", total=None)

            try:
                # Parse document
                parser = get_parser_for_file(file_path)
                parsed_doc = parser.parse(file_path)

                # Process images if enabled
                if process_images and image_processor and parsed_doc.images:
                    progress.update(task, description=f"이미지 처리 중: {file_path.name}")
                    parsed_doc.images = image_processor.process_images(parsed_doc.images)

                # Build knowledge graph
                progress.update(task, description=f"그래프 구축 중: {file_path.name}")
                graph = graph_builder.build_from_document(
                    parsed_doc,
                    extract_relations=extract_relations,
                )

                # Store in databases
                progress.update(task, description=f"저장 중: {file_path.name}")
                kuzu_store.store_knowledge_graph(graph)
                vector_store.store_knowledge_graph(graph)

                progress.remove_task(task)
                console.print(f"[green]✓[/green] {file_path.name}")

            except Exception as e:
                progress.remove_task(task)
                console.print(f"[red]✗[/red] {file_path.name}: {str(e)}")

    # Print summary
    stats = vector_store.get_stats()
    console.print(
        Panel(
            f"청크: {stats['chunks_count']}개\n이미지: {stats['images_count']}개",
            title="인덱싱 완료",
            border_style="green",
        )
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="질문"),
    sources: bool = typer.Option(False, "--sources", "-s", help="출처 표시"),
    stream: bool = typer.Option(False, "--stream", help="스트리밍 출력"),
):
    """질문에 대한 답변을 생성합니다."""
    from document_agent.qa import ResponseGenerator

    settings.ensure_directories()

    console.print(f"\n[bold]질문:[/bold] {question}\n")

    generator = ResponseGenerator()

    if stream:
        console.print("[bold]답변:[/bold]")
        for chunk in generator.stream_generate(question):
            console.print(chunk, end="")
        console.print("\n")
    else:
        with console.status("답변 생성 중..."):
            result = generator.ask(question, include_sources=sources)

        console.print(Panel(result["answer"], title="답변", border_style="blue"))

        if sources and "sources" in result:
            # Show sources table
            table = Table(title="출처")
            table.add_column("유형", style="cyan")
            table.add_column("내용")
            table.add_column("관련도", justify="right")

            for chunk in result["sources"].get("chunks", []):
                table.add_row(
                    "청크",
                    chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"],
                    f"{chunk['score']:.2f}",
                )

            for entity in result["sources"].get("entities", []):
                table.add_row("엔티티", f"{entity['name']} ({entity['type']})", "-")

            console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="서버 호스트"),
    port: int = typer.Option(8000, "--port", "-p", help="서버 포트"),
    reload: bool = typer.Option(False, "--reload", help="자동 리로드"),
):
    """REST API 서버를 시작합니다."""
    import uvicorn

    settings.ensure_directories()

    console.print(f"\n[bold]서버 시작:[/bold] http://{host}:{port}\n")
    console.print("API 문서: http://{host}:{port}/docs\n")

    uvicorn.run(
        "document_agent.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def stats():
    """인덱스 통계를 표시합니다."""
    from document_agent.storage import KuzuStore, VectorStore

    settings.ensure_directories()

    kuzu_store = KuzuStore()
    vector_store = VectorStore()

    vector_stats = vector_store.get_stats()

    table = Table(title="인덱스 통계")
    table.add_column("항목", style="cyan")
    table.add_column("개수", justify="right")

    table.add_row("청크", str(vector_stats["chunks_count"]))
    table.add_row("이미지", str(vector_stats["images_count"]))

    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="검색어"),
    limit: int = typer.Option(10, "--limit", "-n", help="결과 수"),
    entity_type: Optional[str] = typer.Option(None, "--type", "-t", help="엔티티 유형 필터"),
):
    """엔티티를 검색합니다."""
    from document_agent.storage import KuzuStore

    settings.ensure_directories()

    kuzu_store = KuzuStore()
    entities = kuzu_store.search_entities(query, entity_type=entity_type, limit=limit)

    if not entities:
        console.print("[yellow]검색 결과가 없습니다.[/yellow]")
        return

    table = Table(title=f"'{query}' 검색 결과")
    table.add_column("이름", style="cyan")
    table.add_column("유형")
    table.add_column("설명")

    for entity in entities:
        import json

        props = entity.get("properties", "{}")
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except json.JSONDecodeError:
                props = {}

        desc = props.get("description", "") if isinstance(props, dict) else ""
        table.add_row(entity["name"], entity["type"], desc[:50])

    console.print(table)


@app.command()
def version():
    """버전 정보를 표시합니다."""
    console.print(f"Document Agent v{__version__}")


if __name__ == "__main__":
    app()
