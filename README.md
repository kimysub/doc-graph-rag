# Document Agent

A Graph RAG based document Q&A system that parses various document formats and builds a knowledge base for intelligent question answering.

## Features

- **Multi-format Document Parsing**
  - PDF: GLM-OCR (외부 API 서버)
  - DOCX, PPTX, XLSX: Microsoft MarkItDown
  - MD, TXT: Native parsing

- **Image Processing**
  - Extract images from documents
  - Generate descriptions using Vision LLM (GPT-4o, Claude 3)
  - Store images locally with graph references

- **Graph RAG Pipeline**
  - Text chunking with overlap
  - LLM-based entity extraction
  - Relation extraction (subject-predicate-object triples)
  - Knowledge graph construction

- **Hybrid Search**
  - Graph-based retrieval (Kuzu)
  - Vector similarity search (ChromaDB)

- **Interfaces**
  - CLI with Typer
  - REST API with FastAPI

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd document_agent

# Install with Poetry
poetry install

# Copy environment variables
cp .env.example .env
# Edit .env with your settings
```

## GLM-OCR Server Setup

For PDF parsing, you need to run GLM-OCR as an API server:

```bash
# Option 1: vLLM (recommended)
pip install -U vllm
vllm serve zai-org/GLM-OCR --port 8080

# Option 2: SGLang
pip install sglang
python -m sglang.launch_server --model zai-org/GLM-OCR --port 8080

# Option 3: Ollama
ollama run glm-ocr
```

## Usage

### CLI

```bash
# Index documents
docagent index ./documents/

# Ask questions
docagent ask "문서에서 주요 내용은?"

# Start API server
docagent serve --port 8000
```

### REST API

```bash
# Start server
docagent serve --port 8000

# Index document
curl -X POST http://localhost:8000/index \
  -F "file=@document.pdf"

# Ask question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "문서에서 주요 내용은?"}'
```

## Docker

### GPU 환경 (GLM-OCR 포함)

```bash
# 환경변수 설정
cp .env.example .env
# .env 파일에서 LLM_API_KEY 설정

# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f document-agent
```

### CPU 환경 (GLM-OCR 없이)

```bash
# 환경변수 설정
cp .env.example .env

# CPU 전용 설정으로 실행
docker-compose -f docker-compose.cpu.yml up -d
```

### Docker 서비스

| 서비스 | 포트 | 설명 |
|--------|------|------|
| document-agent | 8000 | 메인 API 서버 |
| glm-ocr | 8080 | GLM-OCR vLLM 서버 (GPU 필요) |

### API 테스트

```bash
# 헬스 체크
curl http://localhost:8000/health

# 문서 인덱싱
curl -X POST http://localhost:8000/index \
  -F "file=@./documents/sample.pdf"

# 질문하기
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "문서에서 주요 내용은?"}'
```

## Configuration

See `.env.example` for all available configuration options.

## License

MIT
