# Quick Start Guide

Docker Compose를 사용하여 Document Agent를 빠르게 배포하고 시작하는 가이드입니다.

## 사전 요구사항

- Docker & Docker Compose 설치
- OpenAI API 키 (또는 호환 LLM API)
- (선택) NVIDIA GPU + nvidia-docker (GLM-OCR 사용 시)

## 1. 저장소 클론

```bash
git clone https://github.com/kimysub/doc-graph-rag.git
cd doc-graph-rag
```

## 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
nano .env  # 또는 원하는 에디터 사용
```

**필수 설정:**

```bash
# LLM API 키 (필수)
LLM_API_KEY=sk-your-openai-api-key-here
```

**선택 설정:**

```bash
# LLM 모델 (기본값: gpt-4o)
LLM_MODEL=gpt-4o

# Vision LLM 모델 (기본값: gpt-4o)
VISION_LLM_MODEL=gpt-4o

# 임베딩 모델 (기본값: text-embedding-3-small)
EMBEDDING_MODEL=text-embedding-3-small
```

## 3. 서비스 시작

### Option A: CPU 환경 (GLM-OCR 없이)

PDF OCR 없이 DOCX, PPTX, XLSX, MD, TXT만 처리:

```bash
docker-compose -f docker-compose.cpu.yml up -d
```

### Option B: GPU 환경 (GLM-OCR 포함)

PDF OCR을 포함한 전체 기능:

```bash
# NVIDIA Docker 런타임 필요
docker-compose up -d
```

## 4. 서비스 상태 확인

```bash
# 컨테이너 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f document-agent

# 헬스 체크
curl http://localhost:8000/health
```

정상 응답:
```json
{"status": "healthy"}
```

## 5. 문서 인덱싱

### API로 문서 업로드

```bash
# 단일 파일 인덱싱
curl -X POST http://localhost:8000/index \
  -F "file=@./your-document.pdf"

# 응답 예시
{
  "success": true,
  "message": "문서 'your-document.pdf'이(가) 성공적으로 인덱싱되었습니다.",
  "doc_id": "abc123def456",
  "chunks_count": 42,
  "entities_count": 15,
  "images_count": 3
}
```

### 지원 파일 형식

| 형식 | 확장자 | 파서 |
|------|--------|------|
| PDF | `.pdf` | GLM-OCR (GPU 필요) |
| Word | `.docx`, `.doc` | MarkItDown |
| PowerPoint | `.pptx` | MarkItDown |
| Excel | `.xlsx`, `.xls` | MarkItDown |
| Markdown | `.md` | Native |
| Text | `.txt` | Native |

## 6. 질문하기

```bash
# 질문 API 호출
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "문서에서 주요 내용은 무엇인가요?"}'

# 응답 예시
{
  "answer": "문서의 주요 내용은 다음과 같습니다...",
  "sources": null
}
```

### 출처 정보 포함

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "프로젝트 일정은?", "include_sources": true}'
```

## 7. API 문서

브라우저에서 Swagger UI 접속:

```
http://localhost:8000/docs
```

### 주요 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 헬스 체크 |
| `POST` | `/index` | 문서 인덱싱 |
| `POST` | `/ask` | 질문 & 답변 |
| `POST` | `/search` | 엔티티 검색 |
| `GET` | `/stats` | 인덱스 통계 |
| `GET` | `/entity/{name}/subgraph` | 엔티티 서브그래프 |
| `GET` | `/entity/{name}/chunks` | 엔티티 관련 청크 |

## 8. 서비스 중지

```bash
# 서비스 중지
docker-compose down

# 데이터 포함 완전 삭제
docker-compose down -v
```

## 트러블슈팅

### 포트 충돌

```bash
# 다른 포트 사용
docker-compose -f docker-compose.cpu.yml up -d
# 또는 docker-compose.yml에서 포트 변경
```

### GPU 인식 안됨

```bash
# NVIDIA Docker 런타임 확인
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 메모리 부족

```bash
# GLM-OCR는 약 4GB VRAM 필요
# CPU 버전 사용 권장
docker-compose -f docker-compose.cpu.yml up -d
```

### 로그 확인

```bash
# 전체 로그
docker-compose logs

# 특정 서비스 로그
docker-compose logs document-agent
docker-compose logs glm-ocr

# 실시간 로그
docker-compose logs -f
```

## 다음 단계

- [Architecture Plan](./PLAN.md) - 상세 아키텍처 문서
- [README](../README.md) - 전체 프로젝트 문서
- [API Docs](http://localhost:8000/docs) - Swagger UI
