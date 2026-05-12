# Docker Deployment Plan (rag-backend-local)

This file is a step-by-step runbook to deploy the FastAPI backend in this repo using Docker.

## 0) Prerequisites

- Docker installed on the target machine.
- A `.env` file with the required secrets and configuration.
- Network access to your LLM provider(s) (Gemini and/or OpenAI).

## 1) Confirm the runtime entrypoint

This backend starts via Uvicorn with:

- `main:app`

Health endpoints:

- `GET /` (basic)
- `GET /health` (detailed)

## 2) Prepare your `.env`

Create a `.env` in the repo root (don’t commit it). At minimum:

- `GOOGLE_API_KEY=...`
- `GEMINI_MODEL=gemini-2.5-flash`

Optional (if you want OpenAI paths enabled instead of Gemini):

- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4.1-nano`
- `LLM_PROVIDER=openai`

Recommended for Docker:

- `HOST=0.0.0.0`
- `PORT=8000`
- `DEBUG=false`

If you use OCR inside Docker, you must override Windows defaults (see section 7):

- `TESSERACT_PATH=/usr/bin/tesseract`

## 3) Decide: minimal container vs full multimedia/OCR container

This repo supports document/image/audio/video processing. In Docker (Linux), you typically need Linux packages for:

- `ffmpeg` (audio/video)
- `tesseract-ocr` (OCR)
- `poppler-utils` (PDF rendering for OCR pipelines)

This deployment plan assumes you need **full support** (OCR + PDF + audio/video).

## 4) Add a `Dockerfile`

Create a `Dockerfile` in the repo root with the following contents:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for full functionality:
# - ffmpeg: audio/video processing
# - tesseract-ocr: OCR
# - poppler-utils: PDF rendering (pdftoppm) used by pdf2image
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 5) Add a `.dockerignore`

Recommended ignores to keep builds fast and images small:

- `venv/`
- `__pycache__/`
- `temp/`
- `logs/`
- `data/` (optional; usually mounted as a volume)
- `temp/_unused/`

## 6) Build the image

From repo root:

```bash
docker build -t rag-backend:latest .
```

## 7) Run locally (with persisted ChromaDB + logs)

Mount `data/` and `logs/` so they persist across restarts.

Because settings default `TESSERACT_PATH` to a Windows path, you must override it for Linux:

- `TESSERACT_PATH=/usr/bin/tesseract`

```bash
docker run --rm -p 8000:8000 --env-file .env \
  -e TESSERACT_PATH=/usr/bin/tesseract \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/logs:/app/logs" \
  rag-backend:latest
```

Notes:

- ChromaDB persistence lives under `./data/chroma_db` (config default: `VECTOR_DB_PATH=./data/chroma_db`).
- Logs default to `./logs/app.log`.

## 8) Health check after startup

- `http://localhost:8000/`
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

If `/health` fails, check container logs:

```bash
docker logs <container_id>
```

## 9) Production run options

### Option A — docker compose (recommended)

Create a `docker-compose.yml` in the repo root:

```yaml
services:
  rag-backend:
    build: .
    image: rag-backend:latest
    container_name: rag-backend
    restart: unless-stopped
    env_file:
      - .env
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - TESSERACT_PATH=/usr/bin/tesseract
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
```

Then run:

```bash
docker compose up -d --build
```

### Option B — raw docker run

Add restart policy:

```bash
docker run -d --restart unless-stopped -p 8000:8000 --env-file .env \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/logs:/app/logs" \
  --name rag-backend rag-backend:latest
```

## 10) Secrets and safety

- Do **not** bake API keys into the image.
- Keep `.env` out of git.
- Consider limiting CORS origins in production (`CORS_ORIGINS`).

## 11) Common pitfalls

- **Windows paths in settings**: `TESSERACT_PATH` defaults to a Windows path. In Docker/Linux you must override it.
- **Large ML deps**: `torch`, `sentence-transformers`, `transformers` will make the image large; expect longer builds.
- **Dhakira**:
  - If you want Dhakira enabled inside Docker, the image must be able to import `dhakira`.
  - If you keep Dhakira as in-repo source under `./Dhakira`, ensure it remains in the build context (do not exclude it in `.dockerignore`).
  - Alternatively, install it explicitly in the Dockerfile (editable install), e.g. `pip install -e ./Dhakira`.

## 12) Validation checklist

- [ ] Container starts without exceptions
- [ ] `GET /health` returns `status=healthy`
- [ ] Upload + embed endpoint works: `POST /api/embed-and-transcribe`
- [ ] RAG generation endpoint works: `POST /api/ai-generate-questions`
- [ ] ChromaDB persists after container restart (volume mounted)
