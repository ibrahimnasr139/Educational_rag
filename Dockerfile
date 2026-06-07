FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_PROVIDER=openai \
    TRANSCRIPTION_PROVIDER=openai \
    OCR_PROVIDER=openai \
    ENABLE_AUDIO_PROCESSING=true \
    ENABLE_OCR_PROCESSING=true \
    KEEP_UPLOADED_FILES=false \
    SAVE_TRANSCRIPT_FILES=false \
    SAVE_CHUNKS_TO_POSTGRES=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements-prod.txt

COPY . .
RUN mkdir -p data/uploads data/temp data/transcripts data/chroma_db logs

EXPOSE 8000
CMD ["gunicorn", "main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "180"]
