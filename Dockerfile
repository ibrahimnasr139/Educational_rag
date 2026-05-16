FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-ara \
    poppler-utils \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade "pip<26" "setuptools<81" wheel \
    && pip install --no-build-isolation openai-whisper==20231117 \
    && pip install torch==2.1.2 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

COPY . .
RUN mkdir -p data/uploads data/temp data/transcripts data/chroma_db logs

EXPOSE 8000
CMD ["gunicorn", "main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "180"]
