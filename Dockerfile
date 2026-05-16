FROM python:3.10-slim

# Cache bust: 2026-05-16
ARG CACHEBUST=1

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

# Step 1: Upgrade pip and tools
RUN pip install --upgrade pip setuptools wheel

# Step 2: Install numpy < 2.0.0 FIRST to prevent chromadb from pulling numpy 2.x
RUN pip install "numpy<2.0.0"

# Step 3: Install torch (CPU or GPU)
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Step 4: Install all other requirements (numpy pin will be respected)
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Step 5: Force-install compatible chromadb AFTER all other deps, no upgrades to numpy
RUN pip install "chromadb>=0.5.3" "numpy<2.0.0" --upgrade

COPY . .
RUN mkdir -p data/uploads data/temp data/transcripts data/chroma_db logs

EXPOSE 8000
CMD ["gunicorn", "main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "180"]
