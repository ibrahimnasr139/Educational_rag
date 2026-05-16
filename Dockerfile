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

# Cache bust - change this value to force full pip reinstall: v5
RUN echo "cache-bust-v5"

# Step 1: Upgrade pip and tools
RUN pip install --upgrade pip setuptools wheel

# Step 2: Install numpy < 2.0.0 FIRST before anything else touches it
RUN pip install "numpy<2.0.0"

# Step 3: Install chromadb with numpy pinned (this resolves the np.float_ error)
RUN pip install "chromadb>=0.5.3"

# Step 4: Re-pin numpy in case chromadb upgraded it
RUN pip install "numpy<2.0.0" --upgrade

# Step 5: Install torch
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Step 6: Install remaining requirements
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Step 7: Final safety re-pin of numpy
RUN pip install "numpy<2.0.0"

COPY . .
RUN mkdir -p data/uploads data/temp data/transcripts data/chroma_db logs

EXPOSE 8000
CMD ["gunicorn", "main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "180"]
