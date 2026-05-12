# RAG Educational Backend - Final Metadata-Aware Version

This version includes:

- FastAPI backend for documents, images, audio, and video.
- OCR with Arabic/English Tesseract.
- Whisper transcription.
- Chroma vector DB.
- PostgreSQL/SQLite storage for files, transcripts, timestamps, and chunks.
- Metadata-aware RAG retrieval: subject, grade, semester/term, book, and course-book flag.
- Dhakira optional integration with SentenceTransformer fallback.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
copy .env.example .env
python run_local.py
```

## Run UI

```bash
python run_ui.py
```

## Docker

```bash
docker compose up --build
```

## Dhakira

If you want to use Dhakira directly, clone it inside the project root:

```bash
git clone https://github.com/h9-tec/Dhakira Dhakira
```

The backend will use Dhakira when available, otherwise it safely falls back to SentenceTransformer.
