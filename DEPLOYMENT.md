# Deployment Guide: RAG Educational Backend

This guide outlines the step-by-step process required to deploy the Retrieval-Augmented Generation (RAG) backend application.

## 1. Prerequisites

Before deploying, ensure the target environment has the following software installed:
- **Python 3.10+**
- **Tesseract OCR**: Required for extracting Arabic/English text from images and scanned PDFs.
  - Windows: Download from UB-Mannheim.
  - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-ara`
- **FFmpeg**: Required by Whisper for audio processing.
  - Linux: `sudo apt install ffmpeg`
  - Windows: Install and add to your system PATH.
- **PostgreSQL**: Either installed locally or a managed database (e.g., Railway, Neon).

## 2. Setting Up the Environment

1. **Clone the Project**:
   Ensure you have the latest source code of the project.
2. **Create a Virtual Environment**:
   It is recommended to use a virtual environment to manage dependencies:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   Install all required Python packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you run into issues on Windows, you can use the provided powershell script `install_dependencies.ps1` or `install_dependencies_fixed.ps1`.*

## 3. Configuration (.env)

The application relies on environment variables for sensitive keys and configuration parameters. Create a `.env` file in the root of your project based on this template:

```ini
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Database (Replace with your actual PostgreSQL connection string)
DATABASE_URL=postgresql://user:password@hostname:port/dbname

# Storage Paths
VECTOR_DB_PATH=./data/chroma_db
UPLOAD_PATH=./data/uploads
TEMP_PATH=./data/temp

# AI Provider Keys
GOOGLE_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here

# LLM Providers (Choices: gemini, openai)
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=dhakira

# OCR
TESSERACT_PATH=/usr/bin/tesseract  # Example Linux path (update for Windows)
```

## 4. Setting Up the Database

The application uses SQLAlchemy to automatically create necessary tables when started.
1. Make sure your PostgreSQL server is running and accessible (check `DATABASE_URL`).
2. Run the server once to auto-generate the tables, or use any database migration tools if you decide to implement `Alembic` in the future.

## 5. Running the Application

### Development Environment (Local)
To start the backend server locally:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Environment (Linux)
In production, it is highly recommended to use a production-grade ASGI server like `gunicorn` with `uvicorn` workers to handle concurrent requests robustly.
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
*(Here `-w 4` configures Gunicorn to use 4 worker processes. Adjust based on your server's CPU cores.)*

## 6. Frontend/UI Deployment

If you are using the provided raw HTML/JS UI (in the `ui/` directory) alongside the backend, you can serve it using a lightweight HTTP server or configure a reverse proxy. Alternatively, you can use the included runner:
```bash
python run_ui.py
```

## 7. Initial Data Population (Curriculum Embedding)

If you are deploying for the first time and need to pre-index the curriculum data:
1. Ensure the `منهج` (Curriculum) folder is in the root directory and contains the required PDF textbooks.
2. Run the embedding script:
   ```bash
   python embed_curriculum.py
   ```
   *This process might take a while depending on the OCR requirements and hardware.*

## 8. Reverse Proxy (Nginx / Apache) (Optional but Recommended)

For a real production environment, you should expose the backend through a Reverse Proxy (like Nginx) handling SSL/TLS encryption.

Example simplified Nginx configuration block:
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
