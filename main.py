"""
Main FastAPI application for Arabic-first AI/RAG backend.
Includes file ingestion, transcription, RAG question generation, descriptions, flashcards,
quiz generation, ask-ai, and file-aware assistant endpoints.
"""
from __future__ import annotations

import logging
import os
import json
import time
from contextlib import asynccontextmanager
from typing import Optional

# Add local binaries to PATH
_project_root = os.path.dirname(os.path.abspath(__file__))
_ffmpeg_dir = os.path.join(_project_root, "ffmpeg-bin")
_poppler_dir = os.path.join(_project_root, "poppler-bin", "Library", "bin")

_paths_to_add = []
if os.path.exists(_ffmpeg_dir):
    _paths_to_add.append(_ffmpeg_dir)
if os.path.exists(_poppler_dir):
    _paths_to_add.append(_poppler_dir)

if _paths_to_add:
    os.environ["PATH"] = os.pathsep.join(_paths_to_add) + os.pathsep + os.environ.get("PATH", "")

from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from models.enums import FileType, ProcessingStatus
from models.schemas import (
    EmbedTranscribeResponse,
    GenerateTranscriptResponse,
    EmbedFileResponse,
    GenerateQuestionsRequest,
    GenerateDescriptionRequest,
    GenerateDescriptionResponse,
    SummaryRequest,
    SummaryResponse,
    FlashcardsRequest,
    Flashcard,
    AskAIRequest,
    AskAIResponse,
    GenerateQuizRequest,
    QuizQuestion,
    AIAssistantRequest,
    AIAssistantResponse,
)
from services import document_processing_service, question_service, progress_service
from services.embedding_service import embedding_service
from services.suggest_service import suggest_service
from services.summary_service import summary_service
from services.analytics_service import analytics_service
from services.database_service import database_service
from utils.callbacks import progress_tracker

os.makedirs(os.path.dirname(settings.log_file) if os.path.dirname(settings.log_file) else ".", exist_ok=True)
os.makedirs(settings.upload_path, exist_ok=True)
os.makedirs(settings.temp_path, exist_ok=True)
os.makedirs(getattr(settings, "transcript_path", "./data/transcripts"), exist_ok=True)

file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI/RAG Backend Server...")
    logger.info(f"Vector DB: {settings.vector_db_path}")
    logger.info(f"Upload Path: {settings.upload_path}")
    try:
        database_service.init_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    yield
    logger.info("Shutting down...")
    await progress_tracker.close()


app = FastAPI(
    title="Arabic-First AI/RAG Backend",
    description="AI/RAG APIs for educational content generation and file processing",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=settings.cors_methods.split(","),
    allow_headers=settings.cors_headers.split(","),
)


@app.get("/")
async def root():
    return {"service": "Arabic-First AI/RAG Backend", "status": "running", "version": "1.1.0"}


@app.get("/health")
async def health_check():
    try:
        stats = embedding_service.get_collection_stats()
    except Exception as e:
        stats = {"error": str(e)}
    return {
        "status": "healthy",
        "database": stats,
        "settings": {
            "whisper_model": settings.whisper_model,
            "embedding_model": settings.embedding_model,
            "default_language": settings.default_language,
            "llm_provider": settings.llm_provider,
        },
    }


async def _process_uploaded_file(
    *,
    file: UploadFile,
    file_type: FileType,
    file_id: str,
    job_id: str,
    callback_url: Optional[str] = None,
    translate_to_english: bool = False,
    semester: Optional[str] = None,
    is_course_book: bool = False,
):
    return await document_processing_service.process_file(
        file=file,
        file_id=file_id,
        file_type=file_type,
        job_id=job_id,
        callback_url=callback_url,
        translate_to_english=translate_to_english,
        semester=semester,
        is_course_book=is_course_book,
    )


# -------------------- File processing endpoints --------------------
@app.post("/api/embed-and-transcribe", response_model=EmbedTranscribeResponse)
async def embed_and_transcribe(
    file: UploadFile = File(...),
    type: str = Form(...),
    fileId: str = Form(...),
    jobId: str = Form(...),
    callbackUrl: Optional[str] = Form(None),
    translateToEnglish: bool = Form(False),
    semester: Optional[str] = Form(None),
    isCourseBook: bool = Form(False),
):
    try:
        file_type = FileType(type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file type. Must be: video | audio | document | image")

    try:
        await _process_uploaded_file(
            file=file,
            file_type=file_type,
            file_id=fileId,
            job_id=jobId,
            callback_url=callbackUrl,
            translate_to_english=translateToEnglish,
            semester=semester,
            is_course_book=isCourseBook,
        )
        return EmbedTranscribeResponse(jobId=jobId, status=ProcessingStatus.SUCCESS, fileId=fileId)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/generate-transcript", response_model=GenerateTranscriptResponse)
async def generate_transcript(
    audio: UploadFile = File(...),
    fileId: str = Form(...),
    callbackUrl: Optional[str] = Form(None),
    jobId: str = Form(...),
):
    """Transcribe an audio file and store transcript/chunks for RAG."""
    try:
        await _process_uploaded_file(
            file=audio,
            file_type=FileType.AUDIO,
            file_id=fileId,
            job_id=jobId,
            callback_url=callbackUrl,
        )
        return GenerateTranscriptResponse(jobId=jobId, status="success", fileId=fileId)
    except Exception as e:
        logger.error(f"Transcript generation failed: {e}")
        return GenerateTranscriptResponse(jobId=jobId, status="failed", fileId=fileId)


@app.post("/api/embed-file", response_model=EmbedFileResponse)
async def embed_file(
    file: UploadFile = File(...),
    type: str = Form(...),
    fileId: str = Form(...),
    callbackUrl: Optional[str] = Form(None),
):
    """Embed a document or video file for later RAG use."""
    try:
        file_type = FileType(type)
        if file_type not in {FileType.DOCUMENT, FileType.VIDEO, FileType.IMAGE}:
            raise ValueError("/api/embed-file supports document, video, or image")
        job_id = f"job-{int(time.time() * 1000)}"
        await _process_uploaded_file(
            file=file,
            file_type=file_type,
            file_id=fileId,
            job_id=job_id,
            callback_url=callbackUrl,
        )
        return EmbedFileResponse(status="success", fileId=fileId)
    except Exception as e:
        logger.error(f"File embedding failed: {e}")
        return EmbedFileResponse(status="failed", fileId=fileId)


# -------------------- AI generation endpoints --------------------
@app.post("/api/ai-generate-questions")
async def generate_questions(request: GenerateQuestionsRequest):
    try:
        questions = await question_service.generate_questions(request)
        return [q.model_dump() for q in questions]
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")


@app.post("/api/generate-description", response_model=GenerateDescriptionResponse)
async def generate_description(request: GenerateDescriptionRequest):
    try:
        description = await question_service.generate_description(
            context=request.context,
            content_type=request.type,
            title=request.title,
        )
        return GenerateDescriptionResponse(description=description)
    except Exception as e:
        logger.error(f"Description generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate description: {str(e)}")


@app.post("/api/generate-flashcards", response_model=list[Flashcard])
async def generate_flashcards(request: FlashcardsRequest):
    try:
        return await question_service.generate_flashcards(request)
    except Exception as e:
        logger.error(f"Flashcard generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask-ai", response_model=AskAIResponse)
async def ask_ai(request: AskAIRequest):
    try:
        return await question_service.ask_ai(request)
    except Exception as e:
        logger.error(f"Ask AI failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-quiz", response_model=list[QuizQuestion])
async def generate_quiz(request: GenerateQuizRequest):
    try:
        return await question_service.generate_quiz(request)
    except Exception as e:
        logger.error(f"Quiz generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai-assistant", response_model=AIAssistantResponse)
async def ai_assistant(request: AIAssistantRequest):
    try:
        response = await question_service.ai_assistant(request)
        return AIAssistantResponse(response=response)
    except Exception as e:
        logger.error(f"AI assistant failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Analytics endpoints --------------------
@app.get("/api/analytics/completion")
async def get_completion_analytics():
    try:
        return analytics_service.get_completion_insights()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/performance")
async def get_performance_analytics():
    try:
        return analytics_service.get_performance_insights()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/revenue")
async def get_revenue_analytics():
    try:
        return analytics_service.get_revenue_insights()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/ai-analysis")
async def get_ai_analytics():
    try:
        analysis = await analytics_service.analyze_with_ai()
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Existing helper endpoints --------------------
@app.post("/api/summarise", response_model=SummaryResponse)
async def summarise_file(request: SummaryRequest):
    try:
        result = await summary_service.summarise_file(
            file_id=request.fileId,
            summary_length=request.summaryLength,
            language=request.language,
        )
        return SummaryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Summary failed for {request.fileId}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@app.get("/api/get-chunks/{file_id}")
async def get_chunks_for_transcript(file_id: str):
    chunks = embedding_service.get_all_chunks_for_file(file_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for file_id: {file_id}")
    return chunks


@app.get("/api/get-transcript-raw/{file_id}")
async def get_transcript_raw(file_id: str):
    transcript_path = getattr(settings, "transcript_path", "./data/transcripts")
    try:
        for filename in os.listdir(transcript_path):
            if filename.startswith(file_id) and filename.endswith(".json"):
                with open(os.path.join(transcript_path, filename), "r", encoding="utf-8") as f:
                    return json.load(f)
    except FileNotFoundError:
        pass
    raise HTTPException(status_code=404, detail="Transcript raw data not found")


@app.get("/api/suggest-metadata/{file_id}")
async def suggest_metadata(file_id: str):
    try:
        return await suggest_service.suggest_metadata(file_id)
    except Exception as e:
        logger.error(f"Metadata suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/suggest-topics/{file_id}")
async def suggest_topics(file_id: str):
    try:
        topics = await suggest_service.suggest_topics(file_id)
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Topic suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    progress_service.register_websocket(job_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        progress_service.unregister_websocket(job_id)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {type(e).__name__} - {repr(e)}")
        progress_service.unregister_websocket(job_id)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc) if settings.debug else "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug, workers=1)
