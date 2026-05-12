"""
Document processing service - orchestrates the complete processing pipeline.
Handles document, audio, and video files with progress tracking.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Any
import logging
from PyPDF2 import PdfReader
import docx
from fastapi import UploadFile

from models.enums import FileType, ProcessingStage, ProcessingStatus
from services.file_service import file_service
from services.ocr_service import ocr_service
from services.audio_service import audio_service
from services.embedding_service import embedding_service
from services.progress_service import progress_service
from services.database_service import database_service
from config.settings import settings
from utils.chunker import text_chunker
from utils.language_detector import language_detector
from utils.metadata_extractor import extract_metadata_from_filename, compact_metadata

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """
    Orchestrates complete file processing pipeline:
    1. File validation and storage
    2. Content extraction (text/audio/video)
    3. OCR if needed
    4. Chunking
    5. Embedding generation
    6. Vector database indexing
    """
    
    def __init__(self):
        self.file_service = file_service
        self.ocr_service = ocr_service
        self.audio_service = audio_service
        self.embedding_service = embedding_service
        self.progress_service = progress_service
        self.chunker = text_chunker
    
    async def process_file(
        self,
        file: UploadFile,
        file_id: str,
        job_id: str,
        file_type: FileType,
        callback_url: Optional[str] = None,
        translate_to_english: bool = False,
        semester: Optional[str] = None,
        is_course_book: bool = False
    ) -> dict:
        """
        Process a file through the complete pipeline.
        """
        try:
            # Start job
            await self.progress_service.start_job(job_id, callback_url)

            # 1. Save file (5%)
            await self.progress_service.update(
                job_id, 5, ProcessingStage.UPLOAD, callback_url
            )
            file_path = await self.file_service.save_upload(file, file_id, file_type)

            # Save initial file info to PostgreSQL.
            # Metadata is inferred from the filename, then explicit form values override it.
            inferred_metadata = compact_metadata(extract_metadata_from_filename(file.filename or ''))
            effective_semester = semester or inferred_metadata.get('semester') or inferred_metadata.get('term')
            database_service.save_file_info(
                file_id=file_id,
                original_name=file.filename,
                file_type=file_type.value,
                subject=inferred_metadata.get('subject'),
                grade_level=inferred_metadata.get('grade_level') or inferred_metadata.get('grade'),
                semester=effective_semester,
                is_course_book=is_course_book
            )

            # 2. Extract content based on type
            segments = None
            if file_type == FileType.DOCUMENT:
                text, language = await self._process_document(
                    file_path, job_id, callback_url
                )
            elif file_type == FileType.AUDIO:
                text, language, segments = await self._process_audio(
                    file_path, file_id, job_id, callback_url,
                    translate_to_english=translate_to_english
                )
            elif file_type == FileType.VIDEO:
                text, language, segments = await self._process_video(
                    file_path, file_id, job_id, callback_url,
                    translate_to_english=translate_to_english
                )
            elif file_type == FileType.IMAGE:
                text, language = await self._process_image(
                    file_path, job_id, callback_url
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # 2.5 Save full transcript (JSON & DB) for all types
            self._save_full_transcript(file_id, file_path, text, language, segments)

            # 3. Chunk text (70%)
            await self.progress_service.update(
                job_id, 70, ProcessingStage.CHUNKING, callback_url
            )
            
            # Use specialized chunker for audio/video if segments are available
            if segments:
                logger.info(f"Using Whisper segment chunker for {file_id}. Segments count: {len(segments)}")
                raw_metadata = {
                    "file_id": file_id, 
                    "file_type": file_type.value, 
                    "language": language,
                    "semester": semester or inferred_metadata.get('semester') or inferred_metadata.get('term'),
                    "term": semester or inferred_metadata.get('semester') or inferred_metadata.get('term'),
                    "subject": inferred_metadata.get('subject'),
                    "grade_level": inferred_metadata.get('grade_level') or inferred_metadata.get('grade'),
                    "grade": inferred_metadata.get('grade_level') or inferred_metadata.get('grade'),
                    "book": inferred_metadata.get('book'),
                    "is_course_book": is_course_book
                }
                filtered_metadata = {k: v for k, v in raw_metadata.items() if v is not None}
                chunks = self.chunker.chunk_whisper_segments(
                    segments, 
                    metadata=filtered_metadata
                )
            else:
                logger.info(f"Using fallback text chunker for {file_id}")
                chunks = self._chunk_text(text, file_id, file_type, language, semester, is_course_book)

            if not chunks:
                raise ValueError("No text chunks were created from the uploaded file")

            logger.info(f"Created {len(chunks)} chunks for file {file_id}")

            # 4. Embed and index (75-95%)
            await self._embed_and_index(
                chunks, file_id, job_id, callback_url
            )

            # 5. Complete
            await self.progress_service.complete_job(job_id, callback_url)

            return {
                "status":       ProcessingStatus.SUCCESS,
                "fileId":       file_id,
                "chunksCreated": len(chunks),
                "language":     language,
                "textLength":   len(text)
            }

        except Exception as e:
            logger.error(f"Processing failed for {file_id}: {e}")
            await self.progress_service.fail_job(
                job_id, str(e), callback_url
            )
            raise
    
    async def _process_document(
        self,
        file_path: str,
        job_id: str,
        callback_url: Optional[str]
    ) -> Tuple[str, str]:
        """Process document file with page-by-page OCR fallback."""
        # Validation (10%)
        await self.progress_service.update(
            job_id, 10, ProcessingStage.VALIDATION, callback_url
        )
        
        # 1. Extract direct text page-by-page (15-30%)
        await self.progress_service.update(
            job_id, 15, ProcessingStage.TEXT_EXTRACTION, callback_url
        )
        
        page_texts = await self._extract_text_pages(file_path)
        num_pages = len(page_texts)
        logger.info(f"Directly extracted {num_pages} pages from {file_path}")
        
        # 2. Identify pages needing OCR (30-60%)
        bad_page_indices = []
        for i, text in enumerate(page_texts):
            if not self.ocr_service.is_text_extractable(text):
                bad_page_indices.append(i + 1) # 1-indexed for OCR service
        
        if bad_page_indices:
            logger.info(f"Pages needing OCR: {bad_page_indices}")
            await self.progress_service.update(
                job_id, 30, ProcessingStage.OCR, callback_url
            )
            
            # Extract only problematic pages
            ocr_results = await self.ocr_service.extract_pages_with_ocr(file_path, bad_page_indices)
            
            # Replace with OCR text
            for page_num, ocr_text in ocr_results.items():
                if ocr_text:
                    page_texts[page_num - 1] = ocr_text
                    
            await self.progress_service.update(
                job_id, 60, ProcessingStage.TEXT_EXTRACTION, callback_url
            )
        else:
            await self.progress_service.update(
                job_id, 60, ProcessingStage.TEXT_EXTRACTION, callback_url
            )
        
        # Merge all pages
        full_text = "\n\n--- Page Break ---\n\n".join(page_texts)
        
        # Detect language
        language = language_detector.detect_language(full_text)
        
        return full_text, language
    
    async def _process_audio(
        self,
        file_path: str,
        file_id: str,
        job_id: str,
        callback_url: Optional[str],
        translate_to_english: bool = False
    ) -> Tuple[str, str, list[dict[str, Any]]]:
        """
        Process audio file.
        """
        await self.progress_service.update(
            job_id, 10, ProcessingStage.TRANSCRIPTION, callback_url
        )

        result = await self.audio_service.process_audio(
            file_path,
            translate_to_english=translate_to_english
        )

        text = result.get("text", "")
        language = result.get("language", "unknown")
        return text, language, result.get("segments", [])

    async def _process_video(
        self,
        file_path: str,
        file_id: str,
        job_id: str,
        callback_url: Optional[str],
        translate_to_english: bool = False
    ) -> Tuple[str, str, list[dict[str, Any]]]:
        """
        Process video file.
        """
        await self.progress_service.update(
            job_id, 10, ProcessingStage.AUDIO_EXTRACTION, callback_url
        )

        await self.progress_service.update(
            job_id, 30, ProcessingStage.TRANSCRIPTION, callback_url
        )

        result = await self.audio_service.process_video(
            file_path,
            translate_to_english=translate_to_english
        )

        text = result.get("text", "")
        language = result.get("language", "unknown")
        return text, language, result.get("segments", [])
    
    async def _process_image(
        self,
        file_path: str,
        job_id: str,
        callback_url: Optional[str]
    ) -> Tuple[str, str]:
        """
        Process image file using OCR.
        Extracts text from images using Tesseract OCR.
        """
        await self.progress_service.update(
            job_id, 30, ProcessingStage.OCR, callback_url
        )

        # Extract text using OCR
        text = await self.ocr_service.extract_text_from_image(file_path)
        
        await self.progress_service.update(
            job_id, 60, ProcessingStage.OCR, callback_url
        )

        # Detect language from extracted text
        language = language_detector.detect_language(text) if text.strip() else "unknown"
        
        logger.info(f"OCR extracted {len(text)} characters from image, language: {language}")
        
        return text, language
    
    async def _extract_text_pages(self, file_path: str) -> List[str]:
        """
        Extract text from document file page-by-page.
        """
        ext = file_path.lower().split('.')[-1]

        if ext == 'pdf':
            return await asyncio.to_thread(self._extract_pages_from_pdf, file_path)
        elif ext in ['docx', 'doc']:
            text = await asyncio.to_thread(self._extract_from_docx, file_path)
            return [text]
        elif ext == 'txt':
            text = await asyncio.to_thread(self._extract_from_txt, file_path)
            return [text]
        elif ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']:
            # Images for OCR
            text = await self.ocr_service.extract_text_from_image(file_path)
            return [text]
        else:
            raise ValueError(f"Unsupported document type: {ext}")
    
    async def _extract_text(self, file_path: str) -> str:
        """Extract all text as a single string (wrapper)."""
        pages = await self._extract_text_pages(file_path)
        return "\n\n".join(pages)
    
    def _extract_pages_from_pdf(self, file_path: str) -> List[str]:
        """Extract text from PDF page by page."""
        try:
            reader = PdfReader(file_path)
            pages = []
            
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            
            return pages
            
        except Exception as e:
            logger.warning(f"PDF page extraction failed: {e}")
            return [""]
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX."""
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _chunk_text(
        self,
        text: str,
        file_id: str,
        file_type: FileType,
        language: str,
        semester: Optional[str] = None,
        is_course_book: bool = False
    ) -> list:
        """Chunk text with metadata."""
        db_metadata = database_service.get_file_metadata(file_id) or {}
        file_meta = compact_metadata(extract_metadata_from_filename(file_id or ''))
        raw_metadata = {
            "file_id": file_id,
            "file_type": file_type.value,
            "language": language,
            "semester": semester or db_metadata.get('semester') or file_meta.get('semester') or file_meta.get('term'),
            "term": semester or db_metadata.get('semester') or file_meta.get('semester') or file_meta.get('term'),
            "subject": db_metadata.get('subject') or file_meta.get('subject'),
            "grade_level": db_metadata.get('grade_level') or file_meta.get('grade_level') or file_meta.get('grade'),
            "grade": db_metadata.get('grade_level') or file_meta.get('grade_level') or file_meta.get('grade'),
            "book": file_meta.get('book'),
            "is_course_book": is_course_book or bool(db_metadata.get('is_course_book'))
        }
        metadata = {k: v for k, v in raw_metadata.items() if v is not None}
        
        chunks = self.chunker.chunk_text(text, metadata)
        
        return chunks
    
    async def _embed_and_index(
        self,
        chunks: list,
        file_id: str,
        job_id: str,
        callback_url: Optional[str]
    ):
        """Embed chunks and index in vector DB."""
        # Extract texts and metadatas
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Progress updates during embedding
        total_chunks = len(chunks)
        
        # Batch processing for efficiency
        batch_size = int(getattr(settings, "embedding_batch_size", 32) or 32)
        for i in range(0, total_chunks, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            # Update progress (75-95%)
            progress = 75 + int((i / total_chunks) * 20)
            await self.progress_service.update(
                job_id, progress, ProcessingStage.EMBEDDING, callback_url
            )
            
            # Add batch to database
            await self.embedding_service.add_documents(
                texts=batch_texts,
                metadatas=batch_metadatas,
                file_id=file_id,
                start_idx=i
            )
        
        # Indexing complete
        await self.progress_service.update(
            job_id, 95, ProcessingStage.INDEXING, callback_url
        )

    def _save_full_transcript(self, file_id: str, file_path: str, text: str, language: str, segments: list = None):
        """Save full transcript as JSON and to PostgreSQL."""
        import json
        import os
        
        # 1. Save JSON file (for format=raw in UI)
        transcript_filename = f"{os.path.basename(file_path)}.json"
        os.makedirs(settings.transcript_path, exist_ok=True)
        transcript_file_path = os.path.join(settings.transcript_path, transcript_filename)
        
        # Create full result structure similar to Whisper output
        result = {
            "text": text,
            "language": language,
            "segments": segments or []
        }
        
        try:
            with open(transcript_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved transcript JSON to {transcript_file_path}")
        except Exception as e:
            logger.error(f"Failed to save transcript JSON: {e}")

        # 2. Save to PostgreSQL
        try:
            from services.database_service import database_service
            database_service.save_transcript(
                file_id=file_id,
                full_text=text,
                language=language
            )
            if segments:
                database_service.save_timestamps(file_id, segments)
            logger.info(f"Saved transcript to PostgreSQL for {file_id}")
        except Exception as e:
            logger.error(f"Failed to save transcript to DB: {e}")


# Global instance
document_processing_service = DocumentProcessingService()
