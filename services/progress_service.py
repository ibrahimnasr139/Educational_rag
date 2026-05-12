"""
Progress service for managing async job progress tracking.
Coordinates WebSocket and callback updates for file processing jobs.
"""

import asyncio
from typing import Optional
from models.enums import ProcessingStage, ProcessingStatus
from utils.callbacks import progress_tracker, get_stage_message
import logging

logger = logging.getLogger(__name__)


class ProgressService:
    """
    Manages progress updates for async processing jobs.
    Provides convenience methods for common progress patterns.
    """
    
    def __init__(self):
        self.tracker = progress_tracker
    
    async def start_job(
        self,
        job_id: str,
        callback_url: Optional[str] = None,
        language: str = "ar"
    ):
        """Start a new job."""
        await self.update(
            job_id=job_id,
            progress=0,
            stage=ProcessingStage.UPLOAD,
            callback_url=callback_url,
            language=language
        )
    
    async def update(
        self,
        job_id: str,
        progress: int,
        stage: ProcessingStage,
        callback_url: Optional[str] = None,
        language: str = "ar",
        custom_message: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Update job progress.
        
        Args:
            job_id: Job identifier
            progress: Progress percentage (0-100)
            stage: Current processing stage
            callback_url: Optional callback URL
            language: Language for messages
            custom_message: Optional custom message
            error: Optional error message
        """
        message = custom_message or get_stage_message(stage, language)
        
        await self.tracker.update_progress(
            job_id=job_id,
            progress=progress,
            stage=stage,
            message=message,
            callback_url=callback_url,
            error=error
        )
    
    async def complete_job(
        self,
        job_id: str,
        callback_url: Optional[str] = None,
        language: str = "ar"
    ):
        """Mark job as completed."""
        await self.update(
            job_id=job_id,
            progress=100,
            stage=ProcessingStage.COMPLETED,
            callback_url=callback_url,
            language=language
        )
    
    async def fail_job(
        self,
        job_id: str,
        error: str,
        callback_url: Optional[str] = None,
        language: str = "ar"
    ):
        """Mark job as failed."""
        await self.update(
            job_id=job_id,
            progress=0,
            stage=ProcessingStage.ERROR,
            callback_url=callback_url,
            language=language,
            error=error
        )
    
    def register_websocket(self, job_id: str, websocket):
        """Register WebSocket connection for job."""
        self.tracker.register_websocket(job_id, websocket)
    
    def unregister_websocket(self, job_id: str):
        """Unregister WebSocket connection."""
        self.tracker.unregister_websocket(job_id)


# Global instance
progress_service = ProgressService()
