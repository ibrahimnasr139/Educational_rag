import os
import aiofiles
import hashlib
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException
from config.settings import settings
from models.enums import FileType
import logging

logger = logging.getLogger(__name__)

# Read/write in 1 MB chunks - good balance of memory vs I/O calls
CHUNK_SIZE = 1024 * 1024


class FileService:
    """
    Manages file operations: streaming upload, validation, and storage.
    """

    SUPPORTED_EXTENSIONS = {
        FileType.DOCUMENT: {
            # Text files
            '.pdf', '.txt', '.rtf', '.md',
            # Microsoft Office
            '.docx', '.doc', '.docm',
            '.xlsx', '.xls', '.xlsm', '.csv',
            '.pptx', '.ppt', '.pptm',
            # OpenDocument formats
            '.odt', '.ods', '.odp',
            # Other formats
            '.epub', '.html', '.htm', '.xml'
        },
        FileType.AUDIO: {
            # Common audio formats
            '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac',
            '.wma', '.opus', '.aiff', '.au', '.ra', '.amr'
        },
        FileType.VIDEO: {
            # Common video formats
            '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv',
            '.wmv', '.m4v', '.3gp', '.ogv', '.ts', '.mts', '.m2ts'
        },
        FileType.IMAGE: {
            # Image formats for OCR
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
            '.gif', '.webp', '.svg', '.ico'
        }
    }

    MIME_TYPE_MAP = {
        # Documents
        'application/pdf':                              FileType.DOCUMENT,
        'text/plain':                                   FileType.DOCUMENT,
        'text/rtf':                                     FileType.DOCUMENT,
        'text/markdown':                                FileType.DOCUMENT,
        'text/html':                                    FileType.DOCUMENT,
        'text/xml':                                     FileType.DOCUMENT,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileType.DOCUMENT,
        'application/msword':                           FileType.DOCUMENT,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.template': FileType.DOCUMENT,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':     FileType.DOCUMENT,
        'application/vnd.ms-excel':                     FileType.DOCUMENT,
        'text/csv':                                     FileType.DOCUMENT,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': FileType.DOCUMENT,
        'application/vnd.ms-powerpoint':                FileType.DOCUMENT,
        'application/vnd.oasis.opendocument.text':       FileType.DOCUMENT,
        'application/vnd.oasis.opendocument.spreadsheet': FileType.DOCUMENT,
        'application/vnd.oasis.opendocument.presentation': FileType.DOCUMENT,
        'application/epub+zip':                         FileType.DOCUMENT,
        
        # Audio
        'audio/mpeg':                                   FileType.AUDIO,
        'audio/wav':                                    FileType.AUDIO,
        'audio/mp4':                                    FileType.AUDIO,
        'audio/aac':                                    FileType.AUDIO,
        'audio/ogg':                                    FileType.AUDIO,
        'audio/flac':                                   FileType.AUDIO,
        'audio/x-ms-wma':                               FileType.AUDIO,
        'audio/opus':                                   FileType.AUDIO,
        'audio/x-aiff':                                 FileType.AUDIO,
        'audio/basic':                                  FileType.AUDIO,
        'audio/x-realaudio':                            FileType.AUDIO,
        'audio/amr':                                    FileType.AUDIO,
        
        # Video
        'video/mp4':                                    FileType.VIDEO,
        'video/avi':                                    FileType.VIDEO,
        'video/quicktime':                              FileType.VIDEO,
        'video/x-matroska':                             FileType.VIDEO,
        'video/webm':                                   FileType.VIDEO,
        'video/x-flv':                                  FileType.VIDEO,
        'video/x-ms-wmv':                               FileType.VIDEO,
        'video/x-m4v':                                  FileType.VIDEO,
        'video/3gpp':                                   FileType.VIDEO,
        'video/ogg':                                    FileType.VIDEO,
        'video/mp2t':                                   FileType.VIDEO,
        'video/MP2T':                                   FileType.VIDEO,
        
        # Images
        'image/jpeg':                                   FileType.IMAGE,
        'image/png':                                    FileType.IMAGE,
        'image/bmp':                                    FileType.IMAGE,
        'image/tiff':                                   FileType.IMAGE,
        'image/gif':                                    FileType.IMAGE,
        'image/webp':                                   FileType.IMAGE,
        'image/svg+xml':                                FileType.IMAGE,
        'image/x-icon':                                 FileType.IMAGE
    }

    def __init__(self):
        self.upload_path   = settings.upload_path
        self.temp_path     = settings.temp_path
        self.max_size_bytes = settings.max_file_size * 1024 * 1024

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def save_upload(
        self,
        file: UploadFile,
        file_id: str,
        file_type: FileType
    ) -> str:
        """
        Validate then stream-save an uploaded file to disk.

        The file is written in CHUNK_SIZE (1 MB) chunks using aiofiles
        so it never fully lives in memory and the event loop is not blocked.

        Returns:
            Absolute path to the saved file.
        """
        # Validate file object
        if not file or not hasattr(file, 'read'):
            raise HTTPException(status_code=400, detail="Invalid file object")
        
        # 1. Lightweight validation (extension + MIME, no full read)
        self._validate_extension(file.filename, file_type)
        self._validate_mime(file.content_type, file_type)

        # 2. Build destination path
        filename = self._safe_filename(file.filename, file_id)
        type_dir  = os.path.join(self.upload_path, file_type.value)
        os.makedirs(type_dir, exist_ok=True)
        dest_path = os.path.join(type_dir, filename)

        logger.info(f"Processing upload: {file.filename} -> {dest_path}")

        # 3. Stream file to disk and enforce size limit simultaneously
        total_bytes = await self._stream_to_disk(file, dest_path)

        logger.info(
            f"Saved '{file.filename}' → {dest_path} "
            f"({total_bytes / (1024*1024):.2f} MB)"
        )
        return dest_path

    # ------------------------------------------------------------------
    # Streaming write
    # ------------------------------------------------------------------

    async def _stream_to_disk(self, file: UploadFile, dest_path: str) -> int:
        """
        Write UploadFile to dest_path in chunks.
        Raises HTTPException(413) if the file exceeds MAX_FILE_SIZE.
        Does not swallow read/write errors, so partial uploads cannot look successful.
        """
        total_bytes = 0
        try:
            async with aiofiles.open(dest_path, "wb") as out:
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    total_bytes += len(chunk)
                    if total_bytes > self.max_size_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large. Maximum size is {settings.max_file_size} MB"
                        )

                    await out.write(chunk)

        except HTTPException:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise
        except Exception as e:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            logger.error(f"Failed writing upload to {dest_path}: {e}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")

        if total_bytes == 0:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        return total_bytes

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_extension(self, filename: str, expected_type: FileType):
        """Check file extension is allowed for this file type."""
        ext = os.path.splitext(filename)[1].lower()
        allowed = self.SUPPORTED_EXTENSIONS.get(expected_type, set())
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Extension '{ext}' not allowed for type '{expected_type.value}'. "
                    f"Allowed: {sorted(allowed)}"
                )
            )

    def _validate_mime(self, content_type: Optional[str], expected_type: FileType):
        """Soft MIME check - only warn, don't reject (browsers send inconsistent values)."""
        if not content_type:
            return
        detected = self.MIME_TYPE_MAP.get(content_type)
        if detected and detected != expected_type:
            logger.warning(
                f"MIME mismatch: content_type='{content_type}' suggests {detected}, "
                f"but type param says {expected_type}"
            )

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------

    def _safe_filename(self, original: str, file_id: str) -> str:
        """
        Build a collision-resistant filename.
        Format: <safe_file_id>_<8-char-hash><ext>
        """
        _, ext = os.path.splitext(original)
        safe_id = "".join(c for c in file_id if c.isalnum() or c in "-_")
        short_hash = hashlib.md5(original.encode()).hexdigest()[:8]
        return f"{safe_id}_{short_hash}{ext}"

    # ------------------------------------------------------------------
    # Lookup / cleanup utilities
    # ------------------------------------------------------------------

    def get_file_path(self, file_id: str, file_type: FileType) -> Optional[str]:
        """Find a previously saved file by file_id prefix."""
        type_dir = os.path.join(self.upload_path, file_type.value)
        if not os.path.isdir(type_dir):
            return None
        for name in os.listdir(type_dir):
            if name.startswith(file_id):
                return os.path.join(type_dir, name)
        return None

    async def create_temp_file(self, content: bytes, extension: str) -> str:
        """Write bytes to a temp file and return the path."""
        name = f"tmp_{hashlib.md5(content).hexdigest()}{extension}"
        path = os.path.join(self.temp_path, name)
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)
        return path

    def cleanup_temp_files(self, older_than_hours: int = 24):
        """Delete temp files older than given hours."""
        import time
        if not os.path.isdir(self.temp_path):
            return
        cutoff = time.time() - older_than_hours * 3600
        for name in os.listdir(self.temp_path):
            path = os.path.join(self.temp_path, name)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    logger.info(f"Deleted old temp file: {name}")
            except OSError as e:
                logger.warning(f"Could not delete {name}: {e}")


# Global singleton
file_service = FileService()