"""
Bulk ingest script for all PDFs in the منهج folder.

Steps:
  1. Clear all existing data (Chroma vector DB + PostgreSQL transcripts/chunks/files)
  2. Loop over every PDF in ./منهج and process it through the full pipeline
     (text extraction → OCR → chunking → embeddings → DB storage)

Usage:
    python ingest_books.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import logging

# ── Bootstrap ──────────────────────────────────────────────────────────────
# Ensure project root is on the path before any local imports
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_DHAKIRA = os.path.join(_ROOT, "Dhakira")
if os.path.isdir(_DHAKIRA) and _DHAKIRA not in sys.path:
    sys.path.insert(0, _DHAKIRA)

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/ingest_books.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ingest_books")

# ── Constants ──────────────────────────────────────────────────────────────
BOOKS_DIR = os.path.join(_ROOT, "منهج")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ──────────────────────────────────────────────────────────────────────────
# Step 1 – Clear existing data
# ──────────────────────────────────────────────────────────────────────────

def clear_chroma():
    """Delete all dimension-specific Chroma document collections."""
    from services.embedding_service import embedding_service
    try:
        client = embedding_service.client
        deleted = 0
        for collection in client.list_collections():
            if collection.name == "documents" or collection.name.startswith("documents_"):
                client.delete_collection(collection.name)
                deleted += 1
                logger.info(f"Deleted Chroma collection '{collection.name}'")

        if deleted == 0:
            logger.info("No Chroma document collections existed - skipping")

        # Recreate the active dimension-specific collection so the service is ready.
        embedding_service.get_or_create_collection("documents")
        logger.info("Recreated active Chroma document collection")
    except Exception as e:
        logger.error(f"Failed to clear ChromaDB: {e}")
        raise


def clear_postgres():
    """Delete all rows from all relevant PostgreSQL tables."""
    from services.database_service import database_service
    from models.db_models import FileChunks, VideoTimestamps, Transcripts, Metadata, Files
    try:
        with database_service.get_session() as session:
            # Delete child tables first to avoid FK constraint errors
            n_chunks = session.query(FileChunks).delete()
            n_ts = session.query(VideoTimestamps).delete()
            n_tr = session.query(Transcripts).delete()
            n_meta = session.query(Metadata).delete()
            n_files = session.query(Files).delete()
            session.commit()
        logger.info(
            f"✓ Cleared PostgreSQL — "
            f"files={n_files}, metadata={n_meta}, transcripts={n_tr}, "
            f"chunks={n_chunks}, timestamps={n_ts}"
        )
    except Exception as e:
        logger.error(f"Failed to clear PostgreSQL: {e}")
        raise


def clear_local_transcripts():
    """Remove all JSON transcript files from ./data/transcripts."""
    from config.settings import settings
    transcript_dir = getattr(settings, "transcript_path", "./data/transcripts")
    if not os.path.isdir(transcript_dir):
        return
    removed = 0
    for fname in os.listdir(transcript_dir):
        if fname.endswith(".json"):
            try:
                os.remove(os.path.join(transcript_dir, fname))
                removed += 1
            except Exception as e:
                logger.warning(f"  Could not remove {fname}: {e}")
    logger.info(f"✓ Removed {removed} local transcript JSON files from {transcript_dir}")


# ──────────────────────────────────────────────────────────────────────────
# Step 2 – Process books
# ──────────────────────────────────────────────────────────────────────────

async def ingest_file(file_path: str, file_id: str) -> dict:
    """Run a single file through the full processing pipeline."""
    from fastapi import UploadFile
    from models.enums import FileType
    from services.document_processing_service import document_processing_service

    file_name = os.path.basename(file_path)
    logger.info(f"  → Ingesting: {file_name}  (id={file_id})")

    # Wrap local file in an UploadFile-compatible object
    class LocalFile:
        """Minimal UploadFile shim for local files."""
        def __init__(self, path: str):
            self._path = path
            self.filename = os.path.basename(path)
            self.content_type = "application/pdf"  # all books are PDFs
            self.headers = {}
            self._handle = None

        async def read(self, size: int = -1) -> bytes:
            if self._handle is None:
                self._handle = open(self._path, "rb")
            return self._handle.read() if size == -1 else self._handle.read(size)

        async def seek(self, offset: int):
            if self._handle is None:
                self._handle = open(self._path, "rb")
            self._handle.seek(offset)

        async def close(self):
            if self._handle:
                self._handle.close()
                self._handle = None

    local_file = LocalFile(file_path)
    job_id = f"job-ingest-{file_id}"

    try:
        result = await document_processing_service.process_file(
            file=local_file,
            file_id=file_id,
            job_id=job_id,
            file_type=FileType.DOCUMENT,
            callback_url=None,
            is_course_book=True,   # all منهج books are course books
        )
        logger.info(
            f"  ✓ Done: {file_name} — "
            f"{result.get('chunksCreated', 0)} chunks, "
            f"lang={result.get('language', '?')}"
        )
        return result
    except Exception as e:
        logger.error(f"  ✗ Failed: {file_name} — {e}")
        return {"status": "failed", "error": str(e), "fileId": file_id}
    finally:
        await local_file.close()


async def ingest_all():
    """Main coroutine: clear data then sequentially process all books."""
    # ── 1. Discover books ──────────────────────────────────────────────────
    if not os.path.isdir(BOOKS_DIR):
        logger.error(f"Books directory not found: {BOOKS_DIR}")
        sys.exit(1)

    books = sorted([
        f for f in os.listdir(BOOKS_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])
    if not books:
        logger.warning(f"No supported files found in {BOOKS_DIR}")
        sys.exit(0)

    logger.info(f"Found {len(books)} books in '{BOOKS_DIR}'")
    for b in books:
        logger.info(f"   • {b}")

    # ── 2. Clear existing data (DISABLED for resume mode) ─────────────────
    # If you want to start fresh, uncomment these or run them once manually.
    # logger.info("\n== Clearing existing data ==")
    # clear_chroma()
    # clear_postgres()
    # clear_local_transcripts()
    # logger.info("== All existing data cleared ==\n")

    # ── 3. Process each book ───────────────────────────────────────────────
    from services.database_service import database_service
    total = len(books)
    succeeded, skipped, failed = 0, 0, 0
    start_total = time.time()

    for idx, book_name in enumerate(books, 1):
        file_path = os.path.join(BOOKS_DIR, book_name)
        # Generate a stable, unique file_id from the filename
        import hashlib
        file_id = "book-" + hashlib.md5(book_name.encode("utf-8")).hexdigest()[:16]

        # RESUME CHECK: Skip only when chunks already exist.
        # A file row without chunks means a previous ingestion stopped before embedding.
        try:
            if database_service.file_has_chunks(file_id):
                logger.info(f"[{idx}/{total}] Skipping {book_name} (chunks already exist)")
                skipped += 1
                continue
        except Exception:
            # If DB is unreachable now, we'll try to process anyway (which will fail later gracefully)
            pass

        logger.info(f"\n[{idx}/{total}] {book_name}")
        t0 = time.time()
        result = await ingest_file(file_path, file_id)
        elapsed = time.time() - t0

        if result.get("status") == "failed":
            failed += 1
        else:
            succeeded += 1

        logger.info(f"  Time: {elapsed:.1f}s")

    # ── 4. Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - start_total
    logger.info(
        f"\n{'='*60}\n"
        f"  Ingestion complete!\n"
        f"  Total books  : {total}\n"
        f"  Succeeded    : {succeeded}\n"
        f"  Skipped      : {skipped}\n"
        f"  Failed       : {failed}\n"
        f"  Total time   : {total_time/60:.1f} minutes\n"
        f"{'='*60}"
    )


# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(ingest_all())
