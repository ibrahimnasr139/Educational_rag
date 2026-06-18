"""
Bulk ingest script for all supported files in the curriculum folder.

Steps:
  1. Optionally clear existing data from Chroma and these tables:
     Files, FileChunks, Transcripts, VideoTimestamps, Metadata
  2. Loop over every supported file in ./منهج and process it through the
     normal pipeline: text extraction -> OCR -> chunking -> embeddings -> DB.

Usage:
    python ingest_books.py
    python ingest_books.py --clear
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import sys
import time


# Bootstrap
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_DHAKIRA = os.path.join(_ROOT, "Dhakira")
if os.path.isdir(_DHAKIRA) and _DHAKIRA not in sys.path:
    sys.path.insert(0, _DHAKIRA)


# Logging setup
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/ingest_books.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ingest_books")


# Constants
BOOKS_DIR = os.path.join(_ROOT, "منهج")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


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
                logger.info("Deleted Chroma collection '%s'", collection.name)

        if deleted == 0:
            logger.info("No Chroma document collections existed - skipping")

        embedding_service.get_or_create_collection("documents")
        logger.info("Recreated active Chroma document collection")
    except Exception as e:
        logger.error("Failed to clear ChromaDB: %s", e)
        raise


def clear_postgres():
    """Delete all rows from the project content tables."""
    from models.db_models import FileChunks, Files, Metadata, Transcripts, VideoTimestamps
    from services.database_service import database_service

    try:
        with database_service.get_session() as session:
            n_chunks = session.query(FileChunks).delete()
            n_ts = session.query(VideoTimestamps).delete()
            n_tr = session.query(Transcripts).delete()
            n_meta = session.query(Metadata).delete()
            n_files = session.query(Files).delete()
            session.commit()

        logger.info(
            "Cleared PostgreSQL - files=%s, metadata=%s, transcripts=%s, "
            "chunks=%s, timestamps=%s",
            n_files,
            n_meta,
            n_tr,
            n_chunks,
            n_ts,
        )
    except Exception as e:
        logger.error("Failed to clear PostgreSQL: %s", e)
        raise


def clear_local_transcripts():
    """Remove all JSON transcript files from ./data/transcripts."""
    from config.settings import settings

    transcript_dir = getattr(settings, "transcript_path", "./data/transcripts")
    if not os.path.isdir(transcript_dir):
        return

    removed = 0
    for fname in os.listdir(transcript_dir):
        if not fname.endswith(".json"):
            continue
        try:
            os.remove(os.path.join(transcript_dir, fname))
            removed += 1
        except Exception as e:
            logger.warning("Could not remove %s: %s", fname, e)

    logger.info("Removed %s local transcript JSON files from %s", removed, transcript_dir)


async def ingest_file(file_path: str, file_id: str) -> dict:
    """Run a single file through the full processing pipeline."""
    from models.enums import FileType
    from services.document_processing_service import document_processing_service

    file_name = os.path.basename(file_path)
    logger.info("  -> Ingesting: %s (id=%s)", file_name, file_id)

    class LocalFile:
        """Minimal UploadFile-compatible shim for local files."""

        def __init__(self, path: str):
            self._path = path
            self.filename = os.path.basename(path)
            self.headers = {}
            self._handle = None
            ext = os.path.splitext(path)[1].lower()
            self.content_type = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".txt": "text/plain",
            }.get(ext, "application/octet-stream")

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
            is_course_book=True,
        )
        logger.info(
            "  Done: %s - %s chunks, lang=%s",
            file_name,
            result.get("chunksCreated", 0),
            result.get("language", "?"),
        )
        return result
    except Exception as e:
        logger.error("  Failed: %s - %s", file_name, e)
        return {"status": "failed", "error": str(e), "fileId": file_id}
    finally:
        await local_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest curriculum files into PostgreSQL and ChromaDB.")
    parser.add_argument(
        "--books-dir",
        default=BOOKS_DIR,
        help="Folder containing curriculum files. Default: ./curriculum folder",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear Chroma, Files, FileChunks, Transcripts, VideoTimestamps, Metadata, and local transcript JSON before ingesting.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Process every file even if chunks already exist.",
    )
    return parser.parse_args()


async def ingest_all(args=None):
    """Clear data if requested, then sequentially process all books."""
    args = args or parse_args()
    books_dir = os.path.abspath(args.books_dir)

    if not os.path.isdir(books_dir):
        logger.error("Books directory not found: %s", books_dir)
        sys.exit(1)

    books = sorted(
        f for f in os.listdir(books_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )
    if not books:
        logger.warning("No supported files found in %s", books_dir)
        sys.exit(0)

    logger.info("Found %s supported files in '%s'", len(books), books_dir)
    for book in books:
        logger.info("   - %s", book)

    if args.clear:
        logger.info("\n== Clearing existing data ==")
        clear_chroma()
        clear_postgres()
        clear_local_transcripts()
        logger.info("== All existing data cleared ==\n")

    from services.database_service import database_service
    from tqdm import tqdm

    total = len(books)
    succeeded, skipped, failed = 0, 0, 0
    start_total = time.time()

    with tqdm(total=total, desc="Ingesting curriculum", unit="file") as pbar:
        for idx, book_name in enumerate(books, 1):
            file_path = os.path.join(books_dir, book_name)
            file_id = "book-" + hashlib.md5(book_name.encode("utf-8")).hexdigest()[:16]
            pbar.set_postfix_str(book_name[:40])

            if not args.no_skip:
                try:
                    if database_service.file_has_chunks(file_id):
                        logger.info("[%s/%s] Skipping %s (chunks already exist)", idx, total, book_name)
                        skipped += 1
                        pbar.update(1)
                        continue
                except Exception:
                    pass

            logger.info("\n[%s/%s] %s", idx, total, book_name)
            t0 = time.time()
            result = await ingest_file(file_path, file_id)
            elapsed = time.time() - t0

            if result.get("status") == "failed":
                failed += 1
            else:
                succeeded += 1

            logger.info("  Time: %.1fs", elapsed)
            pbar.update(1)

        pbar.set_postfix_str(f"ok={succeeded} skipped={skipped} failed={failed}")

    total_time = time.time() - start_total
    logger.info(
        "\n%s\n"
        "  Ingestion complete!\n"
        "  Total files  : %s\n"
        "  Succeeded    : %s\n"
        "  Skipped      : %s\n"
        "  Failed       : %s\n"
        "  Total time   : %.1f minutes\n"
        "%s",
        "=" * 60,
        total,
        succeeded,
        skipped,
        failed,
        total_time / 60,
        "=" * 60,
    )


if __name__ == "__main__":
    asyncio.run(ingest_all(parse_args()))
