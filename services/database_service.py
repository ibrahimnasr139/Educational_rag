from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker, Session
from config.settings import settings
from models.db_models import Base, Files, Metadata, Transcripts, VideoTimestamps, FileChunks
from datetime import datetime
from utils.metadata_extractor import extract_metadata_from_filename, compact_metadata
import logging
from sqlalchemy.engine import make_url

logger = logging.getLogger(__name__)


class DatabaseService:
    def __init__(self):
        db_url = make_url(settings.database_url)
        connect_args = {}
        if db_url.drivername.startswith("postgresql"):
            connect_args = {
                "sslmode": "require",
                "application_name": "railway-app"
            }

        self.engine = create_engine(
            settings.database_url,
            poolclass=NullPool,
            pool_pre_ping=True,
            connect_args=connect_args
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def init_db(self):
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database (check DATABASE_URL/SSL/network): {e}")
            raise

    def get_session(self) -> Session:
        return self.SessionLocal()

    def save_file_info(
        self,
        file_id: str,
        original_name: str,
        file_type: str,
        subject: str = None,
        grade_level: str = None,
        semester: str = None,
        is_course_book: bool = False,
        uploaded_by_id: str = None,
        size: int = 0,
        url: str = "",
    ):
        with self.get_session() as session:
            try:
                db_file = session.query(Files).filter(Files.id == file_id).first()
                filename_metadata = compact_metadata(extract_metadata_from_filename(original_name))
                metadata = dict(filename_metadata)
                if subject:
                    metadata["subject"] = subject
                if grade_level:
                    metadata["grade_level"] = grade_level
                    metadata["grade"] = grade_level
                if semester:
                    metadata["semester"] = semester
                    metadata["term"] = semester
                metadata["is_course_book"] = bool(is_course_book)
                
                if db_file is None:
                    db_file = Files(
                        id=file_id,
                        name=original_name,
                        type=file_type,
                        size=size or 0,
                        url=url or "",
                        storage_provider="local",
                        metadata_=metadata,
                        status="processing",
                        uploaded_by=uploaded_by_id,
                        tenant_id=None
                    )
                    session.add(db_file)
                else:
                    db_file.name = original_name or db_file.name
                    db_file.size = size or db_file.size
                    db_file.url = url or db_file.url
                    db_file.metadata_ = metadata
                    if uploaded_by_id:
                        db_file.uploaded_by = uploaded_by_id

                db_metadata = session.query(Metadata).filter(Metadata.file_id == file_id).first()
                if db_metadata is None:
                    db_metadata = Metadata(file_id=file_id)
                    session.add(db_metadata)
                db_metadata.subject = subject or metadata.get("subject")
                db_metadata.grade_level = grade_level or metadata.get("grade_level") or metadata.get("grade")
                db_metadata.semester = semester or metadata.get("semester") or metadata.get("term")
                db_metadata.is_course_book = is_course_book

                session.commit()
                logger.info(f"Saved file and metadata for {file_id}")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save file info: {e}")
                raise

    def save_transcript(self, file_id: str, full_text: str, language: str):
        with self.get_session() as session:
            try:
                clean_text = full_text.replace("\x00", "") if full_text else ""
                db_transcript = session.query(Transcripts).filter(Transcripts.file_id == file_id).first()
                if db_transcript is None:
                    db_transcript = Transcripts(file_id=file_id)
                    session.add(db_transcript)
                db_transcript.full_text = clean_text
                db_transcript.language = language
                db_transcript.created_at = datetime.utcnow()

                session.commit()
                logger.info(f"Saved transcript for {file_id}")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save transcript: {e}")
                raise

    def save_timestamps(self, file_id: str, segments: list):
        with self.get_session() as session:
            try:
                session.query(VideoTimestamps).filter(VideoTimestamps.file_id == file_id).delete()
                for i, seg in enumerate(segments):
                    session.add(VideoTimestamps(
                        file_id=file_id,
                        segment_index=i,
                        text=(seg.get("text", "") or "").replace("\x00", ""),
                        start_time=float(seg.get("start", 0.0) or 0.0),
                        end_time=float(seg.get("end", 0.0) or 0.0),
                    ))

                session.commit()
                logger.info(f"Saved {len(segments)} timestamps for {file_id}")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save timestamps: {e}")
                raise

    def get_file_metadata(self, file_id: str):
        try:
            with self.get_session() as session:
                db_metadata = session.query(Metadata).filter(Metadata.file_id == file_id).first()
                if db_metadata:
                    return {
                        "subject": db_metadata.subject,
                        "grade_level": db_metadata.grade_level,
                        "semester": db_metadata.semester,
                        "is_course_book": db_metadata.is_course_book,
                    }
                return None
        except Exception as e:
            logger.warning(f"Could not fetch metadata from DB for {file_id}: {e}")
            return None

    def file_exists(self, file_id: str) -> bool:
        with self.get_session() as session:
            return session.query(Files).filter(Files.id == file_id).first() is not None

    def save_chunks(self, file_id: str, chunks: list, embeddings: list, model_name: str, metadatas: list | None = None, start_idx: int = 0):
        with self.get_session() as session:
            try:
                file = session.query(Files).filter_by(id=file_id).first()
                base_metadata = dict(file.metadata_ or {}) if file else {}
                if start_idx == 0:
                    session.query(FileChunks).filter(FileChunks.file_id == file_id).delete()

                for i, chunk in enumerate(chunks):
                    chunk_metadata = dict(base_metadata)
                    if metadatas and i < len(metadatas) and metadatas[i]:
                        chunk_metadata.update(metadatas[i])
                    chunk_metadata = compact_metadata(chunk_metadata)

                    row = FileChunks(
                        file_id=file_id,
                        tenant_id=file.tenant_id if file else None,
                        text=(chunk or '').replace('\x00', ''),
                        tokens=len((chunk or '').split()),
                        chunk_index=start_idx + i,
                        model_name=model_name,
                        metadata_=chunk_metadata
                    )
                    session.add(row)

                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save chunks for {file_id}: {e}")
                raise

    def _chunk_to_dict(self, chunk: FileChunks) -> dict:
        return {
            "id": chunk.id,
            "file_id": chunk.file_id,
            "text": chunk.text or "",
            "chunk_index": chunk.chunk_index or 0,
            "metadata": chunk.metadata_ or {},
        }

    def get_filtered_chunks(self, metadata_filters: dict, limit: int = 2000):
        """Return chunks matching metadata. Filtering is done in Python for JSON portability across SQLite/PostgreSQL."""
        filters = compact_metadata(metadata_filters or {})
        with self.get_session() as session:
            rows = session.query(FileChunks).order_by(FileChunks.id.desc()).limit(limit).all()
            output = []
            for row in rows:
                meta = row.metadata_ or {}
                ok = True
                for key, expected in filters.items():
                    if key == "grade":
                        candidates = [meta.get("grade"), meta.get("grade_level")]
                    elif key == "term":
                        candidates = [meta.get("term"), meta.get("semester")]
                    else:
                        candidates = [meta.get(key)]
                    if expected and expected not in candidates:
                        ok = False
                        break
                if ok:
                    output.append(self._chunk_to_dict(row))
            return output

    def get_all_chunks(self, limit: int = 2000):
        with self.get_session() as session:
            rows = session.query(FileChunks).order_by(FileChunks.id.desc()).limit(limit).all()
            return [self._chunk_to_dict(row) for row in rows]


try:
    database_service = DatabaseService()
except Exception as exc:
    logger.error("DatabaseService initialization failed: %s", exc)
    raise
