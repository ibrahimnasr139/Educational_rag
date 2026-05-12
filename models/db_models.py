from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Float, Boolean, Index, JSON
from sqlalchemy.orm import relationship, DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Files(Base):
    __tablename__ = "files"

    id = Column(String, primary_key=True)
    tenant_id = Column(String, nullable=True)
    uploaded_by = Column(String, nullable=True)

    name = Column(String)
    size = Column(Integer)
    type = Column(String)

    url = Column(String)
    storage_provider = Column(String)

    metadata_ = Column("metadata", JSON)
    status = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    metadata_info = relationship("Metadata", back_populates="file", uselist=False, cascade="all, delete-orphan")
    transcript = relationship("Transcripts", back_populates="file", uselist=False, cascade="all, delete-orphan")
    video_timestamps = relationship("VideoTimestamps", back_populates="file", cascade="all, delete-orphan")

class FileChunks(Base):
    __tablename__ = "file_chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String, ForeignKey("files.id"), index=True)
    tenant_id = Column(String, nullable=True)
    text = Column(Text)
    tokens = Column(Integer)
    chunk_index = Column(Integer)
    model_name = Column(String)
    metadata_ = Column("metadata", JSON)



class Metadata(Base):
    __tablename__ = "metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String, ForeignKey("files.id"), unique=True, index=True)
    subject = Column(String, index=True)
    grade_level = Column(String)
    semester = Column(String, index=True)
    is_course_book = Column(Boolean, default=False, index=True)

    file = relationship("Files", back_populates="metadata_info")


class Transcripts(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String, ForeignKey("files.id"), unique=True, index=True)
    full_text = Column(Text)
    language = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    file = relationship("Files", back_populates="transcript")


class VideoTimestamps(Base):
    __tablename__ = "video_timestamps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String, ForeignKey("files.id"), index=True)
    segment_index = Column(Integer, index=True)
    text = Column(Text)
    start_time = Column(Float)
    end_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    file = relationship("Files", back_populates="video_timestamps")

    __table_args__ = (
        Index("idx_video_timestamps_file_segment", "file_id", "segment_index"),
    )

class AiAssistantMessages(Base):
    __tablename__ = "AiAssistantMessages"

    id = Column(String, primary_key=True)
    student_id = Column(String, index=True)
    tenant_id = Column(String, index=True)
    role = Column(String)
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
