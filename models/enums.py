"""
Enumerations for the RAG backend system.
Defines all types, statuses, and difficulty levels.
"""

from enum import Enum


class FileType(str, Enum):
    """Supported file types."""
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    IMAGE = "image"


class QuestionType(str, Enum):
    """Question types."""
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    MIX = "mix"


class DifficultyLevel(str, Enum):
    """Question difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    DIFFICULT = "difficult"
    MIX = "mix"


class ProcessingStatus(str, Enum):
    """Processing status values."""
    SUCCESS = "Success"
    FAILED = "Failed"
    PROCESSING = "Processing"


class ProcessingStage(str, Enum):
    """Processing stages for progress tracking."""
    UPLOAD = "upload"
    VALIDATION = "validation"
    AUDIO_EXTRACTION = "audio_extraction"
    OCR = "ocr"
    TEXT_EXTRACTION = "text_extraction"
    TRANSCRIPTION = "transcription"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    ERROR = "error"


class Language(str, Enum):
    """Supported languages."""
    ARABIC = "ar"
    ENGLISH = "en"
    FRENCH = "fr"
    SPANISH = "es"
