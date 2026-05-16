"""
Pydantic models for the AI/RAG backend.
Compatible with Pydantic v2.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, RootModel
from typing import List, Optional, Dict, Any, Literal
from models.enums import FileType, QuestionType, DifficultyLevel, ProcessingStatus, ProcessingStage


class EmbedTranscribeRequest(BaseModel):
    type: FileType
    fileId: str
    callbackUrl: Optional[str] = None
    jobId: str


class EmbedTranscribeResponse(BaseModel):
    jobId: str
    status: ProcessingStatus
    fileId: str


class ProgressUpdate(BaseModel):
    jobId: str
    progress: int = Field(..., ge=0, le=100)
    stage: ProcessingStage
    message: str
    error: Optional[str] = None


class QuestionMetadata(BaseModel):
    course: Optional[str] = ""
    module: Optional[str] = ""
    title: Optional[str] = ""
    description: Optional[str] = ""
    subject: Optional[str] = "General"
    grade: Optional[str] = "General"
    semester: Optional[str] = None
    is_course_book: bool = False


class QuestionOption(BaseModel):
    id: str
    label: str
    isCorrect: bool


class GeneratedQuestion(BaseModel):
    id: str = ""
    order: int = 0
    question: str
    marks: int = Field(default=2)
    type: QuestionType
    options: Optional[List[QuestionOption]] = None
    difficulty: DifficultyLevel
    correctAnswer: Optional[str] = None

    @field_validator("options", mode="after")
    @classmethod
    def validate_options(cls, v, info):
        if info.data.get("type") == QuestionType.MCQ and not v:
            raise ValueError("MCQ questions must have options")
        return v


class GenerateQuestionsRequest(BaseModel):
    metadata: QuestionMetadata = Field(default_factory=QuestionMetadata)
    prompt: Optional[str] = ""
    questionsNumber: int = Field(default=10, ge=1, le=50)
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MIX)
    type: QuestionType = Field(default=QuestionType.MCQ)

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, v):
        return "difficult" if v == "hard" else v


class GenerateQuestionsResponse(RootModel[List[GeneratedQuestion]]):
    pass


class CourseContext(BaseModel):
    title: Optional[str] = ""
    level: Optional[str] = ""
    description: Optional[str] = ""


class ModuleContext(BaseModel):
    title: Optional[str] = ""
    description: Optional[str] = ""


class LessonContext(BaseModel):
    title: Optional[str] = ""
    description: Optional[str] = ""


class DescriptionContext(BaseModel):
    course: Optional[CourseContext] = None
    module: Optional[ModuleContext] = None
    lesson: Optional[LessonContext] = None
    quiz: Optional[Dict[str, Any]] = None
    assignment: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


DescriptionType = Literal["course", "module", "lesson", "quiz", "assignment", "exam", "content"]


class GenerateDescriptionRequest(BaseModel):
    # New contract: context + type. title is kept for backwards compatibility with old UI.
    context: Optional[DescriptionContext] = None
    type: DescriptionType = Field(default="content", validation_alias="descriptionType")
    title: Optional[str] = None


class GenerateDescriptionResponse(BaseModel):
    description: str


class SummaryRequest(BaseModel):
    fileId: str
    summaryLength: str = "medium"
    language: Optional[str] = None

    @field_validator("summaryLength")
    @classmethod
    def validate_length(cls, v: str) -> str:
        allowed = {"short", "medium", "long"}
        if v not in allowed:
            raise ValueError(f"summaryLength must be one of {allowed}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"ar", "en"}:
            raise ValueError("language must be 'ar' or 'en'")
        return v


class SummaryResponse(BaseModel):
    fileId: str
    summary: str
    keyPoints: List[str]
    language: str
    sourceLanguage: str
    fileType: Optional[str]
    wordCount: int
    chunksUsed: int


class ProcessingJob(BaseModel):
    jobId: str
    fileId: str
    type: FileType
    status: ProcessingStatus
    progress: int = 0
    stage: ProcessingStage = ProcessingStage.UPLOAD
    callbackUrl: Optional[str] = None
    error: Optional[str] = None
    createdAt: str
    updatedAt: str


class DocumentChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]
    chunkId: str
    fileId: str


class EmbeddingResult(BaseModel):
    embedding: List[float]
    text: str
    metadata: Dict[str, Any]


class RAGContext(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]


class GenerateTranscriptResponse(BaseModel):
    jobId: str
    status: Literal["failed", "success"]
    fileId: str


class EmbedFileResponse(BaseModel):
    status: Literal["failed", "success"]
    fileId: str


class FlashcardsRequest(BaseModel):
    subject: str
    chapter: str
    topic: str
    goal: Optional[str] = None
    numberOfCards: int = Field(default=10, ge=1, le=50)


class Flashcard(BaseModel):
    front: str
    back: str


class AskAIRequest(BaseModel):
    question: str
    previousAnswer: Optional[str] = None


class AskAIResponse(BaseModel):
    question: str
    explanation: str
    examples: List[str] = []


class GenerateQuizRequest(BaseModel):
    subject: str = Field(..., validation_alias="topic")
    numberOfQuestions: int = Field(default=10, ge=1, le=50, validation_alias="questionsNumber")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    chapter: str = Field(default="", validation_alias="module")

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, v):
        if v == "hard": return "difficult"
        if v == "mix": return "mix"
        return v


class QuizOption(BaseModel):
    text: str
    isCorrect: bool


class QuizQuestion(BaseModel):
    question: str
    options: List[QuizOption]
    explanation: str
    type: Literal["mcq"] = "mcq"


class AIAssistantRequest(BaseModel):
    message: str
    fileId: str
    student_id: str
    tenant_id: str
    course: Optional[str] = ""
    module: Optional[str] = ""
    lesson: Optional[str] = ""


class AIAssistantResponse(BaseModel):
    response: str
