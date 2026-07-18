import asyncio

from models.enums import DifficultyLevel, QuestionType
from models.schemas import FlashcardsRequest, GenerateQuestionsRequest, GenerateQuizRequest
from services.question_service import QuestionService


ARABIC_GRAMMAR = "\u0627\u0644\u0642\u0648\u0627\u0639\u062f \u0627\u0644\u0646\u062d\u0648\u064a\u0629"
ARABIC_NAHW = "\u0627\u0644\u0646\u062d\u0648"
ARABIC_TOPIC = "\u0627\u0644\u0645\u0628\u062a\u062f\u0623 \u0648\u0627\u0644\u062e\u0628\u0631"


def test_difficulty_accepts_hard_and_normalizes_legacy_difficult():
    assert GenerateQuestionsRequest(difficulty="hard").difficulty == DifficultyLevel.HARD
    assert GenerateQuestionsRequest(difficulty="difficult").difficulty == DifficultyLevel.HARD
    assert GenerateQuizRequest(subject="Physics", difficulty="hard").difficulty == DifficultyLevel.HARD
    assert GenerateQuizRequest(subject="Physics", difficulty="difficult").difficulty == DifficultyLevel.HARD


def test_question_output_contract_uses_hard_not_difficult():
    schema = QuestionService()._get_output_schema(QuestionType.MCQ)

    assert schema["items"]["difficulty"] == "easy|medium|hard"


def test_quiz_uses_material_language_not_curriculum_language():
    service = QuestionService()
    captured = {}

    async def fake_structured(prompt, schema, system_instruction="", context=None):
        captured["prompt"] = prompt
        captured["system_instruction"] = system_instruction
        return []

    service._structured = fake_structured

    asyncio.run(service.generate_quiz(GenerateQuizRequest(subject="English", chapter=ARABIC_GRAMMAR)))

    assert "Use Arabic." in captured["prompt"]
    assert "in Arabic" in captured["system_instruction"]


def test_flashcards_uses_material_language_not_curriculum_language():
    service = QuestionService()
    captured = {}

    async def fake_structured(prompt, schema, system_instruction="", context=None):
        captured["prompt"] = prompt
        captured["system_instruction"] = system_instruction
        return []

    service._structured = fake_structured

    asyncio.run(
        service.generate_flashcards(
            FlashcardsRequest(subject="English", chapter=ARABIC_NAHW, topic=ARABIC_TOPIC)
        )
    )

    assert "Use Arabic." in captured["prompt"]
    assert "in Arabic" in captured["system_instruction"]
