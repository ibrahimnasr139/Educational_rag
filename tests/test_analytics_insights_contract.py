import pytest

from services.analytics_service import AnalyticsService, rag_service


@pytest.mark.asyncio
async def test_insights_return_empty_array_when_data_is_insufficient(monkeypatch):
    service = AnalyticsService()
    monkeypatch.setattr(service, "get_completion_insights", lambda tenant_id: [])
    monkeypatch.setattr(
        service,
        "get_performance_insights",
        lambda tenant_id: [{"avg_grade": 87.5, "student_count": 1, "grades_count": 1}],
    )
    monkeypatch.setattr(
        service,
        "get_revenue_insights",
        lambda tenant_id: [{"month": "2026-07", "total_revenue_egp": 400, "currency": "EGP"}],
    )

    async def generation_must_not_run(**kwargs):
        raise AssertionError("The LLM must not run for insufficient analytics data")

    monkeypatch.setattr(rag_service, "generate_directly", generation_must_not_run)

    assert await service.analyze_with_ai(8) == []


@pytest.mark.asyncio
async def test_insights_follow_the_array_contract(monkeypatch):
    service = AnalyticsService()
    monkeypatch.setattr(
        service,
        "get_completion_insights",
        lambda tenant_id: [{"title": "JavaScript Bootcamp", "avg_progress": 32, "student_count": 5}],
    )
    monkeypatch.setattr(service, "get_performance_insights", lambda tenant_id: [])
    monkeypatch.setattr(service, "get_revenue_insights", lambda tenant_id: [])

    async def fake_generation(**kwargs):
        return """[
            {
                "id": "anything",
                "type": "warning",
                "category": "completion",
                "title": "انخفاض معدل إكمال الكورس",
                "description": "متوسط الإكمال الحالي منخفض ويبلغ 32٪.",
                "confidence": "high",
                "suggestedActions": [{"label": "راجع محتوى الوحدات التي يتوقف عندها الطلاب"}],
                "courseName": "JavaScript Bootcamp"
            }
        ]"""

    monkeypatch.setattr(rag_service, "generate_directly", fake_generation)

    result = await service.analyze_with_ai(8)

    assert len(result) == 1
    assert result[0].id == "1"
    assert result[0].category == "completion"
    assert result[0].courseName == "JavaScript Bootcamp"


@pytest.mark.asyncio
async def test_revenue_insight_using_dollars_is_discarded(monkeypatch):
    service = AnalyticsService()
    monkeypatch.setattr(service, "get_completion_insights", lambda tenant_id: [])
    monkeypatch.setattr(service, "get_performance_insights", lambda tenant_id: [])
    monkeypatch.setattr(
        service,
        "get_revenue_insights",
        lambda tenant_id: [
            {"month": "2026-07", "total_revenue_egp": 400, "currency": "EGP"},
            {"month": "2026-06", "total_revenue_egp": 300, "currency": "EGP"},
        ],
    )

    async def fake_generation(**kwargs):
        return """[{
            "id": "1",
            "type": "success",
            "category": "revenue",
            "title": "زيادة الإيرادات إلى $400",
            "description": "تحسن الإيراد الشهري.",
            "confidence": "high",
            "suggestedActions": []
        }]"""

    monkeypatch.setattr(rag_service, "generate_directly", fake_generation)

    assert await service.analyze_with_ai(8) == []
