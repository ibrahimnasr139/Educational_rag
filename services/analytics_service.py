from services.database_service import database_service
from services.rag_service import rag_service
from models.schemas import AIInsight
from sqlalchemy import text
from pydantic import ValidationError
import json
import logging

logger = logging.getLogger(__name__)

class AnalyticsService:
    MIN_SAMPLE_SIZE = 3
    VALID_TYPES = {"urgent", "warning", "critical", "success", "info"}
    VALID_CONFIDENCE = {"high", "medium", "low"}
    
    def _execute(self, query, params=None):
        with database_service.get_session() as session:
            result = session.execute(text(query), params or {})
            return [dict(row._mapping) for row in result.fetchall()]

    def get_completion_insights(self, tenant_id: int):
        try:
            query = """
            SELECT 
                c."Title" as title, 
                AVG(CAST(cp."CompletedLessons" AS FLOAT) / NULLIF(cp."TotalLessons", 0) * 100.0) as avg_progress,
                COUNT(DISTINCT cp."StudentId") as student_count
            FROM "CourseProgresses" cp
            JOIN "Courses" c ON c."Id" = cp."CourseId"
            WHERE c."TenantId" = :tenant_id
            GROUP BY c."Title"
            """
            return self._execute(query, {"tenant_id": tenant_id})
        except Exception as e:
            logger.warning(f"Analytics table missing (completion): {e}")
            return []

    def get_performance_insights(self, tenant_id: int):
        try:
            query = """
            SELECT
                (
                    SELECT AVG(CAST(g."Score" AS FLOAT) / NULLIF(g."TotalMarks", 0) * 100.0)
                    FROM "StudentGrades" g
                    WHERE g."TenantId" = :tenant_id
                      AND g."TotalMarks" > 1
                ) as avg_grade,
                (
                    SELECT COUNT(DISTINCT e."StudentId")
                    FROM "Enrollments" e
                    WHERE e."TenantId" = :tenant_id
                ) as student_count,
                (
                    SELECT COUNT(g."Id")
                    FROM "StudentGrades" g
                    WHERE g."TenantId" = :tenant_id
                      AND g."TotalMarks" > 1
                ) as grades_count
            WHERE EXISTS (
                SELECT 1
                FROM "StudentGrades" g
                WHERE g."TenantId" = :tenant_id
                  AND g."TotalMarks" > 1
            )
            """
            return self._execute(query, {"tenant_id": tenant_id})
        except Exception as e:
            logger.warning(f"Analytics table missing (performance): {e}")
            return []

    def get_revenue_insights(self, tenant_id: int):
        try:
            query = """
            SELECT 
                DATE_TRUNC('month', "ApprovedAt") as month,
                SUM("PricePaid") as total_revenue_egp,
                COUNT("Id") as approved_orders_count
            FROM "Orders"
            WHERE "TenantId" = :tenant_id
              AND "Status" = 1
              AND "ApprovedAt" IS NOT NULL
            GROUP BY month
            ORDER BY month DESC
            """
            rows = self._execute(query, {"tenant_id": tenant_id})
            for row in rows:
                row["currency"] = "EGP"
            return rows
        except Exception as e:
            logger.warning(f"Analytics table missing (revenue): {e}")
            return []

    @staticmethod
    def _parse_json_array(response: str):
        cleaned = (response or "").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        parsed = json.loads(cleaned.strip())
        if isinstance(parsed, dict):
            parsed = parsed.get("insights", [])
        return parsed if isinstance(parsed, list) else []

    def _eligible_categories(self, completion, performance, revenue):
        eligible = set()
        if any(int(row.get("student_count") or 0) >= self.MIN_SAMPLE_SIZE for row in completion):
            eligible.add("completion")
        if any(int(row.get("student_count") or 0) >= self.MIN_SAMPLE_SIZE for row in performance):
            eligible.add("performance")
        # One month is a total, not a trend, so it must not generate an insight.
        if len(revenue) >= 2:
            eligible.add("revenue")
        return eligible

    def _normalize_candidate(self, candidate: dict, insight_id: int) -> dict:
        normalized = dict(candidate)
        normalized["id"] = str(insight_id)

        if normalized.get("type") not in self.VALID_TYPES:
            normalized["type"] = "info"
        if normalized.get("confidence") not in self.VALID_CONFIDENCE:
            normalized["confidence"] = "medium"

        actions = normalized.get("suggestedActions") or []
        normalized_actions = []
        for action in actions:
            if isinstance(action, dict) and action.get("label"):
                normalized_actions.append({"label": str(action["label"])})
            elif isinstance(action, str) and action.strip():
                normalized_actions.append({"label": action.strip()})
        normalized["suggestedActions"] = normalized_actions

        if "courseName" in normalized and normalized["courseName"] is not None:
            normalized["courseName"] = str(normalized["courseName"])

        return normalized

    def _generate_fallback_insights(self, completion, performance, revenue, eligible_categories) -> list[AIInsight]:
        insights = []
        if "completion" in eligible_categories:
            for row in completion:
                if int(row.get("student_count") or 0) >= self.MIN_SAMPLE_SIZE:
                    title = row.get("title") or "الدورة"
                    avg_progress = float(row.get("avg_progress") or 0.0)
                    insight_type = "warning" if avg_progress < 50.0 else "info"
                    insights.append(AIInsight(
                        id=str(len(insights) + 1),
                        type=insight_type,
                        category="completion",
                        title=f"معدل إكمال دورة {title}",
                        description=f"بلغ متوسط إكمال الطلاب في دورة {title} نسبة {avg_progress:.1f}%.",
                        confidence="high",
                        suggestedActions=[
                            {"label": "أنشئ فيديو ملخص قصير (3-5 دقائق) للوحدات القادمة"},
                            {"label": "أرسل رسالة متابعة للطلاب غير النشطين"}
                        ],
                        courseName=title
                    ))

        if "performance" in eligible_categories:
            for row in performance:
                if int(row.get("student_count") or 0) >= self.MIN_SAMPLE_SIZE:
                    avg_grade = float(row.get("avg_grade") or 0.0)
                    insight_type = "warning" if avg_grade < 70.0 else "success"
                    insights.append(AIInsight(
                        id=str(len(insights) + 1),
                        type=insight_type,
                        category="performance",
                        title="تحليل درجات الطلاب الأكاديمية",
                        description=f"متوسط درجات الطلاب هو {avg_grade:.1f}%.",
                        confidence="high",
                        suggestedActions=[
                            {"label": "تقديم اختبارات تجريبية ومراجعات إضافية"}
                        ]
                    ))

        if "revenue" in eligible_categories and len(revenue) >= 2:
            current_rev = float(revenue[0].get("total_revenue_egp") or 0.0)
            prev_rev = float(revenue[1].get("total_revenue_egp") or 0.0)
            diff = current_rev - prev_rev
            trend_str = "ارتفعت" if diff >= 0 else "انخفضت"
            insight_type = "success" if diff >= 0 else "warning"
            insights.append(AIInsight(
                id=str(len(insights) + 1),
                type=insight_type,
                category="revenue",
                title="مؤشر إيرادات الاشتراكات الشهرية",
                description=f"{trend_str} الإيرادات الشهرية لتبلغ {current_rev:.0f} جنيه مصري.",
                confidence="high",
                suggestedActions=[
                    {"label": "متابعة أداء الحملات التسويقية والخصومات"}
                ]
            ))

        return insights

    async def analyze_with_ai(self, tenant_id: int) -> list[AIInsight]:
        completion = self.get_completion_insights(tenant_id)
        performance = self.get_performance_insights(tenant_id)
        revenue = self.get_revenue_insights(tenant_id)
        eligible_categories = self._eligible_categories(completion, performance, revenue)

        if not eligible_categories:
            return []

        analytics_data = {
            "completion_rates": completion,
            "student_performance": performance,
            "revenue_trends": revenue,
            "revenue_currency": "EGP",
            "eligible_categories": sorted(eligible_categories),
        }

        prompt = f"""
Analyze the following LMS data and return ONLY a valid JSON array.

DATA:
{json.dumps(analytics_data, ensure_ascii=False, default=str)}

Each array item must exactly match this shape:
{{
  "id": "1",
  "type": "urgent | warning | critical | success | info",
  "category": "completion | performance | revenue",
  "title": "Arabic title",
  "description": "Arabic description supported by the supplied data",
  "confidence": "high | medium | low",
  "suggestedActions": [{{"label": "Arabic actionable step"}}],
  "courseName": "optional course name"
}}

Rules:
- Return [] when the data contains no meaningful, actionable insight.
- Only create insights for categories listed in eligible_categories.
- Do not describe missing or insufficient data as an insight.
- Never invent comparisons, percentages, time periods, course names, or causes.
- Revenue is in Egyptian pounds (EGP). Say "جنيه مصري" and never use dollars or the $ symbol.
- All titles, descriptions, and suggested actions must be in Arabic.
- Do not include markdown, explanations, or a wrapper object.
        """

        system = (
            "You are a precise LMS data analyst. Output evidence-based JSON only. "
            "If the evidence is insufficient, output an empty JSON array."
        )

        try:
            response = await rag_service.generate_directly(prompt=prompt, system_instruction=system)
            candidates = self._parse_json_array(response)

            insights = []
            for candidate in candidates:
                if not isinstance(candidate, dict) or candidate.get("category") not in eligible_categories:
                    continue
                try:
                    normalized = self._normalize_candidate(candidate, len(insights) + 1)
                    insight = AIInsight.model_validate(normalized)
                    if insight.category == "revenue":
                        revenue_text = f"{insight.title} {insight.description}".lower()
                        if "$" in revenue_text or "usd" in revenue_text or "دولار" in revenue_text:
                            continue
                    insights.append(insight)
                except ValidationError as ve:
                    logger.warning("Discarding invalid AI insight payload (%s): %s", ve, candidate)

            return insights
        except Exception as e:
            logger.error(f"AI Analytics analysis failed: {e}")
            return self._generate_fallback_insights(completion, performance, revenue, eligible_categories)

analytics_service = AnalyticsService()

