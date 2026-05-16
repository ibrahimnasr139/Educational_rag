from services.database_service import database_service
from services.rag_service import rag_service
import logging

logger = logging.getLogger(__name__)

class AnalyticsService:
    
    def _execute(self, query, params=None):
        with database_service.get_session() as session:
            result = session.execute(query, params or {})
            return [dict(row._mapping) for row in result.fetchall()]

    def get_completion_insights(self):
        try:
            query = """
            SELECT 
                c.title, 
                AVG(cp.progress_percentage) as avg_progress
            FROM course_progress cp
            JOIN courses c ON c.id = cp.course_id
            GROUP BY c.title
            """
            return self._execute(query)
        except Exception as e:
            logger.warning(f"Analytics table missing (completion): {e}")
            return []

    def get_performance_insights(self):
        try:
            query = """
            SELECT 
                s.full_name, 
                AVG(g.grade) as avg_grade
            FROM student_grades g
            JOIN students s ON s.id = g.student_id
            GROUP BY s.full_name
            """
            return self._execute(query)
        except Exception as e:
            logger.warning(f"Analytics table missing (performance): {e}")
            return []

    def get_revenue_insights(self):
        try:
            # Using DATE_TRUNC as requested (PostgreSQL)
            query = """
            SELECT 
                DATE_TRUNC('month', created_at) as month, 
                SUM(amount) as total_revenue
            FROM student_subscriptions
            GROUP BY month
            ORDER BY month DESC
            """
            return self._execute(query)
        except Exception as e:
            logger.warning(f"Analytics table missing (revenue): {e}")
            return []

    async def analyze_with_ai(self):
        try:
            completion = self.get_completion_insights()
            performance = self.get_performance_insights()
            revenue = self.get_revenue_insights()

            analytics_data = {
                "completion_rates": completion,
                "student_performance": performance,
                "revenue_trends": revenue
            }

            prompt = f"""
            Analyze this LMS analytics data.
            
            DATA:
            {analytics_data}
            
            Return:
            - risks
            - insights
            - recommendations
            """
            
            system = "You are a senior LMS data analyst. Provide a professional and actionable analysis."
            return await rag_service.generate_directly(prompt=prompt, system_instruction=system)
        except Exception as e:
            logger.error(f"AI Analytics analysis failed: {e}")
            return f"Analysis failed: {str(e)}"

analytics_service = AnalyticsService()
