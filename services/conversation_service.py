from services.database_service import database_service
from sqlalchemy import text
from uuid import uuid4
from datetime import datetime

class ConversationService:

    def save_message(
        self,
        student_id,
        tenant_id,
        role,
        message
    ):
        with database_service.get_session() as session:
            session.execute(text("""
                INSERT INTO AiAssistantMessages
                (id, student_id, tenant_id, role, message, created_at)
                VALUES
                (:id, :student_id, :tenant_id, :role, :message, :created_at)
            """), {
                "id": str(uuid4()),
                "student_id": student_id,
                "tenant_id": tenant_id,
                "role": role,
                "message": message,
                "created_at": datetime.utcnow()
            })

            session.commit()

    def get_last_messages(
        self,
        student_id,
        limit=5
    ):
        with database_service.get_session() as session:

            rows = session.execute(text("""
                SELECT role, message
                FROM AiAssistantMessages
                WHERE student_id = :student_id
                ORDER BY created_at DESC
                LIMIT :limit
            """), {
                "student_id": student_id,
                "limit": limit
            }).fetchall()

            return [{"role": r.role, "message": r.message} for r in reversed(rows)]

conversation_service = ConversationService()
