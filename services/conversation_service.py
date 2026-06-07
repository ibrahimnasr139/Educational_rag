from services.database_service import database_service
from sqlalchemy import text
from datetime import datetime
import re

def get_valid_student_id(session, student_id_candidate):
    try:
        val = int(student_id_candidate)
        exists = session.execute(text('SELECT 1 FROM "Students" WHERE "Id" = :val'), {"val": val}).fetchone()
        if exists:
            return val
    except Exception:
        digits = re.findall(r'\d+', str(student_id_candidate))
        if digits:
            val = int(digits[0])
            exists = session.execute(text('SELECT 1 FROM "Students" WHERE "Id" = :val'), {"val": val}).fetchone()
            if exists:
                return val
                
    row = session.execute(text('SELECT "Id" FROM "Students" LIMIT 1')).fetchone()
    if row:
        return row[0]
    return 1

def get_valid_lesson_id(session, lesson_id_candidate):
    try:
        val = int(lesson_id_candidate)
        exists = session.execute(text('SELECT 1 FROM "ModuleItems" WHERE "Id" = :val'), {"val": val}).fetchone()
        if exists:
            return val
    except Exception:
        digits = re.findall(r'\d+', str(lesson_id_candidate))
        if digits:
            val = int(digits[0])
            exists = session.execute(text('SELECT 1 FROM "ModuleItems" WHERE "Id" = :val'), {"val": val}).fetchone()
            if exists:
                return val
    
    row = session.execute(text('SELECT "Id" FROM "ModuleItems" LIMIT 1')).fetchone()
    if row:
        return row[0]
    return 16


class ConversationService:

    def save_message(
        self,
        student_id,
        tenant_id,
        role,
        message
    ):
        with database_service.get_session() as session:
            valid_student_id = get_valid_student_id(session, student_id)
            valid_lesson_id = get_valid_lesson_id(session, tenant_id)
            
            role_str = str(role).lower()
            role_int = 0
            if "assistant" in role_str:
                role_int = 1
            elif "system" in role_str:
                role_int = 2

            session.execute(text("""
                INSERT INTO "AiAssistantMessages"
                ("StudentId", "LessonId", "Role", "Content", "CreatedAt")
                VALUES
                (:student_id, :lesson_id, :role, :content, :created_at)
            """), {
                "student_id": valid_student_id,
                "lesson_id": valid_lesson_id,
                "role": role_int,
                "content": message or "",
                "created_at": datetime.utcnow()
            })

            session.commit()

    def get_last_messages(
        self,
        student_id,
        limit=5
    ):
        with database_service.get_session() as session:
            valid_student_id = get_valid_student_id(session, student_id)

            rows = session.execute(text("""
                SELECT "Role", "Content"
                FROM "AiAssistantMessages"
                WHERE "StudentId" = :student_id
                ORDER BY "CreatedAt" DESC
                LIMIT :limit
            """), {
                "student_id": valid_student_id,
                "limit": limit
            }).fetchall()

            output = []
            for r in reversed(rows):
                role_str = "user"
                if r[0] == 1:
                    role_str = "assistant"
                elif r[0] == 2:
                    role_str = "system"
                output.append({"role": role_str, "message": r[1] or ""})
            return output

conversation_service = ConversationService()
