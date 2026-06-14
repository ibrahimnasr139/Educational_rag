from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from config.settings import settings


DEFAULT_UPLOADED_BY = "d521a6ac-157e-4719-8964-b9f6bf1cf389"


def _engine():
    db_url = make_url(settings.database_url)
    connect_args = {}
    if db_url.drivername.startswith("postgresql"):
        connect_args = {
            "sslmode": "require",
            "application_name": "railway-book-table-migration",
        }
    return create_engine(settings.database_url, connect_args=connect_args)


def migrate_books() -> dict[str, int]:
    engine = _engine()
    with engine.begin() as conn:
        files_result = conn.execute(
            text(
                """
                INSERT INTO "Files" (
                    "Id", "Name", "Size", "Type", "Status", "Url",
                    "StorageProvider", "Metadata", "TenantId", "UploadedById",
                    "CreatedAt", "UpdatedAt"
                )
                SELECT
                    f.id,
                    COALESCE(f.name, ''),
                    COALESCE(f.size, 0)::bigint,
                    CASE lower(COALESCE(f.type, 'document'))
                        WHEN 'video' THEN 0
                        WHEN 'image' THEN 1
                        WHEN 'document' THEN 2
                        WHEN 'audio' THEN 3
                        ELSE 2
                    END,
                    CASE
                        WHEN lower(COALESCE(f.status, 'processing')) IN ('success', 'completed') THEN 1
                        WHEN lower(COALESCE(f.status, 'processing')) IN ('failed', 'error') THEN 3
                        ELSE 0
                    END,
                    COALESCE(f.url, ''),
                    COALESCE(f.storage_provider, 'Local'),
                    f.metadata::jsonb,
                    CASE
                        WHEN f.tenant_id ~ '^[0-9]+$' THEN f.tenant_id::integer
                        ELSE NULL
                    END,
                    COALESCE(NULLIF(f.uploaded_by, ''), :default_uploaded_by),
                    COALESCE(f.created_at, now()),
                    f.updated_at
                FROM files f
                WHERE f.id LIKE 'book-%'
                  AND NOT EXISTS (
                      SELECT 1 FROM "Files" cf WHERE cf."Id" = f.id
                  )
                """
            ),
            {"default_uploaded_by": DEFAULT_UPLOADED_BY},
        )

        metadata_result = conn.execute(
            text(
                """
                INSERT INTO "Metadata" (
                    "Subject", "GradeLevel", "Semester", "IsCourseBook", "FileId"
                )
                SELECT
                    COALESCE(NULLIF(m.subject, ''), 'General'),
                    COALESCE(NULLIF(m.grade_level, ''), 'General'),
                    COALESCE(NULLIF(m.semester, ''), 'General'),
                    COALESCE(m.is_course_book, true),
                    m.file_id
                FROM metadata m
                JOIN "Files" f ON f."Id" = m.file_id
                WHERE m.file_id LIKE 'book-%'
                  AND NOT EXISTS (
                      SELECT 1 FROM "Metadata" cm WHERE cm."FileId" = m.file_id
                  )
                """
            )
        )

        transcripts_result = conn.execute(
            text(
                """
                INSERT INTO "Transcripts" (
                    "Language", "FullText", "CreatedAt", "FileId"
                )
                SELECT
                    COALESCE(NULLIF(t.language, ''), 'unknown'),
                    COALESCE(t.full_text, ''),
                    COALESCE(t.created_at, now()),
                    t.file_id
                FROM transcripts t
                JOIN "Files" f ON f."Id" = t.file_id
                WHERE t.file_id LIKE 'book-%'
                  AND NOT EXISTS (
                      SELECT 1 FROM "Transcripts" ct WHERE ct."FileId" = t.file_id
                  )
                """
            )
        )

    return {
        "files_inserted": files_result.rowcount or 0,
        "metadata_inserted": metadata_result.rowcount or 0,
        "transcripts_inserted": transcripts_result.rowcount or 0,
    }


if __name__ == "__main__":
    result = migrate_books()
    print(result)
