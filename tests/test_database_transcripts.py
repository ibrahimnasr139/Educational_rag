from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.db_models import Base, Files, Transcripts, VideoTimestamps
from services.database_service import DatabaseService, EXPECTED_SCHEMA


def _sqlite_service():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    service = DatabaseService.__new__(DatabaseService)
    service.engine = engine
    service.SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    return service


def test_database_models_use_exact_pascal_case_schema_names():
    actual = {
        table.name: {column.name for column in table.columns}
        for table in Base.metadata.sorted_tables
    }

    for table_name, expected_columns in EXPECTED_SCHEMA.items():
        assert table_name in actual
        assert expected_columns <= actual[table_name]


def test_save_transcript_upserts_text_and_timestamps_atomically():
    service = _sqlite_service()
    with service.get_session() as session:
        session.add(Files(id="file-1", name="audio.mp3", type="audio"))
        session.commit()

    service.save_transcript(
        "file-1",
        "first\x00 transcript",
        "ar",
        segments=[{"text": "part one", "start": 1.25, "end": 2.5}],
    )
    service.save_transcript(
        "file-1",
        "updated transcript",
        "en",
        segments=[{"text": "replacement", "start": 3, "end": 4}],
    )

    with service.get_session() as session:
        transcripts = session.query(Transcripts).all()
        timestamps = session.query(VideoTimestamps).all()

        assert len(transcripts) == 1
        assert transcripts[0].full_text == "updated transcript"
        assert transcripts[0].language == "en"
        assert len(timestamps) == 1
        assert timestamps[0].text == "replacement"
        assert timestamps[0].start_time == 3.0


def test_save_transcript_rolls_back_text_when_a_timestamp_is_invalid():
    service = _sqlite_service()
    with service.get_session() as session:
        session.add(Files(id="file-2", name="audio.mp3", type="audio"))
        session.commit()

    service.save_transcript("file-2", "original", "ar", segments=[])

    try:
        service.save_transcript(
            "file-2",
            "must not persist",
            "en",
            segments=[{"text": "bad", "start": "not-a-number", "end": 2}],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid timestamps must fail the transaction")

    with service.get_session() as session:
        transcript = session.query(Transcripts).one()
        assert transcript.full_text == "original"
        assert transcript.language == "ar"


def test_document_pipeline_does_not_hide_transcript_database_failures():
    import importlib

    module = importlib.import_module("services.document_processing_service")

    class FailingDatabase:
        def save_transcript(self, **_kwargs):
            raise RuntimeError("database unavailable")

    original_database = module.database_service
    original_file_setting = module.settings.save_transcript_files
    module.database_service = FailingDatabase()
    module.settings.save_transcript_files = False
    try:
        processor = module.DocumentProcessingService.__new__(
            module.DocumentProcessingService
        )
        try:
            processor._save_full_transcript(
                "file-3", "unused.wav", "text", "ar", []
            )
        except RuntimeError as exc:
            assert str(exc) == "database unavailable"
        else:
            raise AssertionError("database failures must fail the processing job")
    finally:
        module.database_service = original_database
        module.settings.save_transcript_files = original_file_setting
