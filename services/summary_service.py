from __future__ import annotations

import logging
from typing import Optional
from services.embedding_service import embedding_service
from services.rag_service import rag_service
from utils.language_detector import language_detector

logger = logging.getLogger(__name__)


class SummaryService:
    async def summarise_file(self, file_id: str, summary_length: str = 'medium', language: Optional[str] = None) -> dict:
        chunks = embedding_service.get_all_chunks_for_file(file_id)
        if not chunks:
            raise ValueError(f'No chunks found for file_id: {file_id}')

        text = '\n\n'.join(c.get('text', '') for c in chunks[:30])
        source_lang = language_detector.detect_language(text)
        out_lang = language or source_lang or 'ar'
        words = {'short': '80-120', 'medium': '150-250', 'long': '300-500'}.get(summary_length, '150-250')
        is_ar = out_lang == 'ar'

        prompt = f"""
Summarize the following educational content.
Output language: {'Arabic' if is_ar else 'English'}.
Length: {words} words.
Return JSON only with keys: summary, keyPoints.

CONTENT:
{text[:14000]}
"""
        schema = {
            'type': 'object',
            'properties': {
                'summary': {'type': 'string'},
                'keyPoints': {'type': 'array', 'items': {'type': 'string'}},
            },
            'required': ['summary', 'keyPoints'],
        }
        try:
            raw = await rag_service.generate_structured_output(
                prompt=prompt,
                context=[],
                output_schema=schema,
                system_instruction='You are an accurate educational summarizer. Return valid JSON only.',
            )
        except Exception as exc:
            logger.warning('Structured summary failed, using direct fallback: %s', exc)
            summary = await rag_service.generate_directly(prompt=prompt, system_instruction='Return a concise summary only.')
            raw = {'summary': summary, 'keyPoints': []}

        return {
            'fileId': file_id,
            'summary': raw.get('summary', ''),
            'keyPoints': raw.get('keyPoints', []) or [],
            'language': out_lang,
            'sourceLanguage': source_lang,
            'fileType': (chunks[0].get('metadata') or {}).get('file_type'),
            'wordCount': len((raw.get('summary') or '').split()),
            'chunksUsed': len(chunks),
        }


summary_service = SummaryService()
