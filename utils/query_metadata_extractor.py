from __future__ import annotations

from typing import Dict
from utils.metadata_extractor import extract_metadata_from_filename, compact_metadata


def extract_query_metadata(query: str) -> Dict[str, str]:
    """Extract metadata filters from natural-language user queries."""
    meta = compact_metadata(extract_metadata_from_filename(query or ''))
    filters: Dict[str, str] = {}
    if meta.get('subject'):
        filters['subject'] = meta['subject']
    if meta.get('grade') or meta.get('grade_level'):
        filters['grade_level'] = meta.get('grade_level') or meta.get('grade')
    if meta.get('semester') or meta.get('term'):
        filters['semester'] = meta.get('semester') or meta.get('term')
    if meta.get('book'):
        filters['book'] = meta['book']
    return filters
