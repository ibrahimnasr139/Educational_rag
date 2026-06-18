from __future__ import annotations

import re
from typing import Any, Dict, Optional


def _normalize_ar(text: str) -> str:
    text = (text or '').lower().replace('_', ' ').replace('-', ' ')
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه').replace('ى', 'ي')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


SUBJECT_RULES = [
    ('علوم متكاملة', ['علوم متكامله', 'علوم متكاملة', 'integrated science', 'science']),
    ('رياضيات', ['رياضيات', 'math', 'mathematics']),
    ('لغة عربية', ['لغه عربيه', 'لغة عربية', 'عربي', 'arabic']),
    ('لغة إنجليزية', ['لغه انجليزيه', 'لغة إنجليزية', 'انجليزي', 'english']),
    ('فيزياء', ['فيزياء', 'physics']),
    ('كيمياء', ['كيمياء', 'chemistry']),
    ('أحياء', ['احياء', 'biology']),
    ('تاريخ', ['تاريخ', 'history']),
    ('جغرافيا', ['جغرافيا', 'geography']),
]

GRADE_RULES = [
    ('أولى ثانوي', ['اولي ثانوي', 'اولى ثانوي', 'الصف الاول الثانوي', '1 ثانوي', '1sec', 'sec1', 'first secondary']),
    ('ثانية ثانوي', ['ثانيه ثانوي', 'ثانية ثانوي', 'الصف الثاني الثانوي', '2 ثانوي', '2sec', 'sec2', 'second secondary']),
    ('ثالثة ثانوي', ['ثالثه ثانوي', 'ثالثة ثانوي', 'الصف الثالث الثانوي', '3 ثانوي', '3sec', 'sec3', 'third secondary']),
    ('أولى إعدادي', ['اولي اعدادي', 'اولى اعدادي', '1 اعدادي', 'prep1']),
    ('ثانية إعدادي', ['ثانيه اعدادي', 'ثانية اعدادي', '2 اعدادي', 'prep2']),
    ('ثالثة إعدادي', ['ثالثه اعدادي', 'ثالثة اعدادي', '3 اعدادي', 'prep3']),
]

TERM_RULES = [
    ('ترم أول', ['ترم اول', 'ترم 1', 'ترم1', 'الفصل الاول', 'term 1', 'term1', 't1']),
    ('ترم ثاني', ['ترم ثاني', 'ترم تاني', 'ترم 2', 'ترم2', 'الفصل الثاني', 'term 2', 'term2', 't2']),
]

BOOK_RULES = [
    ('المعاصر', ['المعاصر']),
    ('الامتحان', ['الامتحان']),
    ('الأضواء', ['الاضواء', 'الأضواء']),
    ('سلاح التلميذ', ['سلاح التلميذ']),
]


def _find_value(name: str, rules) -> Optional[str]:
    for value, keywords in rules:
        for kw in keywords:
            if _normalize_ar(kw) in name:
                return value
    return None


def extract_metadata_from_filename(filename: str) -> Dict[str, Optional[str]]:
    """Extract stable educational metadata from Arabic/English file names."""
    clean = _normalize_ar(filename.rsplit('.', 1)[0])
    return {
        'subject': _find_value(clean, SUBJECT_RULES),
        'grade': _find_value(clean, GRADE_RULES),
        'grade_level': _find_value(clean, GRADE_RULES),
        'term': _find_value(clean, TERM_RULES),
        'semester': _find_value(clean, TERM_RULES),
        'book': _find_value(clean, BOOK_RULES),
    }

def compact_metadata(metadata: Dict[str, Any] | None) -> Dict[str, str]:
    """
    Convert all metadata values to strings because
    ChromaDB filtering becomes more predictable when
    keys and values are stored as strings only.
    """

    if not metadata:
        return {}

    result: Dict[str, str] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, bool):
            result[str(key)] = "true" if value else "false"

        elif isinstance(value, (int, float)):
            result[str(key)] = str(value)

        elif isinstance(value, (list, tuple, set)):
            result[str(key)] = ",".join(map(str, value))

        else:
            result[str(key)] = str(value)

    return result