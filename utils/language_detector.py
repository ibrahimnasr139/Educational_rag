"""
Language detection and management utilities.
Handles Arabic-first language detection with fallback support.
"""

from langdetect import detect, LangDetectException
from typing import Optional
from config.settings import settings
import re


class LanguageDetector:
    """Detects and manages language preferences with Arabic-first approach."""
    
    def __init__(self):
        self.default_lang = settings.default_language
        self.fallback_lang = settings.fallback_language
        self.supported_langs = settings.supported_languages_list
    
    def detect_language(self, text: str) -> str:
        """
        Detect language from text with Arabic bias.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language code (ar, en, fr, etc.)
        """
        if not text or not text.strip():
            return self.default_lang
        
        # Check for Arabic characters
        if self._has_arabic(text):
            return "ar"
        
        try:
            detected = detect(text)
            # Map detected language to supported languages
            if detected in self.supported_langs:
                return detected
            return self.fallback_lang
        except LangDetectException:
            return self.default_lang
    
    def _has_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        return bool(arabic_pattern.search(text))
    
    def get_preferred_language(self, text: Optional[str] = None) -> str:
        """
        Get preferred language for response generation.
        Defaults to Arabic unless text is clearly in another language.
        
        Args:
            text: Optional context text
            
        Returns:
            Language code
        """
        if text:
            detected = self.detect_language(text)
            # Strong preference for Arabic
            if detected == "ar" or self._has_arabic(text):
                return "ar"
            return detected
        return self.default_lang
    
    def should_use_arabic(self, text: Optional[str] = None) -> bool:
        """Determine if Arabic should be used for generation."""
        lang = self.get_preferred_language(text)
        return lang == "ar"
    
    def get_language_name(self, code: str) -> str:
        """Get language name from code."""
        language_names = {
            "ar": "العربية",
            "en": "English",
            "fr": "Français",
            "es": "Español"
        }
        return language_names.get(code, code)


# Global instance
language_detector = LanguageDetector()
