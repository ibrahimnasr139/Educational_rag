"""
Service for generating AI-powered suggestions for metadata and topics based on transcripts.
"""

import logging
import json
from typing import Dict, Any, List
from services.rag_service import rag_service
from services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class SuggestService:
    """Service to suggest metadata and topics from transcribed content."""
    
    def __init__(self):
        self.rag = rag_service

    async def suggest_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Analyze the transcript of a file and suggest metadata.
        
        Args:
            file_id: The ID of the file to analyze
            
        Returns:
            Dictionary with suggested subject, grade, title, and module.
        """
        try:
            # 1. Fetch chunks for this file
            chunks = embedding_service.get_all_chunks_for_file(file_id)
            if not chunks:
                logger.warning(f"No chunks found for file {file_id}")
                return self._get_empty_suggestion()

            # 2. Get a representative sample (first few chunks)
            sample_text = "\n\n".join([c['text'] for c in chunks[:5]])
            
            # 3. Ask AI to analyze
            prompt = f"""
            Analyze the following transcript sample and suggest educational metadata.
            
            TRANSCRIPT SAMPLE:
            {sample_text}
            
            Return a JSON object with:
            - subject: The academic subject (e.g., Math, Science, History, Physics)
            - grade: The appropriate grade level (e.g., Elementary, High School, University)
            - title: A concise, descriptive title for this specific topic
            - module: The general module or chapter name
            - prompt: A suggested search prompt for generating questions (e.g., "Explain the laws of thermodynamics")
            """
            
            schema = {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "grade": {"type": "string"},
                    "title": {"type": "string"},
                    "module": {"type": "string"},
                    "prompt": {"type": "string"}
                },
                "required": ["subject", "grade", "title", "module", "prompt"]
            }
            
            suggestions = await self.rag.generate_structured_output(
                prompt=prompt,
                context=[], # No RAG needed for self-analysis
                output_schema=schema,
                system_instruction="You are an educational curriculum expert. Suggest accurate metadata in the language of the transcript (Arabic or English)."
            )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest metadata: {e}")
            return self._get_empty_suggestion()

    async def suggest_topics(self, file_id: str) -> List[str]:
        """Suggest 3 specific topics/prompts for question generation."""
        try:
            chunks = embedding_service.get_all_chunks_for_file(file_id)
            if not chunks:
                return []
                
            sample_text = "\n\n".join([c['text'] for c in chunks[:10]])
            
            prompt = f"Based on this transcript, suggest 3 specific topics or concepts to generate questions about:\n\n{sample_text}"
            
            schema = {
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 3
                    }
                },
                "required": ["topics"]
            }
            
            result = await self.rag.generate_structured_output(
                prompt=prompt,
                context=[],
                output_schema=schema,
                system_instruction="Provide 3 distinct, concise educational topics in the same language as the transcript."
            )
            
            return result.get('topics', [])
            
        except Exception as e:
            logger.error(f"Failed to suggest topics: {e}")
            return []

    def _get_empty_suggestion(self) -> Dict[str, Any]:
        return {
            "subject": "",
            "grade": "",
            "title": "",
            "module": "",
            "prompt": ""
        }

# Global instance
suggest_service = SuggestService()
