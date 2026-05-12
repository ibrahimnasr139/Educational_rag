"""
RAG service integrating Dhakira embeddings with Google Gemini for generation.
Handles retrieval-augmented generation for question creation and descriptions.
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging
import json
from config.settings import settings
from services.embedding_service import embedding_service
from utils.language_detector import language_detector

logger = logging.getLogger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation service.
    Combines semantic search with Google Gemini for context-aware generation.
    """
    
    def __init__(self):
        # Determine provider
        self.provider = settings.llm_provider.lower()
        # Always define generation_config because structured generation uses it even when OpenAI is selected.
        self.generation_config = {
            "temperature": settings.gemini_temperature,
            "max_output_tokens": settings.gemini_max_tokens if settings.gemini_max_tokens > 4096 else 8192,
        }
        # Always initialize OpenAI if key is present to allow forcing
        self.openai_llm = None
        self.gpt4_nano_llm = None
        if getattr(settings, "openai_api_key", None):
            try:
                from dhakira.config import LLMConfig
                from dhakira.llm.openai_ import OpenAILLM
                
                config = LLMConfig(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                    temperature=settings.openai_temperature,
                    max_tokens=settings.openai_max_tokens
                )
                self.openai_llm = OpenAILLM(config)
                
                # Special instance for gpt-4.1-nano (custom model requested by user)
                gpt4_nano_config = LLMConfig(
                    api_key=settings.openai_api_key,
                    model="gpt-4.1-nano",
                    temperature=0.7, 
                    max_tokens=settings.openai_max_tokens
                )
                self.gpt4_nano_llm = OpenAILLM(gpt4_nano_config)
                logger.info(f"OpenAI fallback initialized ({settings.openai_model} & gpt-4.1-nano)")
            except ImportError as e:
                logger.warning(f"Dhakira not available, OpenAI fallback disabled. Error: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI fallback: {e}")

        if self.provider == "openai" and self.openai_llm:
            self.llm = self.openai_llm
            logger.info(f"RAG Service default initialized with OpenAI ({settings.openai_model})")
        else:
            if self.provider == "openai" and not self.openai_llm:
                logger.warning("LLM provider is set to 'openai' but OpenAI fallback is unavailable; falling back to Gemini")
                self.provider = "gemini"
            # Configure Google Gemini
            genai.configure(api_key=settings.google_api_key)
            self.model = genai.GenerativeModel(settings.gemini_model)
            
            logger.info(f"RAG Service default initialized with Gemini ({settings.gemini_model})")
    
    async def retrieve_context(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector database.
        
        Args:
            query: Search query
            n_results: Number of results
            metadata_filter: Optional metadata filters
            
        Returns:
            List of relevant contexts
        """
        try:
            results = await embedding_service.search(
                query=query,
                n_results=n_results,
                filter_metadata=metadata_filter
            )
            
            logger.info(f"Retrieved {len(results)} context chunks")
            return results
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []

    async def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Metadata-aware hybrid retrieval.
        Combines semantic search results with metadata match scoring.
        Formula: final_score = (semantic_score * 0.7) + (metadata_score * 0.3)
        """
        from utils.query_metadata_extractor import extract_query_metadata
        
        # 1. Extract and merge filters
        extracted = extract_query_metadata(query)
        filters = {**extracted, **(metadata_filter or {})}
        filters = {k: v for k, v in filters.items() if v not in (None, '', [], {})}

        # 2. Semantic Search (ChromaDB)
        # We fetch more than top_k to allow re-ranking
        semantic_results = await embedding_service.search(
            query=query,
            n_results=top_k * 3,
            filter_metadata=None # Don't hard-filter if we want true hybrid, or keep it for efficiency
        )

        if not semantic_results:
            return []

        # 3. Apply Hybrid Scoring
        scored: List[Dict[str, Any]] = []
        for res in semantic_results:
            semantic_score = res.get('score', 0.0)
            meta = res.get('metadata') or {}
            
            # Calculate metadata match score
            matched = 0
            total = 0
            for key, expected in filters.items():
                total += 1
                # Handle common aliases
                candidates = [meta.get(key)]
                if key == 'grade_level': candidates.append(meta.get('grade'))
                if key == 'semester': candidates.append(meta.get('term'))
                
                if expected in candidates:
                    matched += 1
            
            metadata_score = (matched / total) if total > 0 else 0.0
            
            # Hybrid Formula: 0.7 Semantic + 0.3 Metadata
            final_score = (semantic_score * 0.7) + (metadata_score * 0.3)
            
            if final_score >= min_score:
                scored.append({
                    'text': res.get('text', ''),
                    'metadata': meta,
                    'score': final_score,
                    'semantic_score': semantic_score,
                    'metadata_score': metadata_score,
                })

        # 4. Re-sort and limit
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

    async def generate_directly(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        force_provider: Optional[str] = None
    ) -> str:
        """
        Generate response using selected provider directly without context.
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            force_provider: Optional override to force 'openai', 'gpt-4o-mini', or 'gemini'
            
        Returns:
            Generated response
        """
        try:
            # Build full prompt with system instruction (mostly for Gemini)
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
            else:
                full_prompt = prompt
            
            # Generate response
            provider_to_use = (force_provider or self.provider).lower()
            if provider_to_use == "gpt-4.1-nano" and self.gpt4_nano_llm:
                return await self.gpt4_nano_llm.generate(prompt=prompt, system=system_instruction)
            elif provider_to_use == "openai" and self.openai_llm:
                return await self.openai_llm.generate(prompt=prompt, system=system_instruction)
            else:
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=self.generation_config
                )
                return response.text
            
        except Exception as e:
            logger.error(f"Direct generation failed: {e}")
            raise
    
    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, Any]],
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate response using retrieved context and Gemini.
        
        Args:
            prompt: User prompt
            context: Retrieved context chunks
            system_instruction: Optional system instruction
            
        Returns:
            Generated response
        """
        try:
            # Build context string
            context_str = self._build_context_string(context)
            
            # Build full prompt
            full_prompt = self._build_prompt(prompt, context_str, system_instruction)
            
            # Generate response
            if self.provider == "openai":
                return await self.llm.generate(prompt=full_prompt)
            else:
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=self.generation_config
                )
                return response.text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _build_context_string(self, context: List[Dict[str, Any]]) -> str:
        """Build formatted context string from retrieved chunks."""
        if not context:
            return ""
        
        context_parts = []
        for idx, item in enumerate(context, 1):
            text = item.get('text', '')
            score = item.get('score', 0)
            context_parts.append(f"[{idx}] (Relevance: {score:.2f})\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        user_prompt: str,
        context: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """Build complete prompt with context and instructions."""
        parts = []
        
        if system_instruction:
            parts.append(f"=== INSTRUCTIONS ===\n{system_instruction}\n")
        
        if context:
            parts.append(f"=== CONTEXT ===\n{context}\n")
        
        parts.append(f"=== USER REQUEST ===\n{user_prompt}")
        
        return "\n".join(parts)
    
    async def rag_query(
        self,
        query: str,
        n_context: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            query: User query
            n_context: Number of context chunks to retrieve
            metadata_filter: Optional metadata filters
            system_instruction: Optional system instruction
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve context with metadata-aware hybrid retrieval
        context = await self.retrieve_with_metadata(
            query=query,
            top_k=n_context,
            metadata_filter=metadata_filter
        )
        
        # Generate response
        response = await self.generate_with_context(
            prompt=query,
            context=context,
            system_instruction=system_instruction
        )
        
        return {
            "response": response,
            "context_used": len(context),
            "sources": [c.get('metadata', {}) for c in context]
        }
    
    async def generate_structured_output(
        self,
        prompt: str,
        context: List[Dict[str, Any]],
        output_schema: Dict[str, Any],
        system_instruction: Optional[str] = None
    ) -> Any:
        """
        Generate structured JSON output using RAG.
        
        Args:
            prompt: User prompt
            context: Retrieved context
            output_schema: Expected output structure
            system_instruction: Optional system instruction
            
        Returns:
            Parsed structured output
        """
        # Add JSON formatting instruction
        json_instruction = (
            f"\n\nYou must respond with ONLY valid JSON matching this schema:\n"
            f"{json.dumps(output_schema, indent=2)}\n"
            f"Do not include any text before or after the JSON."
        )
        
        full_instruction = system_instruction or ""
        full_instruction += json_instruction
        
        # Use higher token limit for question generation
        question_config = self.generation_config.copy()
        question_config["max_output_tokens"] = 8192  # Ensure enough tokens for multiple questions
        
        # Generate
        response_text = await self.generate_with_context(
            prompt=prompt,
            context=context,
            system_instruction=full_instruction
        )
        
        # Parse JSON
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            cleaned = cleaned.strip()
            
            # Check if JSON is truncated (common issue with long responses)
            if not cleaned.endswith('}') and not cleaned.endswith(']'):
                logger.warning("Response appears to be truncated, attempting to fix...")
                # Try to fix truncated JSON by adding missing brackets
                open_brackets = cleaned.count('{') - cleaned.count('}')
                open_arrays = cleaned.count('[') - cleaned.count(']')
                
                for _ in range(open_brackets):
                    cleaned += '}'
                for _ in range(open_arrays):
                    cleaned += ']'
            
            parsed = json.loads(cleaned)
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response_text[:500]}...")  # Log first 500 chars
            logger.error(f"Cleaned response: {cleaned[:500]}...")  # Log first 500 chars
            raise ValueError("Generated response was not valid JSON")


# Global instance
rag_service = RAGService()
