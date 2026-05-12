"""
Text chunking utilities for document processing.
Implements intelligent chunking with overlap for better context preservation.
"""

from typing import List, Dict, Any
from config.settings import settings
import re


class TextChunker:
    """
    Chunks text into overlapping segments for embedding.
    Optimized for Arabic and multilingual content.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of chunks with metadata
        """
        if not text:
            return []
        
        # Clean and normalize text
        text = self._normalize_text(text)
        
        # Split into sentences first for better chunking
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If single sentence exceeds chunk size, split it
            if sentence_length > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # If adding sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk,
                    self.chunk_overlap
                )
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Create chunk objects with metadata
        chunk_objects = []
        for idx, chunk_text in enumerate(chunks):
            # Create a copy of metadata and add chunk_id
            chunk_meta = (metadata or {}).copy()
            chunk_meta["chunk_id"] = idx
            
            chunk_obj = {
                "text": chunk_text,
                "chunk_id": idx,
                "metadata": chunk_meta
            }
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
        return chunk_objects
    
    def chunk_whisper_segments(
        self,
        segments: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Group Whisper segments into chunks while preserving timestamps.
        
        Args:
            segments: List of Whisper segment dicts (text, start, end, etc.)
            metadata: Base metadata to attach to chunks
            
        Returns:
            List of chunks with timestamp metadata
        """
        if not segments:
            return []
            
        chunks = []
        current_chunk_text = []
        current_length = 0
        current_start_time = segments[0]['start']
        
        for segment in segments:
            text = segment['text'].strip()
            length = len(text)
            
            # If adding this segment exceeds chunk size, save current chunk
            if current_length + length > self.chunk_size and current_chunk_text:
                chunks.append({
                    "text": " ".join(current_chunk_text),
                    "timestamp": self._format_timestamp(current_start_time)
                })
                
                # Setup next chunk (Whisper segments don't easily allow overlap without splitting sentences)
                # For transcripts, we prioritize timing accuracy over overlap context
                current_chunk_text = [text]
                current_length = length
                current_start_time = segment['start']
            else:
                current_chunk_text.append(text)
                current_length += length
        
        # Add final chunk
        if current_chunk_text:
            chunks.append({
                "text": " ".join(current_chunk_text),
                "timestamp": self._format_timestamp(current_start_time)
            })
            
        # Create final chunk objects
        chunk_objects = []
        for idx, chunk_data in enumerate(chunks):
            # Merge base metadata with specific timestamp
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata["timestamp"] = chunk_data["timestamp"]
            chunk_metadata["chunk_id"] = idx
            
            chunk_objects.append({
                "text": chunk_data["text"],
                "chunk_id": idx,
                "metadata": chunk_metadata
            })
            
        return chunk_objects

    def _format_timestamp(self, seconds: float) -> str:
        """Convert float seconds to HH:MM:SS string."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and special characters."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        Handles Arabic and English sentence boundaries.
        """
        # Arabic and English sentence terminators
        sentence_endings = r'[.!?؟۔।]'
        
        # Split on sentence endings followed by space or end of string
        sentences = re.split(f'({sentence_endings})\\s+', text)
        
        # Recombine sentences with their terminators
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            if sentence.strip():
                result.append(sentence.strip())
        
        # Handle last sentence if no terminator
        if sentences and not re.search(sentence_endings, sentences[-1]):
            if sentences[-1].strip():
                result.append(sentences[-1].strip())
        
        return result if result else [text]
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence into smaller chunks."""
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        target_overlap: int
    ) -> List[str]:
        """Get last few sentences for overlap."""
        overlap = []
        overlap_length = 0
        
        for sentence in reversed(sentences):
            sentence_length = len(sentence)
            if overlap_length + sentence_length > target_overlap:
                break
            overlap.insert(0, sentence)
            overlap_length += sentence_length
        
        return overlap


# Global instance
text_chunker = TextChunker()
