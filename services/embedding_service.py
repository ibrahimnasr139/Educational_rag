"""
Embedding service using Dhakira RAG model.
Handles text embedding and vector database operations.
"""

import os

# Avoid Chroma's Rust SQLite backend panic on incompatible/corrupted persisted DBs.
os.environ.setdefault("CHROMA_RUST_BINDINGS_SKIP", "1")

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import logging
from config.settings import settings
import hashlib
import sys
import asyncio

logger = logging.getLogger(__name__)

# Dhakira imports
try:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _dhakira_src = os.path.join(_repo_root, "Dhakira")
    if os.path.isdir(_dhakira_src) and _dhakira_src not in sys.path:
        sys.path.insert(0, _dhakira_src)
    from dhakira import Memory
    DHAKIRA_AVAILABLE = True
    logger.info("✓ Dhakira available")
except ImportError as e:
    DHAKIRA_AVAILABLE = False
    logger.warning(f"Dhakira not available, using fallback embedding model. Error: {e}")

class EmbeddingService:
    """
    Manages text embeddings using Dhakira RAG model with ChromaDB storage.
    Provides efficient embedding creation and retrieval.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.vector_db_path
        self.client = None
        self.model = None
        self.model_type = None  # Track model type: 'dhakira' or 'sentence_transformer'
        self._fallback_model = None
        self._openai_client = None
        self._embedding_dimension = None
        self._chroma_init_error = None
        
        # Initialize ChromaDB client
        self._init_chroma_client()

        # Embedding models are initialized lazily. Loading Torch/SentenceTransformer
        # at import time is expensive on small Railway instances.
    
    def _init_embedding_model(self):
        """Initialize Dhakira or fallback embedding model."""
        if self.model is not None:
            return

        provider = (settings.embedding_provider or "sentence_transformer").lower()

        if provider == "openai":
            self._init_openai_model()
            return

        if provider == "dhakira" and DHAKIRA_AVAILABLE:
            try:
                logger.info("Initializing Dhakira Memory model using Gemini via OpenAI provider")
                from dhakira.config import DhakiraConfig, LLMConfig
                dhakira_config = DhakiraConfig(
                    llm=LLMConfig(
                        provider="openai",
                        model=settings.openai_model,
                        api_key=settings.openai_api_key,
                        # No base_url needed for OpenAI
                    )
                )
                self.model = Memory(config=dhakira_config)
                self.model_type = "dhakira"
                self._embedding_dimension = getattr(dhakira_config.embeddings, "dim", 128)
                logger.info("Dhakira Memory model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Dhakira initialization failed: {e}")
                logger.info("Falling back to SentenceTransformer")

        if provider == "dhakira" and not DHAKIRA_AVAILABLE:
            logger.warning("Dhakira requested but unavailable; falling back to SentenceTransformer")

        logger.info(f"Initializing fallback model: {settings.embedding_model}")
        self._init_fallback_model()
        self.model_type = "sentence_transformer"
        self.model = self._fallback_model

    def _init_openai_model(self):
        """Initialize lightweight API-based embeddings."""
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        from openai import AsyncOpenAI

        self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model_type = "openai"
        self.model = self._openai_client
        self._embedding_dimension = self._configured_openai_dimension()
        logger.info(f"OpenAI embeddings initialized: {settings.openai_embedding_model}")
    
    def _init_fallback_model(self):
        """Initialize fallback SentenceTransformer model."""
        if self._fallback_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("GPU not available, using CPU")
            self.model = SentenceTransformer(settings.embedding_model, device=device)
            self.model_type = "sentence_transformer"
            self._fallback_model = self.model  # Store reference for emergency fallback
            self._embedding_dimension = self._get_sentence_transformer_dimension(self.model)
            logger.info(f"Fallback model loaded on {device}")
        except ImportError:
            self.model = SentenceTransformer(settings.embedding_model)
            self.model_type = "sentence_transformer"
            self._fallback_model = self.model  # Store reference for emergency fallback
            self._embedding_dimension = self._get_sentence_transformer_dimension(self.model)
            logger.info("Fallback model loaded (CPU)")
    
    def _init_chroma_client(self):
        """Initialize ChromaDB persistent client."""
        logger.info(f"Initializing ChromaDB at {self.db_path}")

        try:
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            self._chroma_init_error = None
            logger.info("ChromaDB initialized successfully")
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as e:
            self.client = None
            self._chroma_init_error = e
            logger.exception(
                "ChromaDB failed to initialize. If this is a Rust bindings panic, "
                "reinstall dependencies with chromadb==0.5.23 and recreate the "
                "persisted vector DB at %s if it was written by an incompatible version.",
                self.db_path,
            )

    def _ensure_chroma_client(self):
        """Return an initialized ChromaDB client or raise a clear service error."""
        if self.client is None:
            if self._chroma_init_error is None:
                self._init_chroma_client()
            if self.client is None:
                raise RuntimeError(
                    "ChromaDB is unavailable. Reinstall with chromadb==0.5.23; "
                    f"if the persisted database is incompatible, delete '{self.db_path}' "
                    "and reprocess the documents."
                ) from self._chroma_init_error
        return self.client

    def _configured_openai_dimension(self) -> int:
        model_name = (settings.openai_embedding_model or "").lower()
        if "3-large" in model_name:
            return 3072
        return 1536

    def _get_sentence_transformer_dimension(self, model) -> int:
        try:
            dimension = model.get_sentence_embedding_dimension()
            if dimension:
                return int(dimension)
        except Exception:
            pass
        return 384

    def _configured_embedding_dimension(self) -> int:
        provider = (settings.embedding_provider or "sentence_transformer").lower()
        if provider == "openai":
            return self._configured_openai_dimension()
        if provider == "dhakira":
            return 128
        return 384

    def _set_embedding_dimension_from_vectors(self, embeddings: List[List[float]]):
        for embedding in embeddings:
            if embedding:
                self._embedding_dimension = len(embedding)
                return

    def _effective_collection_name(self, collection_name: str, dimension: Optional[int] = None) -> str:
        """Avoid mixing embeddings with incompatible dimensions in one collection."""
        if collection_name == "documents":
            dim = dimension or self._embedding_dimension or self._configured_embedding_dimension()
            return f"{collection_name}_{dim}"
        return collection_name
    
    def get_or_create_collection(self, collection_name: str = "documents", dimension: Optional[int] = None):
        """
        Get or create a collection in ChromaDB.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            ChromaDB collection
        """
        try:
            collection_name = self._effective_collection_name(collection_name, dimension)
            client = self._ensure_chroma_client()
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{collection_name}' ready")
            return collection
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            self._init_embedding_model()
            if self.model_type == "dhakira":
                # Dhakira embedding - use add method to get embeddings
                # Add text to memory and immediately retrieve to get embedding
                temp_id = f"temp_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                
                # Validate input
                if not text or not text.strip():
                    logger.warning("Empty text provided for embedding")
                    return []
                
                try:
                    embedding = await asyncio.to_thread(self.model.embed, text)
                    if embedding:
                        self._embedding_dimension = len(embedding)
                except Exception as dhakira_error:
                    logger.warning(f"Dhakira embedding failed, falling back: {dhakira_error}")
                    embedding = []
                
                # Fallback to sentence transformer if Dhakira fails
                if not embedding:
                    self._init_fallback_model()
                    encoded = await asyncio.to_thread(self._fallback_model.encode, text, convert_to_numpy=True)
                    embedding = encoded.tolist()
                    if embedding:
                        self.model_type = "sentence_transformer"
                        self._embedding_dimension = len(embedding)
            else:
                if self.model_type == "openai":
                    response = await self._openai_client.embeddings.create(
                        model=settings.openai_embedding_model,
                        input=text
                    )
                    embedding = response.data[0].embedding
                    self._embedding_dimension = len(embedding)
                    return embedding

                # SentenceTransformer fallback
                encoded = await asyncio.to_thread(self.model.encode, text, convert_to_numpy=True)
                embedding = encoded.tolist()
                if embedding:
                    self._embedding_dimension = len(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            # Return empty embedding as fallback
            return []
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        try:
            self._init_embedding_model()
            if self.model_type == "dhakira":
                # Dhakira batch embedding - add all texts then search
                embeddings = []
                temp_ids = []
                
                # Validate inputs
                valid_texts = [text for text in texts if text and text.strip()]
                if not valid_texts:
                    logger.warning("No valid texts provided for batch embedding")
                    return []
                
                try:
                    embeddings_valid = await asyncio.to_thread(self.model.embed_batch, valid_texts)
                    self._set_embedding_dimension_from_vectors(embeddings_valid)
                    # Map back to original texts to maintain length
                    embeddings = []
                    valid_idx = 0
                    for text in texts:
                        if text and text.strip() and valid_idx < len(embeddings_valid):
                            embeddings.append(embeddings_valid[valid_idx])
                            valid_idx += 1
                        else:
                            embeddings.append([])
                    return embeddings
                except Exception as dhakira_error:
                    logger.warning(f"Dhakira batch embedding failed, falling back: {dhakira_error}")
                    self._init_fallback_model()
                    encoded = await asyncio.to_thread(self._fallback_model.encode, texts, convert_to_numpy=True)
                    embeddings = encoded.tolist()
                    self.model_type = "sentence_transformer"
                    self._set_embedding_dimension_from_vectors(embeddings)
            else:
                if self.model_type == "openai":
                    response = await self._openai_client.embeddings.create(
                        model=settings.openai_embedding_model,
                        input=texts
                    )
                    embeddings = [item.embedding for item in response.data]
                    self._set_embedding_dimension_from_vectors(embeddings)
                    return embeddings

                # SentenceTransformer batch embedding
                encoded = await asyncio.to_thread(self.model.encode, texts, convert_to_numpy=True)
                embeddings = encoded.tolist()
                self._set_embedding_dimension_from_vectors(embeddings)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Return empty list as fallback
            return []
    
    async def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        file_id: str,
        start_idx: int = 0,
        collection_name: str = "documents"
    ) -> List[str]:
        """
        Add documents to vector database.
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dictionaries
            file_id: File identifier
            collection_name: Target collection
            
        Returns:
            List of document IDs
        """
        try:
            # Generate embeddings
            embeddings = await self.embed_batch(texts)
            self._set_embedding_dimension_from_vectors(embeddings)
            collection = self.get_or_create_collection(collection_name, self._embedding_dimension)
            
            if settings.save_chunks_to_postgres:
                from services.database_service import database_service
                import asyncio
                model_name = self.model_type or "sentence-transformer"
                try:
                    await asyncio.to_thread(
                        database_service.save_chunks,
                        file_id=file_id,
                        chunks=texts,
                        embeddings=embeddings,
                        model_name=model_name,
                        metadatas=metadatas,
                        start_idx=start_idx
                    )
                except Exception as db_e:
                    logger.error(f"Failed to save chunks to DB: {db_e}")
            
            # Generate unique IDs
            ids = [
                self._generate_doc_id(file_id, start_idx + idx)
                for idx in range(len(texts))
            ]
            
            # Keep only successful embeddings to avoid Chroma dimension errors
            valid_items = [(i, text, meta, emb) for i, (text, meta, emb) in enumerate(zip(texts, metadatas, embeddings)) if emb]
            if not valid_items:
                logger.warning("No valid embeddings generated for this batch")
                return []

            valid_ids = [ids[i] for i, _, _, _ in valid_items]
            valid_texts = [text for _, text, _, _ in valid_items]
            valid_metadatas = [meta for _, _, meta, _ in valid_items]
            valid_embeddings = [emb for _, _, _, emb in valid_items]

            # Upsert keeps repeated processing of the same fileId idempotent.
            collection.upsert(
                embeddings=valid_embeddings,
                documents=valid_texts,
                metadatas=valid_metadatas,
                ids=valid_ids
            )
            
            logger.info(f"Added {len(texts)} documents to collection '{collection_name}'")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        collection_name: str = "documents",
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            collection_name: Collection to search
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with text, metadata, and scores
        """
        try:
            # Generate query embedding
            query_embedding = await self.embed_text(query)
            
            if not query_embedding:
                return []
            collection = self.get_or_create_collection(collection_name, len(query_embedding))

            clean_filter = {k: v for k, v in (filter_metadata or {}).items() if v not in (None, '', [], {})}

            # Search asynchronously
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=clean_filter or None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0 and results['documents'][0]:
                for idx in range(len(results['documents'][0])):
                    formatted_results.append({
                        'text': results['documents'][0][idx],
                        'metadata': results['metadatas'][0][idx] if results['metadatas'] and len(results['metadatas'][0]) > idx else {},
                        'score': 1.0 - results['distances'][0][idx] if results['distances'] and len(results['distances'][0]) > idx else 0.0
                    })
            
            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def _generate_doc_id(self, file_id: str, index: int) -> str:
        """
        Generate unique document ID.
        
        Args:
            file_id: File identifier
            index: Chunk index
            
        Returns:
            Unique document ID
        """
        # Create deterministic ID
        content = f"{file_id}_{index}"
        doc_id = hashlib.md5(content.encode()).hexdigest()
        return doc_id
    
    async def delete_file_documents(
        self,
        file_id: str,
        collection_name: str = "documents"
    ):
        """
        Delete all documents associated with a file.
        
        Args:
            file_id: File identifier
            collection_name: Collection name
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # Query for ALL documents with this file_id (high limit to avoid truncation)
            results = collection.get(
                where={"file_id": file_id},
                limit=10000
            )
            
            if results and results['ids']:
                collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents for file {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")

    def get_all_chunks_for_file(
        self,
        file_id: str,
        collection_name: str = "documents"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ALL stored chunks for a given file_id, sorted by chunk_id.
        Used by SummaryService to reconstruct the full document text.

        Args:
            file_id:         The file identifier used during embedding.
            collection_name: Collection to query (default: "documents").

        Returns:
            List of {"text": str, "metadata": dict} sorted by chunk_id.
            Empty list if file_id not found.
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            results = collection.get(
                where={"file_id": file_id},
                include=["documents", "metadatas"],
                limit=10000  # Default is 10, increasing to 10k to ensure full text reconstruction
            )

            if not results or not results.get("documents"):
                logger.info(f"No chunks found for file_id='{file_id}'")
                return []

            chunks = [
                {"text": text, "metadata": meta or {}}
                for text, meta in zip(results["documents"], results["metadatas"])
            ]

            # Sort by original chunk order
            chunks.sort(key=lambda c: c["metadata"].get("chunk_id", 0))
            logger.info(f"Retrieved {len(chunks)} chunks for file_id='{file_id}'")
            return chunks

        except Exception as e:
            logger.error(f"get_all_chunks_for_file failed for '{file_id}': {e}")
            return []
    
    def get_collection_stats(self, collection_name: str = "documents") -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Statistics dictionary
        """
        try:
            effective_collection_name = self._effective_collection_name(collection_name)
            collection = self.get_or_create_collection(collection_name)
            count = collection.count()
            
            return {
                "collection_name": effective_collection_name,
                "document_count": count
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Global instance
embedding_service = EmbeddingService()
