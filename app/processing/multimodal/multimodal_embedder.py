"""Multimodal embedding generation for text and visual content."""

import logging
from typing import Any

import numpy as np

from app.config import get_settings
from app.models.chunk import EmbeddedChunk, MultimodalChunk, TextChunk, VisualChunk

logger = logging.getLogger(__name__)


class MultimodalEmbedder:
    """Embedder for multimodal content using text and vision encoders."""

    def __init__(
        self,
        text_model: str | None = None,
        vision_model: str | None = None,
    ):
        """Initialize multimodal embedder.
        
        Args:
            text_model: Text embedding model name
            vision_model: Vision embedding model name
        """
        settings = get_settings()
        self.text_model = text_model or settings.openai_embedding_model
        self.vision_model = vision_model or settings.vision_encoder
        
        logger.info(
            f"Initialized MultimodalEmbedder: text={self.text_model}, vision={self.vision_model}"
        )
        
        # Initialize text embedder (OpenAI)
        self._init_text_embedder()
        
        # Initialize vision embedder (CLIP)
        self._init_vision_embedder()

    def _init_text_embedder(self):
        """Initialize text embedding client."""
        try:
            from openai import AsyncOpenAI
            
            settings = get_settings()
            if settings.openai_api_key:
                self.text_client = AsyncOpenAI(api_key=settings.openai_api_key)
                self.text_enabled = True
                logger.info("Text embedder initialized")
            else:
                logger.warning("OpenAI API key not set, text embedding disabled")
                self.text_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize text embedder: {e}")
            self.text_enabled = False

    def _init_vision_embedder(self):
        """Initialize vision embedding model."""
        try:
            # Try to load CLIP model
            import torch
            from transformers import CLIPModel, CLIPProcessor
            
            logger.info(f"Loading vision model: {self.vision_model}")
            self.vision_processor = CLIPProcessor.from_pretrained(self.vision_model)
            self.vision_model_obj = CLIPModel.from_pretrained(self.vision_model)
            self.vision_enabled = True
            logger.info("Vision embedder initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize vision embedder: {e}")
            self.vision_enabled = False

    async def embed_text_chunks(
        self, chunks: list[TextChunk]
    ) -> list[EmbeddedChunk]:
        """Generate text embeddings.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embedded chunks
        """
        if not self.text_enabled:
            logger.warning("Text embedding not enabled")
            return []
        
        embedded_chunks = []
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]
            
            try:
                # Call OpenAI embedding API
                response = await self.text_client.embeddings.create(
                    model=self.text_model, input=texts
                )
                
                # Create embedded chunks
                for chunk, embedding_obj in zip(batch, response.data):
                    embedded = EmbeddedChunk(
                        chunk=chunk,
                        embedding=embedding_obj.embedding,
                        modality="text",
                    )
                    embedded_chunks.append(embedded)
            
            except Exception as e:
                logger.error(f"Failed to embed text batch: {e}")
                # Continue with next batch
        
        logger.info(f"Embedded {len(embedded_chunks)} text chunks")
        return embedded_chunks

    async def embed_visual_chunks(
        self, chunks: list[VisualChunk]
    ) -> list[EmbeddedChunk]:
        """Generate visual embeddings using vision encoder.
        
        Args:
            chunks: List of visual chunks
            
        Returns:
            List of embedded chunks
        """
        if not self.vision_enabled:
            logger.warning("Vision embedding not enabled")
            return []
        
        embedded_chunks = []
        
        for chunk in chunks:
            try:
                # Load image from bytes
                from PIL import Image
                import io
                
                image = Image.open(io.BytesIO(chunk.image_data))
                
                # Process image with CLIP
                inputs = self.vision_processor(images=image, return_tensors="pt")
                
                # Get image features
                with torch.no_grad():
                    image_features = self.vision_model_obj.get_image_features(**inputs)
                
                # Convert to list
                embedding = image_features[0].numpy().tolist()
                
                embedded = EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding,
                    modality="visual",
                )
                embedded_chunks.append(embedded)
            
            except Exception as e:
                logger.error(f"Failed to embed visual chunk: {e}")
                # Continue with next chunk
        
        logger.info(f"Embedded {len(embedded_chunks)} visual chunks")
        return embedded_chunks

    async def embed_multimodal_chunks(
        self, chunks: list[MultimodalChunk]
    ) -> list[EmbeddedChunk]:
        """Generate unified multimodal embeddings.
        
        Args:
            chunks: List of multimodal chunks
            
        Returns:
            List of embedded chunks
        """
        embedded_chunks = []
        
        for chunk in chunks:
            try:
                # Get text embedding
                text_embedding = None
                if self.text_enabled:
                    response = await self.text_client.embeddings.create(
                        model=self.text_model, input=[chunk.text_content]
                    )
                    text_embedding = response.data[0].embedding
                
                # Get visual embeddings for associated visuals
                visual_embeddings = []
                if self.vision_enabled and chunk.visual_elements:
                    for visual in chunk.visual_elements:
                        try:
                            from PIL import Image
                            import io
                            
                            image = Image.open(io.BytesIO(visual.image_data))
                            inputs = self.vision_processor(images=image, return_tensors="pt")
                            
                            with torch.no_grad():
                                features = self.vision_model_obj.get_image_features(**inputs)
                            
                            visual_embeddings.append(features[0].numpy())
                        except Exception as e:
                            logger.warning(f"Failed to embed visual element: {e}")
                
                # Fuse embeddings
                if text_embedding and visual_embeddings:
                    fused_embedding = self.fuse_embeddings(
                        text_embedding, visual_embeddings
                    )
                elif text_embedding:
                    fused_embedding = text_embedding
                elif visual_embeddings:
                    # Average visual embeddings
                    fused_embedding = np.mean(visual_embeddings, axis=0).tolist()
                else:
                    logger.warning("No embeddings available for multimodal chunk")
                    continue
                
                embedded = EmbeddedChunk(
                    chunk=chunk,
                    embedding=fused_embedding,
                    modality="multimodal",
                )
                embedded_chunks.append(embedded)
            
            except Exception as e:
                logger.error(f"Failed to embed multimodal chunk: {e}")
        
        logger.info(f"Embedded {len(embedded_chunks)} multimodal chunks")
        return embedded_chunks

    async def embed_query(self, query: str, modality: str = "text") -> list[float]:
        """Generate embedding for search query.
        
        Args:
            query: Query text
            modality: Query modality (text or multimodal)
            
        Returns:
            Query embedding vector
        """
        if modality == "text" and self.text_enabled:
            response = await self.text_client.embeddings.create(
                model=self.text_model, input=[query]
            )
            return response.data[0].embedding
        else:
            logger.warning(f"Query embedding for modality '{modality}' not supported")
            return []

    def fuse_embeddings(
        self, text_emb: list[float], visual_embs: list[np.ndarray]
    ) -> list[float]:
        """Fuse text and visual embeddings into unified representation.
        
        Args:
            text_emb: Text embedding vector
            visual_embs: List of visual embedding arrays
            
        Returns:
            Fused embedding vector
        """
        # Simple fusion: concatenate text embedding with averaged visual embeddings
        text_array = np.array(text_emb)
        
        if visual_embs:
            # Average visual embeddings
            visual_avg = np.mean(visual_embs, axis=0)
            
            # Concatenate
            fused = np.concatenate([text_array, visual_avg])
        else:
            fused = text_array
        
        return fused.tolist()
