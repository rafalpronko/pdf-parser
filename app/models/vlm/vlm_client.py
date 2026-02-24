"""Vision-Language Model client for visual content understanding."""

import base64
import io
import logging
from typing import Any

from PIL import Image

from app.config import get_settings
from app.models.parsing import ChartBlock, ImageBlock, TableBlock

logger = logging.getLogger(__name__)


class VLMClient:
    """Client for Vision-Language Model operations.
    
    Supports multiple VLM backends:
    - OpenAI GPT-4V (gpt-4-vision-preview)
    - Local models (LLaVA, Qwen-VL) - future support
    """

    def __init__(self, provider: str | None = None, model: str | None = None):
        """Initialize VLM client.
        
        Args:
            provider: VLM provider (openai, local). If None, uses config.
            model: VLM model name. If None, uses config.
        """
        settings = get_settings()
        self.provider = provider or settings.vlm_provider
        self.model = model or settings.vlm_model
        self.enabled = settings.enable_vlm
        
        logger.info(f"Initialized VLM client: provider={self.provider}, model={self.model}")
        
        # Initialize provider-specific client
        if self.provider == "openai":
            self._init_openai()
        else:
            logger.warning(f"VLM provider '{self.provider}' not yet implemented")
            self.enabled = False

    def _init_openai(self):
        """Initialize OpenAI VLM client."""
        try:
            from openai import AsyncOpenAI
            
            settings = get_settings()
            if not settings.openai_api_key:
                logger.warning("OpenAI API key not set, VLM will be disabled")
                self.enabled = False
                return
            
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI VLM client initialized")
        except ImportError:
            logger.error("OpenAI library not installed")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI VLM: {e}")
            self.enabled = False

    async def describe_image(self, image: ImageBlock) -> str:
        """Generate textual description of image content.
        
        Args:
            image: ImageBlock containing image data
            
        Returns:
            Textual description of the image
            
        Raises:
            RuntimeError: If VLM is not enabled or fails
        """
        if not self.enabled:
            logger.warning("VLM not enabled, returning placeholder description")
            return f"Image on page {image.page}"
        
        try:
            if self.provider == "openai":
                return await self._describe_image_openai(image)
            else:
                raise NotImplementedError(f"Provider {self.provider} not implemented")
        except Exception as e:
            logger.error(f"Failed to describe image: {e}")
            # Fallback to basic description
            return f"Image on page {image.page} (description failed: {str(e)})"

    async def _describe_image_openai(self, image: ImageBlock) -> str:
        """Describe image using OpenAI GPT-4V.
        
        Args:
            image: ImageBlock containing image data
            
        Returns:
            Description from GPT-4V
        """
        # Convert image bytes to base64
        image_base64 = base64.b64encode(image.image_data).decode("utf-8")
        
        # Determine image format
        image_format = image.format.lower()
        if image_format not in ["png", "jpeg", "jpg", "gif", "webp"]:
            image_format = "png"  # Default
        
        # Create prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail. Focus on the main content, objects, text, and any important visual elements.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_base64}"
                        },
                    },
                ],
            }
        ]
        
        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=300, temperature=0.3
        )
        
        description = response.choices[0].message.content
        logger.info(f"Generated image description: {description[:100]}...")
        
        return description

    async def analyze_chart(self, chart: ChartBlock) -> dict[str, Any]:
        """Extract data and insights from charts/diagrams.
        
        Args:
            chart: ChartBlock containing chart image
            
        Returns:
            Dictionary with chart analysis including type, data points, insights
        """
        if not self.enabled:
            logger.warning("VLM not enabled, returning placeholder analysis")
            return {
                "chart_type": chart.chart_type or "unknown",
                "description": f"Chart on page {chart.page}",
                "data_points": [],
            }
        
        try:
            if self.provider == "openai":
                return await self._analyze_chart_openai(chart)
            else:
                raise NotImplementedError(f"Provider {self.provider} not implemented")
        except Exception as e:
            logger.error(f"Failed to analyze chart: {e}")
            return {
                "chart_type": chart.chart_type or "unknown",
                "description": f"Chart analysis failed: {str(e)}",
                "data_points": [],
            }

    async def _analyze_chart_openai(self, chart: ChartBlock) -> dict[str, Any]:
        """Analyze chart using OpenAI GPT-4V.
        
        Args:
            chart: ChartBlock containing chart image
            
        Returns:
            Chart analysis dictionary
        """
        # Convert image bytes to base64
        image_base64 = base64.b64encode(chart.image_data).decode("utf-8")
        
        # Create prompt for chart analysis
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyze this chart/diagram. Provide:
1. Chart type (bar, line, pie, scatter, etc.)
2. Main insights and trends
3. Key data points if visible
4. Axis labels and units if present

Format your response as a structured analysis.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]
        
        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=500, temperature=0.3
        )
        
        analysis_text = response.choices[0].message.content
        
        # Parse response into structured format
        return {
            "chart_type": chart.chart_type or "detected from image",
            "description": analysis_text,
            "data_points": [],  # Could parse from response
            "raw_analysis": analysis_text,
        }

    async def extract_table_from_image(self, image: ImageBlock) -> TableBlock:
        """Convert visual table to structured data.
        
        Args:
            image: ImageBlock containing table image
            
        Returns:
            TableBlock with extracted table data
        """
        if not self.enabled:
            logger.warning("VLM not enabled, returning empty table")
            return TableBlock(rows=[], page=image.page, bbox=image.bbox)
        
        try:
            if self.provider == "openai":
                return await self._extract_table_openai(image)
            else:
                raise NotImplementedError(f"Provider {self.provider} not implemented")
        except Exception as e:
            logger.error(f"Failed to extract table: {e}")
            return TableBlock(rows=[], page=image.page, bbox=image.bbox)

    async def _extract_table_openai(self, image: ImageBlock) -> TableBlock:
        """Extract table using OpenAI GPT-4V.
        
        Args:
            image: ImageBlock containing table image
            
        Returns:
            TableBlock with extracted data
        """
        # Convert image bytes to base64
        image_base64 = base64.b64encode(image.image_data).decode("utf-8")
        
        # Create prompt for table extraction
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract the table data from this image. 
Format the output as rows separated by newlines, with cells separated by | (pipe).
Include headers if present.
Example format:
Header1 | Header2 | Header3
Value1 | Value2 | Value3""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]
        
        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=1000, temperature=0.1
        )
        
        table_text = response.choices[0].message.content
        
        # Parse table text into rows
        rows = []
        for line in table_text.strip().split("\n"):
            if "|" in line:
                cells = [cell.strip() for cell in line.split("|")]
                rows.append(cells)
        
        return TableBlock(rows=rows, page=image.page, bbox=image.bbox)

    async def answer_visual_question(self, image: ImageBlock, question: str) -> str:
        """Answer questions about specific visual content.
        
        Args:
            image: ImageBlock containing the image
            question: Question about the image
            
        Returns:
            Answer to the question
        """
        if not self.enabled:
            logger.warning("VLM not enabled, returning placeholder answer")
            return "VLM is not enabled. Cannot answer visual questions."
        
        try:
            if self.provider == "openai":
                return await self._answer_visual_question_openai(image, question)
            else:
                raise NotImplementedError(f"Provider {self.provider} not implemented")
        except Exception as e:
            logger.error(f"Failed to answer visual question: {e}")
            return f"Failed to answer question: {str(e)}"

    async def _answer_visual_question_openai(
        self, image: ImageBlock, question: str
    ) -> str:
        """Answer visual question using OpenAI GPT-4V.
        
        Args:
            image: ImageBlock containing the image
            question: Question about the image
            
        Returns:
            Answer from GPT-4V
        """
        # Convert image bytes to base64
        image_base64 = base64.b64encode(image.image_data).decode("utf-8")
        
        # Create prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]
        
        # Call OpenAI API
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=500, temperature=0.5
        )
        
        answer = response.choices[0].message.content
        return answer
