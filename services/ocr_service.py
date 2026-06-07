"""
OCR service for extracting text from scanned documents and images.
Uses Tesseract OCR with Arabic and English support.
"""

import asyncio
import base64
import mimetypes
import os
from typing import List, Dict, Optional, Any
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class OCRService:
    """
    Handles Optical Character Recognition for scanned documents.
    Optimized for Arabic and multilingual text extraction.
    """
    
    def __init__(self):
        self._openai_client = None
        # Set Tesseract path if specified
        if settings.enable_ocr_processing and self._provider() == "local" and settings.tesseract_path:
            pytesseract = self._get_pytesseract()
            # Use absolute path and normalize slashes for Windows
            tess_path = os.path.abspath(settings.tesseract_path)
            pytesseract.pytesseract.tesseract_cmd = tess_path
            
            # Set TESSDATA_PREFIX to ensure Arabic data is found correctly
            # On some Windows builds, this should be the dir containing 'tessdata'
            tess_dir = os.path.dirname(tess_path)
            tessdata_dir = os.path.join(tess_dir, "tessdata")
            
            if os.path.exists(tessdata_dir):
                # Tesseract 5.x on Windows often prefers the directory WITH 'tessdata'
                os.environ["TESSDATA_PREFIX"] = tessdata_dir
                logger.info(f"Set TESSDATA_PREFIX to {tessdata_dir}")
            
            # Prevent Access Violation crashes on Windows by limiting threads
            # This can help with the ObjectCache LEAK and 0xC0000005 errors
            os.environ["OMP_THREAD_LIMIT"] = "1"
            os.environ["TESSERACT_THREAD_LIMIT"] = "1"
        
        self.ocr_languages = settings.ocr_languages
        self.temp_path = settings.temp_path

    def _provider(self) -> str:
        return (settings.ocr_provider or "local").lower()

    def _ensure_enabled(self):
        if not settings.enable_ocr_processing:
            raise RuntimeError("OCR processing is disabled. Set ENABLE_OCR_PROCESSING=true to enable it.")

    def _get_pytesseract(self):
        self._ensure_enabled()
        try:
            import pytesseract
            return pytesseract
        except ImportError as exc:
            raise RuntimeError("pytesseract is not installed. Use full requirements or enable an OCR worker.") from exc

    def _get_convert_from_path(self):
        self._ensure_enabled()
        try:
            from pdf2image import convert_from_path
            return convert_from_path
        except ImportError as exc:
            raise RuntimeError("pdf2image is not installed. Use full requirements or enable an OCR worker.") from exc

    def _open_image(self, image_path: str):
        self._ensure_enabled()
        try:
            from PIL import Image
            return Image.open(image_path)
        except ImportError as exc:
            raise RuntimeError("Pillow is not installed. Use full requirements or enable an OCR worker.") from exc
    
    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using OCR.
        Converts PDF pages to images then applies OCR.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        if self._provider() == "openai":
            return await asyncio.to_thread(self._extract_text_with_openai, pdf_path, "pdf")

        try:
            logger.info(f"Starting OCR on PDF: {pdf_path}")
            convert_from_path = self._get_convert_from_path()
            
            # Convert PDF to images with slightly lower DPI for stability
            # 200-300 is ideal for OCR. 200 saves memory on many-page documents.
            poppler_path = None
            if os.path.exists("./poppler-bin/Library/bin"):
                poppler_path = os.path.abspath("./poppler-bin/Library/bin")
                
            images = convert_from_path(
                pdf_path,
                dpi=200, 
                fmt='jpeg',
                poppler_path=poppler_path
            )
            
            num_pages = len(images)
            logger.info(f"Converted {num_pages} pages to images")
            
            # Extract text from each page with terminal progress bar
            from tqdm import tqdm
            all_text = []
            
            pbar = tqdm(total=num_pages, desc=f"OCR {os.path.basename(pdf_path)[:15]}", unit="pg")
            
            for i, image in enumerate(images):
                logger.debug(f"Processing page {i + 1}/{num_pages}")
                text = self._extract_text_from_image(image)
                if text.strip():
                    all_text.append(text)
                
                # Free memory explicitly if needed
                images[i] = None 
                pbar.update(1)
            
            pbar.close()
            combined_text = "\n\n".join(all_text)
            logger.info(f"OCR completed. Extracted {len(combined_text)} characters")
            
            return combined_text
            
        except Exception as e:
            logger.error(f"OCR failed for PDF {pdf_path}: {e}")
            raise

    async def extract_pages_with_ocr(self, pdf_path: str, page_numbers: List[int]) -> Dict[int, str]:
        """
        Extract text from specific PDF pages using OCR.
        
        Args:
            pdf_path: Path to PDF
            page_numbers: List of 1-indexed page numbers
            
        Returns:
            Dictionary mapping page number to extracted text
        """
        if not page_numbers:
            return {}

        if self._provider() == "openai":
            text = await self.extract_text_from_pdf(pdf_path)
            return {page_numbers[0]: text}
            
        try:
            results = {}
            from tqdm import tqdm
            
            # Process in small batches to save memory
            batch_size = 5
            for i in range(0, len(page_numbers), batch_size):
                batch = page_numbers[i:i + batch_size]
                logger.info(f"OCR on PDF batch: pages {batch}")
                
                # Convert only the required pages
                for page_num in batch:
                    try:
                        poppler_path = None
                        if os.path.exists("./poppler-bin/Library/bin"):
                            poppler_path = os.path.abspath("./poppler-bin/Library/bin")
                        convert_from_path = self._get_convert_from_path()

                        # Use to_thread to keep the event loop responsive
                        images = await asyncio.to_thread(
                            convert_from_path,
                            pdf_path,
                            dpi=200,
                            fmt='jpeg',
                            first_page=page_num,
                            last_page=page_num,
                            poppler_path=poppler_path
                        )
                        if images:
                            text = await asyncio.to_thread(
                                self._extract_text_from_image, 
                                images[0]
                            )
                            if not text.strip():
                                logger.warning(f"OCR returned empty text for page {page_num}")
                            results[page_num] = text
                    except Exception as pg_err:
                        logger.warning(f"Failed OCR on page {page_num}: {pg_err}")
                        results[page_num] = ""
                        
            return results
        except Exception as e:
            logger.error(f"Batch OCR failed for {pdf_path}: {e}")
            return {}
    
    async def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        if self._provider() == "openai":
            return await asyncio.to_thread(self._extract_text_with_openai, image_path, "image")

        try:
            logger.info(f"Starting OCR on image: {image_path}")
            image = self._open_image(image_path)
            text = self._extract_text_from_image(image)
            logger.info(f"OCR completed. Extracted {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"OCR failed for image {image_path}: {e}")
            raise
    
    def _extract_text_from_image(self, image: Any) -> str:
        """
        Apply Tesseract OCR to PIL Image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text
        """
        try:
            # Preprocess image for better OCR
            image = self._preprocess_image(image)
            
            # Apply OCR with specified languages
            config = r'--oem 3 --psm 6'  # LSTM engine, assume uniform text block
            pytesseract = self._get_pytesseract()
            text = pytesseract.image_to_string(
                image,
                lang=self.ocr_languages,
                config=config
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            # Return empty string on error rather than failing
            return ""
    
    def _preprocess_image(self, image: Any) -> Any:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        image = image.convert('L')
        
        # Increase contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        return image
    
    def is_text_extractable(self, text: str) -> bool:
        """
        Check if extracted text is meaningful or if OCR is needed.
        
        Args:
            text: Extracted text
            
        Returns:
            True if text is extractable, False if OCR needed
        """
        if not text or len(text.strip()) < 50:
            return False
        
        # Check for mostly gibberish or encoding issues
        printable_ratio = sum(c.isprintable() for c in text) / len(text)
        
        return printable_ratio > 0.7
    
    async def extract_with_fallback(
        self,
        pdf_path: str,
        direct_text: Optional[str] = None
    ) -> str:
        """
        Extract text with OCR fallback if direct extraction fails.
        
        Args:
            pdf_path: Path to PDF
            direct_text: Text from direct extraction (if available)
            
        Returns:
            Best available text
        """
        # If direct text is good, use it
        if direct_text and self.is_text_extractable(direct_text):
            logger.info("Using direct text extraction (no OCR needed)")
            return direct_text
        
        # Otherwise, apply OCR
        logger.info("Direct extraction insufficient, applying OCR")
        return await self.extract_text_from_pdf(pdf_path)

    def _get_openai_client(self):
        if not settings.openai_api_key:
            raise RuntimeError("OCR_PROVIDER=openai requires OPENAI_API_KEY")
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    def _extract_text_with_openai(self, file_path: str, file_kind: str) -> str:
        self._ensure_enabled()
        client = self._get_openai_client()
        mime_type = mimetypes.guess_type(file_path)[0]
        if not mime_type:
            mime_type = "application/pdf" if file_kind == "pdf" else "image/png"

        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")

        prompt = (
            "Extract all readable text from this file for a RAG index. "
            "Preserve Arabic and English text exactly where possible. "
            "Return only the extracted text, with page breaks when visible."
        )

        if file_kind == "pdf":
            file_content = {
                "type": "input_file",
                "filename": os.path.basename(file_path),
                "file_data": f"data:{mime_type};base64,{encoded}",
            }
        else:
            file_content = {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
            }

        response = client.responses.create(
            model=settings.openai_ocr_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        file_content,
                    ],
                }
            ],
        )

        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        if hasattr(response, "model_dump"):
            data = response.model_dump()
            output_parts = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    value = content.get("text")
                    if value:
                        output_parts.append(value)
            return "\n".join(output_parts).strip()

        return str(response).strip()


# Global instance
ocr_service = OCRService()
