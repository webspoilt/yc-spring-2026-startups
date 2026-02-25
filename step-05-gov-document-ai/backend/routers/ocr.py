from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any

router = APIRouter()


@router.post("/ocr/extract")
async def extract_from_image(file: UploadFile = File(...)):
    """Extract text from image using AWS Textract (simulated)."""
    content = await file.read()
    
    # Simulate OCR extraction
    extracted_text = """
    OFFICIAL DOCUMENT
    Date: January 15, 2024
    
    This is a simulated government document for testing purposes.
    The quick brown fox jumps over the lazy dog.
    
    Signature: ___________________
    """
    
    return {
        "filename": file.filename,
        "extracted_text": extracted_text.strip(),
        "confidence": 0.95,
        "blocks": [
            {"text": "OFFICIAL DOCUMENT", "type": "LINE", "confidence": 0.99},
            {"text": "Date: January 15, 2024", "type": "LINE", "confidence": 0.97}
        ]
    }


@router.post("/ocr/extract-pdf")
async def extract_from_pdf(file: UploadFile = File(...)):
    """Extract text from PDF using AWS Textract."""
    content = await file.read()
    
    # Simulate PDF extraction
    return {
        "filename": file.filename,
        "num_pages": 5,
        "extracted_text": "Simulated PDF text extraction...",
        "tables": [],
        "forms": {}
    }


@router.get("/ocr/languages")
async def get_supported_languages():
    """Get supported OCR languages."""
    return {
        "languages": ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        "default": "en"
    }
