"""
AWS Textract OCR Integration
Extracts text and data from documents.
"""

import boto3
from typing import Dict, List, Any, Optional
import os


class TextractOCR:
    """
    AWS Textract wrapper for OCR operations.
    """
    
    def __init__(self):
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        
        if self.aws_access_key and self.aws_secret:
            self.client = boto3.client(
                'textract',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret,
                region_name=self.region
            )
        else:
            self.client = None
    
    async def analyze_document(
        self,
        document_bytes: bytes
    ) -> Dict[str, Any]:
        """Analyze document and extract text."""
        if not self.client:
            return self._mock_analyze()
        
        response = self.client.analyze_document(
            Document={'Bytes': document_bytes},
            FeatureTypes=['TABLES', 'FORMS']
        )
        
        return self._parse_response(response)
    
    async def detect_document_text(
        self,
        document_bytes: bytes
    ) -> List[Dict[str, str]]:
        """Detect text in document (simpler API)."""
        if not self.client:
            return self._mock_detect()
        
        response = self.client.detect_document_text(
            Document={'Bytes': document_bytes}
        )
        
        blocks = []
        for block in response.get('Blocks', []):
            if block['BlockType'] == 'LINE':
                blocks.append({
                    'text': block.get('Text', ''),
                    'confidence': block.get('Confidence', 0)
                })
        
        return blocks
    
    async def get_document_analysis(
        self,
        job_id: str
    ) -> Dict[str, Any]:
        """Get async analysis job result."""
        if not self.client:
            return {"status": "COMPLETE", "blocks": []}
        
        response = self.client.get_document_analysis(JobId=job_id)
        
        return {
            "status": response.get('JobStatus'),
            "blocks": response.get('Blocks', [])
        }
    
    def _mock_analyze(self) -> Dict[str, Any]:
        """Mock analysis for testing."""
        return {
            "text": "This is extracted text from the document.",
            "tables": [],
            "forms": {"key": "value"},
            "blocks": [
                {"text": "This is extracted text", "type": "LINE", "confidence": 0.95}
            ]
        }
    
    def _mock_detect(self) -> List[Dict[str, str]]:
        """Mock detection for testing."""
        return [
            {"text": "OFFICIAL DOCUMENT", "confidence": 0.99},
            {"text": "Government Form 1234", "confidence": 0.97}
        ]
    
    def _parse_response(self, response: Dict) -> Dict[str, Any]:
        """Parse Textract response."""
        blocks = response.get('Blocks', [])
        
        text_blocks = [b['Text'] for b in blocks if b['BlockType'] == 'LINE']
        
        tables = []
        forms = {}
        
        for block in blocks:
            if block['BlockType'] == 'TABLE':
                tables.append(block)
            elif block['BlockType'] == 'KEY_VALUE_SET':
                if 'KEY' in block.get('EntityTypes', []):
                    key = block.get('Text', 'unknown')
                    value = block.get('Relationships', [{}])[0].get('Ids', [])
                    forms[key] = value
        
        return {
            "text": "\n".join(text_blocks),
            "tables": tables,
            "forms": forms,
            "blocks": blocks
        }
