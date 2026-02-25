"""
PII Redaction Module
Redacts personally identifiable information from text.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RedactionResult:
    """Result of PII redaction."""
    original_text: str
    redacted_text: str
    redactions: List[Dict[str, any]]
    confidence: float


class PIIRedactor:
    """
    PII detection and redaction.
    Supports SSN, phone, email, addresses, etc.
    """
    
    # Regex patterns for PII detection
    PATTERNS = {
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "phone": r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "address": r'\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)\b',
        "zip_code": r'\b\d{5}(-\d{4})?\b',
        "date_of_birth": r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b',
        "passport": r'\b[A-Z]{1,2}\d{6,9}\b',
        "drivers_license": r'\b[A-Z]{1,2}\d{5,8}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "bank_account": r'\b\d{8,17}\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    }
    
    def __init__(self):
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }
    
    def redact(
        self,
        text: str,
        custom_patterns: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Redact all PII from text.
        
        Args:
            text: Input text
            custom_patterns: Optional custom patterns to add
        
        Returns:
            Redacted text
        """
        redacted = text
        
        # Apply built-in patterns
        for name, pattern in self.compiled_patterns.items():
            redacted = pattern.sub(self._get_replacement(name), redacted)
        
        # Apply custom patterns
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                compiled = re.compile(pattern, re.IGNORECASE)
                redacted = compiled.sub(self._get_replacement(name), redacted)
        
        return redacted
    
    def detect(
        self,
        text: str
    ) -> List[Dict[str, any]]:
        """
        Detect PII in text without redacting.
        
        Returns:
            List of detected PII with positions
        """
        detections = []
        
        for name, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                detections.append({
                    "type": name,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "context": text[max(0, match.start()-20):min(len(text), match.end()+20)]
                })
        
        return detections
    
    def redact_with_analysis(
        self,
        text: str
    ) -> RedactionResult:
        """
        Redact and analyze the text.
        
        Returns:
            RedactionResult with details
        """
        detections = self.detect(text)
        redacted = self.redact(text)
        
        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            redactions=detections,
            confidence=0.95 if detections else 1.0
        )
    
    def _get_replacement(self, pii_type: str) -> str:
        """Get replacement string for PII type."""
        replacements = {
            "ssn": "[SSN_REDACTED]",
            "phone": "[PHONE_REDACTED]",
            "email": "[EMAIL_REDACTED]",
            "address": "[ADDRESS_REDACTED]",
            "zip_code": "[ZIP_REDACTED]",
            "date_of_birth": "[DOB_REDACTED]",
            "passport": "[PASSPORT_REDACTED]",
            "drivers_license": "[DL_REDACTED]",
            "credit_card": "[CC_REDACTED]",
            "bank_account": "[ACCOUNT_REDACTED]",
            "ip_address": "[IP_REDACTED]"
        }
        return replacements.get(pii_type, "[REDACTED]")
    
    def batch_redact(
        self,
        texts: List[str]
    ) -> List[str]:
        """Redact multiple texts."""
        return [self.redact(text) for text in texts]
