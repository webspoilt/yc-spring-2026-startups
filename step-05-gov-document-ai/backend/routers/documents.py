from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

router = APIRouter()

documents_db = {}


class Document(BaseModel):
    id: str
    name: str
    content: str
    summary: Optional[str] = None
    extracted_data: dict = {}
    pii_redacted: bool = False
    created_at: datetime


class QueryRequest(BaseModel):
    document_id: str
    question: str


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing."""
    content = await file.read()
    text_content = content.decode("utf-8", errors="ignore")
    
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        name=file.filename,
        content=text_content[:5000],  # Limit for demo
        created_at=datetime.now()
    )
    documents_db[doc_id] = doc
    
    return {"document_id": doc_id, "filename": file.filename, "status": "uploaded"}


@router.get("/documents")
async def list_documents():
    """List all documents."""
    return list(documents_db.values())


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document details."""
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    return documents_db[doc_id]


@router.post("/documents/{doc_id}/summarize")
async def summarize_document(doc_id: str):
    """Summarize document using LLM."""
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_db[doc_id]
    
    # Mock summarization
    doc.summary = f"Summary of {doc.name}: This document discusses important government policy matters..."
    
    return {"document_id": doc_id, "summary": doc.summary}


@router.post("/query")
async def query_document(request: QueryRequest):
    """Query document using private LLM."""
    if request.document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_db[request.document_id]
    
    # Mock LLM response
    response = f"Based on the document '{doc.name}', regarding your question about '{request.question}': The document outlines key procedures and guidelines that address this matter..."
    
    return {"question": request.question, "answer": response, "sources": [doc.name]}


@router.post("/redact")
async def redact_pii(text: str):
    """Redact PII from text."""
    from ...ai_engine.redaction.pii_redactor import PIIRedactor
    
    redactor = PIIRedactor()
    redacted = redactor.redact(text)
    
    return {"original_length": len(text), "redacted_length": len(redacted), "redacted_text": redacted}
