from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .routers import documents, ocr

app = FastAPI(
    title="Gov AI API",
    description="Private LLM pipeline for government documents with OCR and PII redaction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(ocr.router, prefix="/api/v1", tags=["ocr"])


@app.get("/")
def root():
    return {"name": "Gov AI API", "version": "1.0.0", "features": ["OCR", "PII Redaction", "Private LLM"]}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
