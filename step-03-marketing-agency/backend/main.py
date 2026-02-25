from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import campaigns

app = FastAPI(
    title="AI Agency API",
    description="Multi-agent AI content generation pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(campaigns.router, prefix="/api/v1", tags=["campaigns"])


@app.get("/")
def root():
    return {
        "name": "AI Agency API",
        "version": "1.0.0",
        "description": "Content generation pipeline with AI agents",
        "agents": ["researcher", "copywriter", "designer", "reviewer"]
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
