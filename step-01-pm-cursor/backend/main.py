from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import projects, ai
from .database import init_db

app = FastAPI(
    title="PM Cursor API",
    description="Product Management Tool with AI-powered specification generation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(projects.router, prefix="/api/v1", tags=["projects"])
app.include_router(ai.router, prefix="/api/v1", tags=["ai"])


@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    init_db()


@app.get("/")
def root():
    return {
        "name": "PM Cursor API",
        "version": "1.0.0",
        "description": "Product Management Tool with AI-powered features",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
