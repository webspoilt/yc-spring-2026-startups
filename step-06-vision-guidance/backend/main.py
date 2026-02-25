from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .routers import vision, guidance

app = FastAPI(
    title="Vision Guidance API",
    description="Computer vision for exercise/gait analysis with audio guidance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vision.router, prefix="/api/v1", tags=["vision"])
app.include_router(guidance.router, prefix="/api/v1", tags=["guidance"])


@app.get("/")
def root():
    return {"name": "Vision Guidance API", "version": "1.0.0", "features": ["Tool Detection", "Pose Estimation", "Audio Guidance"]}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
