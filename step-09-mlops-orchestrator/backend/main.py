from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import gpu, training, monitoring

app = FastAPI(
    title="MLOps Platform API",
    description="GPU fleet management, DeepSpeed distributed training, and training health monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gpu.router, prefix="/api/v1/gpu", tags=["gpu-management"])
app.include_router(training.router, prefix="/api/v1/training", tags=["distributed-training"])
app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["loss-monitoring"])


@app.get("/")
def root():
    return {
        "name": "MLOps Platform API",
        "version": "1.0.0",
        "features": [
            "GPU Fleet Management",
            "DeepSpeed ZeRO-3 Launcher",
            "Training Loss Monitor",
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
