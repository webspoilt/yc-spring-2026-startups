from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import transform

app = FastAPI(title="Spatial Transformer API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(transform.router, prefix="/api/v1")

@app.get("/")
def root(): return {"name": "Spatial Transformer API", "version": "1.0.0"}
@app.get("/health")
def health_check(): return {"status": "healthy"}
