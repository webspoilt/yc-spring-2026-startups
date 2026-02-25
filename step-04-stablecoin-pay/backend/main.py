from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import stablecoin, monitoring

app = FastAPI(
    title="Stablecoin Protocol API",
    description="DeFi stablecoin with yield optimization and KYC",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stablecoin.router, prefix="/api/v1", tags=["stablecoin"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])


@app.get("/")
def root():
    return {
        "name": "Stablecoin Protocol API",
        "version": "1.0.0",
        "features": ["Polygon monitoring", "Yield optimization", "KYC middleware"]
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
