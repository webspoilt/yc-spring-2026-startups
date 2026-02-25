from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import trading, analysis

app = FastAPI(
    title="Quant Fund API",
    description="AI-powered quantitative trading system with multi-agent analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trading.router, prefix="/api/v1", tags=["trading"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])


@app.get("/")
def root():
    return {
        "name": "Quant Fund API",
        "version": "1.0.0",
        "description": "Multi-agent quantitative trading system",
        "agents": ["fundamental", "technical", "sentiment", "risk_manager"]
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
