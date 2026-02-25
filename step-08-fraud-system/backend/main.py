from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import fraud, analytics, complaints

app = FastAPI(
    title="Fraud Detection System API",
    description="Neo4j-powered transaction fraud detection with Benford's Law analysis and FCA complaint generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fraud.router, prefix="/api/v1/fraud", tags=["fraud-detection"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(complaints.router, prefix="/api/v1/complaints", tags=["complaints"])


@app.get("/")
def root():
    return {
        "name": "Fraud Detection System API",
        "version": "1.0.0",
        "features": [
            "Neo4j Graph Analysis",
            "Benford's Law Anomaly Detection",
            "FCA Regulatory Complaint Generation",
            "Circular Transfer Detection",
            "Structuring Detection",
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
