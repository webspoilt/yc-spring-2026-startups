from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import planning, sensors, quality

app = FastAPI(
    title="AI Manufacturing Execution System API",
    description="Genetic algorithm production scheduling, IoT sensor simulation, and quality inspection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(planning.router, prefix="/api/v1/planning", tags=["production-planning"])
app.include_router(sensors.router, prefix="/api/v1/sensors", tags=["iot-sensors"])
app.include_router(quality.router, prefix="/api/v1/quality", tags=["quality-inspection"])


@app.get("/")
def root():
    return {
        "name": "AI MES API",
        "version": "1.0.0",
        "features": [
            "Genetic Algorithm Production Scheduling",
            "MQTT IoT Sensor Simulation",
            "Product Quality Inspection & Alerts",
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
