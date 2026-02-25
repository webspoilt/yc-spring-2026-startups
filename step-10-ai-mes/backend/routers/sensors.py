"""IoT sensor simulation endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field

from ai_engine.mqtt_simulator import (
    MQTTIoTSimulator, SensorReading, SensorConfig, SensorType,
)

router = APIRouter()
simulator = MQTTIoTSimulator()
simulator.add_default_sensors()


class SimulatorControlRequest(BaseModel):
    interval_seconds: float = Field(default=1.0, ge=0.1, le=60.0)


class AddSensorRequest(BaseModel):
    sensor_id: str
    sensor_type: str
    machine_id: str
    unit: str
    base_value: float
    noise_std: float = 0.5
    anomaly_probability: float = 0.02


@router.post("/start")
async def start_simulator(request: SimulatorControlRequest):
    """Start the IoT sensor simulator in the background."""
    simulator.start(interval_seconds=request.interval_seconds)
    return {
        "status": "started",
        "interval_seconds": request.interval_seconds,
        "active_sensors": len(simulator._sensors),
    }


@router.post("/stop")
async def stop_simulator():
    """Stop the IoT sensor simulator."""
    simulator.stop()
    return {"status": "stopped"}


@router.post("/generate")
async def generate_batch():
    """Generate one batch of sensor readings (one per sensor)."""
    readings = simulator.generate_batch()
    return {
        "readings_count": len(readings),
        "readings": [r.model_dump() for r in readings],
        "anomalies": [r.model_dump() for r in readings if r.is_anomaly],
        "warnings": [r.model_dump() for r in readings if r.quality_flag == "warning"],
        "critical": [r.model_dump() for r in readings if r.quality_flag == "critical"],
    }


@router.get("/recent")
async def get_recent_readings(limit: int = 100):
    """Get recent sensor readings from the buffer."""
    readings = simulator.get_recent_readings(limit)
    return {
        "count": len(readings),
        "readings": [r.model_dump() for r in readings],
    }


@router.get("/stats")
async def get_sensor_stats():
    """Get per-sensor statistics from buffered readings."""
    stats = simulator.get_sensor_stats()
    return {
        "sensors": len(stats),
        "stats": stats,
    }


@router.post("/sensors/add")
async def add_sensor(request: AddSensorRequest):
    """Add a new sensor to the simulation."""
    config = SensorConfig(
        sensor_id=request.sensor_id,
        sensor_type=SensorType(request.sensor_type),
        machine_id=request.machine_id,
        unit=request.unit,
        base_value=request.base_value,
        noise_std=request.noise_std,
        anomaly_probability=request.anomaly_probability,
    )
    simulator.add_sensor(config)
    return {"status": "added", "sensor_id": request.sensor_id}


@router.get("/sensors")
async def list_sensors():
    """List all configured sensors."""
    return {
        "count": len(simulator._sensors),
        "sensors": [
            {
                "sensor_id": s.sensor_id,
                "type": s.sensor_type.value,
                "machine_id": s.machine_id,
                "unit": s.unit,
                "base_value": s.base_value,
            }
            for s in simulator._sensors
        ],
    }
