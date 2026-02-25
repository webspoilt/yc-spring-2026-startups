"""GPU fleet management endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Optional

from ...ai_engine.gpu_manager import GPUManager, GPUFleetStatus, GPUDevice

router = APIRouter()
manager = GPUManager()


@router.get("/fleet", response_model=GPUFleetStatus)
async def get_fleet_status():
    """Get real-time status of all GPUs in the fleet."""
    try:
        return manager.get_fleet_status()
    except RuntimeError:
        # Fallback to mock data when nvidia-smi is unavailable
        return manager.get_mock_fleet()


@router.get("/fleet/mock", response_model=GPUFleetStatus)
async def get_mock_fleet():
    """Get mock GPU fleet data for testing and development."""
    return manager.get_mock_fleet()


@router.get("/select")
async def select_best_gpu(required_memory_mib: int = 0):
    """
    Select the best available GPU for a new training job.
    Returns the GPU with the most free memory that meets requirements.
    """
    try:
        gpu = manager.select_best_gpu(required_memory_mib)
    except RuntimeError:
        # Use mock fleet for selection
        fleet = manager.get_mock_fleet()
        available = [d for d in fleet.devices if d.is_available and d.memory_free_mib >= required_memory_mib]
        gpu = available[0] if available else None

    if gpu is None:
        raise HTTPException(
            status_code=503,
            detail=f"No GPU available with {required_memory_mib} MiB free memory"
        )
    return {
        "selected_gpu": gpu.model_dump(),
        "recommendation": f"Use GPU {gpu.index} ({gpu.name}) with {gpu.memory_free_mib} MiB free",
    }


@router.get("/devices/{index}")
async def get_device(index: int):
    """Get detailed info for a specific GPU by index."""
    try:
        fleet = manager.get_fleet_status()
    except RuntimeError:
        fleet = manager.get_mock_fleet()

    for device in fleet.devices:
        if device.index == index:
            return device.model_dump()

    raise HTTPException(status_code=404, detail=f"GPU {index} not found")


@router.get("/alerts")
async def get_gpu_alerts():
    """Get all active GPU health alerts."""
    try:
        fleet = manager.get_fleet_status()
    except RuntimeError:
        fleet = manager.get_mock_fleet()

    return {
        "total_alerts": len(fleet.alerts),
        "alerts": fleet.alerts,
        "devices_with_alerts": [
            {"gpu_index": d.index, "gpu_name": d.name, "health": d.health, "alerts": d.alerts}
            for d in fleet.devices if d.alerts
        ],
    }
