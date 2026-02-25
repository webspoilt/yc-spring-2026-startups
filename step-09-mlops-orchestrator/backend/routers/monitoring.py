"""Training loss monitoring endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from ...ai_engine.loss_monitor import (
    LossSpikeMonitor, TrainingMetrics, MonitorConfig, SpikeAlert,
)

router = APIRouter()

# Store monitors per job
_monitors: Dict[str, LossSpikeMonitor] = {}


def _get_or_create_monitor(job_id: str,
                           config: Optional[MonitorConfig] = None) -> LossSpikeMonitor:
    if job_id not in _monitors:
        _monitors[job_id] = LossSpikeMonitor(config)
    return _monitors[job_id]


class MonitorCreateRequest(BaseModel):
    job_id: str
    window_size: int = 100
    spike_sigma_threshold: float = 3.0
    divergence_threshold: float = 1e6
    plateau_patience: int = 500
    gradient_norm_threshold: float = 100.0


class MetricsIngestRequest(BaseModel):
    job_id: str
    metrics: List[TrainingMetrics]


class SingleMetricRequest(BaseModel):
    job_id: str
    step: int
    loss: float
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    throughput_samples_per_sec: float = 0.0
    gpu_memory_used_mib: int = 0


@router.post("/create")
async def create_monitor(request: MonitorCreateRequest):
    """Create a new loss monitor for a training job."""
    config = MonitorConfig(
        window_size=request.window_size,
        spike_sigma_threshold=request.spike_sigma_threshold,
        divergence_threshold=request.divergence_threshold,
        plateau_patience=request.plateau_patience,
        gradient_norm_threshold=request.gradient_norm_threshold,
    )
    monitor = _get_or_create_monitor(request.job_id, config)
    return {
        "job_id": request.job_id,
        "status": "created",
        "config": config.model_dump(),
    }


@router.post("/ingest")
async def ingest_metrics(request: MetricsIngestRequest):
    """Ingest a batch of training metrics and return any triggered alerts."""
    monitor = _get_or_create_monitor(request.job_id)
    all_alerts = []
    for metric in request.metrics:
        alerts = monitor.ingest(metric)
        all_alerts.extend(alerts)
    return {
        "job_id": request.job_id,
        "ingested": len(request.metrics),
        "alerts_triggered": len(all_alerts),
        "alerts": [a.model_dump() for a in all_alerts],
    }


@router.post("/ingest/single")
async def ingest_single_metric(request: SingleMetricRequest):
    """Ingest a single step's metrics."""
    monitor = _get_or_create_monitor(request.job_id)
    metric = TrainingMetrics(
        step=request.step,
        loss=request.loss,
        learning_rate=request.learning_rate,
        gradient_norm=request.gradient_norm,
        throughput_samples_per_sec=request.throughput_samples_per_sec,
        gpu_memory_used_mib=request.gpu_memory_used_mib,
    )
    alerts = monitor.ingest(metric)
    return {
        "job_id": request.job_id,
        "step": request.step,
        "alerts": [a.model_dump() for a in alerts],
        "healthy": len(alerts) == 0,
    }


@router.get("/summary/{job_id}")
async def get_monitor_summary(job_id: str):
    """Get the current monitoring summary for a job."""
    if job_id not in _monitors:
        raise HTTPException(status_code=404, detail=f"No monitor for job '{job_id}'")
    return _monitors[job_id].get_summary()


@router.get("/loss-curve/{job_id}")
async def get_loss_curve(job_id: str):
    """Get the full loss curve data for visualization."""
    if job_id not in _monitors:
        raise HTTPException(status_code=404, detail=f"No monitor for job '{job_id}'")
    return _monitors[job_id].get_loss_curve()


@router.delete("/reset/{job_id}")
async def reset_monitor(job_id: str):
    """Reset a training monitor (e.g., after checkpoint restore)."""
    if job_id not in _monitors:
        raise HTTPException(status_code=404, detail=f"No monitor for job '{job_id}'")
    _monitors[job_id].reset()
    return {"job_id": job_id, "status": "reset"}


@router.get("/jobs")
async def list_monitored_jobs():
    """List all active monitored training jobs."""
    return {
        "active_jobs": len(_monitors),
        "jobs": [
            {
                "job_id": jid,
                "total_steps": m.state.total_steps,
                "total_alerts": m.state.alert_count,
                "best_loss": m.state.best_loss,
            }
            for jid, m in _monitors.items()
        ],
    }
