"""Production planning / scheduling endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field

from ai_engine.planning_engine import (
    GeneticPlanningEngine, ProductionJob, MachineConfig,
    ScheduleResult, JobPriority,
)

router = APIRouter()
engine = GeneticPlanningEngine()


class ScheduleRequest(BaseModel):
    """Request to optimize a production schedule."""
    jobs: List[ProductionJob]
    machines: Optional[List[MachineConfig]] = None
    population_size: int = Field(default=100, ge=10)
    generations: int = Field(default=200, ge=10)
    crossover_rate: float = Field(default=0.85, ge=0, le=1)
    mutation_rate: float = Field(default=0.15, ge=0, le=1)


@router.post("/optimize", response_model=ScheduleResult)
async def optimize_schedule(request: ScheduleRequest):
    """
    Run genetic algorithm to optimize production schedule.
    Returns an optimized schedule with makespan, tardiness, and utilization metrics.
    """
    opt = GeneticPlanningEngine(
        population_size=request.population_size,
        generations=request.generations,
        crossover_rate=request.crossover_rate,
        mutation_rate=request.mutation_rate,
    )
    result = opt.optimize(request.jobs, request.machines)
    return result


@router.post("/optimize/quick", response_model=ScheduleResult)
async def quick_optimize(request: ScheduleRequest):
    """Quick optimization with reduced generations (50) for faster response."""
    opt = GeneticPlanningEngine(
        population_size=50,
        generations=50,
        crossover_rate=request.crossover_rate,
        mutation_rate=request.mutation_rate,
    )
    result = opt.optimize(request.jobs, request.machines)
    return result


@router.post("/demo")
async def demo_schedule():
    """Run a demo optimization with sample manufacturing jobs."""
    demo_jobs = [
        ProductionJob(
            job_id="JOB-001", product_name="Mounting Bracket",
            operations=["cnc", "welding", "painting", "inspection"],
            processing_times=[45, 30, 20, 10],
            priority=JobPriority.HIGH, quantity=5,
        ),
        ProductionJob(
            job_id="JOB-002", product_name="Control Panel",
            operations=["cnc", "assembly", "inspection"],
            processing_times=[60, 40, 15],
            priority=JobPriority.MEDIUM, quantity=3,
        ),
        ProductionJob(
            job_id="JOB-003", product_name="Sensor Housing",
            operations=["cnc", "cnc", "painting", "inspection"],
            processing_times=[30, 25, 15, 10],
            priority=JobPriority.CRITICAL, quantity=10,
        ),
        ProductionJob(
            job_id="JOB-004", product_name="Cable Assembly",
            operations=["assembly", "inspection", "packaging"],
            processing_times=[20, 10, 5],
            priority=JobPriority.LOW, quantity=20,
        ),
        ProductionJob(
            job_id="JOB-005", product_name="Motor Mount",
            operations=["cnc", "welding", "assembly", "inspection"],
            processing_times=[50, 35, 25, 15],
            priority=JobPriority.HIGH, quantity=2,
            dependencies=["JOB-001"],
        ),
    ]

    result = engine.optimize(demo_jobs)
    return {
        "demo_jobs": [j.model_dump() for j in demo_jobs],
        "optimized_schedule": result.model_dump(),
    }
