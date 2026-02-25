"""Quality inspection endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field

from ai_engine.quality_alerts import (
    QualityInspector, ProductSpec, QualityMeasurement,
    QualityAlert, InspectionResult, ToleranceSpec,
)

router = APIRouter()
inspector = QualityInspector()


class InspectionRequest(BaseModel):
    """Request to run quality inspection."""
    spec: ProductSpec
    measurements: List[QualityMeasurement]


class QuickInspectionRequest(BaseModel):
    """Simplified inspection request using example spec."""
    measurements: List[QualityMeasurement]


@router.post("/inspect", response_model=InspectionResult)
async def run_inspection(request: InspectionRequest):
    """
    Inspect measurements against a product specification.
    Returns pass/fail results, Cpk estimates, and quality alerts.
    """
    result = inspector.inspect(request.spec, request.measurements)
    return result


@router.post("/inspect/quick", response_model=InspectionResult)
async def quick_inspection(request: QuickInspectionRequest):
    """Run inspection using the built-in example product spec."""
    spec = inspector.example_spec()
    result = inspector.inspect(spec, request.measurements)
    return result


@router.get("/example-spec", response_model=ProductSpec)
async def get_example_spec():
    """Get the built-in example product specification."""
    return inspector.example_spec()


@router.post("/demo")
async def demo_inspection():
    """Run a demo inspection with sample measurements."""
    import random

    spec = inspector.example_spec()
    measurements = []

    for tol in spec.tolerances:
        for i in range(20):
            # Most measurements in-spec, some near edge, some out
            if random.random() < 0.85:
                # In-spec measurement
                noise = random.gauss(0, tol.upper_tolerance * 0.3)
            elif random.random() < 0.5:
                # Warning zone
                noise = random.gauss(0, tol.upper_tolerance * 0.8)
            else:
                # Out of spec
                noise = random.gauss(0, tol.upper_tolerance * 2.0)

            value = tol.nominal + noise
            measurements.append(QualityMeasurement(
                dimension=tol.dimension,
                measured_value=round(value, 4),
                unit=tol.unit,
                part_id=f"PART-{i+1:04d}",
                machine_id="cnc-01",
            ))

    result = inspector.inspect(spec, measurements)
    return {
        "spec": spec.model_dump(),
        "measurement_count": len(measurements),
        "result": result.model_dump(),
    }
