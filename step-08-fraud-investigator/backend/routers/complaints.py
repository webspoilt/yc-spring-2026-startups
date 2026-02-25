"""FCA complaint generation endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from ...ai_engine.complaint_generator import (
    FCAComplaintGenerator,
    FraudEvidence,
    FCAComplaint,
)

router = APIRouter()
generator = FCAComplaintGenerator()


class ComplaintRequest(BaseModel):
    case_id: str = Field(..., description="Internal case reference number")
    account_ids: List[str] = Field(..., min_length=1)
    total_suspicious_amount: float = Field(..., gt=0)
    currency: str = Field(default="GBP")
    detection_method: str = Field(
        default="benford",
        description="Detection method: benford | circular | fan_out | manual"
    )
    anomaly_details: Dict[str, Any] = Field(default_factory=dict)
    transactions: List[Dict] = Field(default_factory=list)
    risk_scores: Dict[str, float] = Field(default_factory=dict)


class BatchComplaintRequest(BaseModel):
    cases: List[ComplaintRequest]


@router.post("/generate", response_model=FCAComplaint)
async def generate_complaint(request: ComplaintRequest):
    """
    Generate an FCA-compliant Suspicious Activity Report (SAR)
    from fraud evidence using GPT-4o.
    """
    evidence = FraudEvidence(
        case_id=request.case_id,
        account_ids=request.account_ids,
        total_suspicious_amount=request.total_suspicious_amount,
        currency=request.currency,
        detection_method=request.detection_method,
        anomaly_details=request.anomaly_details,
        transactions=request.transactions,
        risk_scores=request.risk_scores,
    )
    complaint = generator.generate_complaint(evidence)
    return complaint


@router.post("/generate/batch")
async def generate_batch_complaints(request: BatchComplaintRequest):
    """Generate complaints for multiple fraud cases."""
    evidence_list = [
        FraudEvidence(
            case_id=case.case_id,
            account_ids=case.account_ids,
            total_suspicious_amount=case.total_suspicious_amount,
            currency=case.currency,
            detection_method=case.detection_method,
            anomaly_details=case.anomaly_details,
            transactions=case.transactions,
            risk_scores=case.risk_scores,
        )
        for case in request.cases
    ]
    complaints = generator.generate_batch(evidence_list)
    return {
        "generated": len(complaints),
        "complaints": [c.model_dump() for c in complaints],
    }
