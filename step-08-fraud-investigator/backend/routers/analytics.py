"""Benford's Law analytics endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel, Field

from ...ai_engine.benford import BenfordAnalyzer, BenfordResult

router = APIRouter()
analyzer = BenfordAnalyzer()


class AmountsRequest(BaseModel):
    amounts: List[float] = Field(..., min_length=1,
                                 description="List of transaction amounts to analyze")
    confidence_level: float = Field(default=0.05, gt=0, lt=1)


class AccountTransactionsRequest(BaseModel):
    transactions: List[dict] = Field(..., min_length=1,
                                     description="List of {account_id, amount} dicts")


class StructuringRequest(BaseModel):
    amounts: List[float] = Field(..., min_length=1)
    threshold: float = Field(default=10000.0, gt=0,
                             description="Regulatory reporting threshold")
    margin: float = Field(default=0.1, gt=0, lt=1,
                          description="Fraction below threshold to check")


@router.post("/benford", response_model=BenfordResult)
async def run_benford_analysis(request: AmountsRequest):
    """
    Run Benford's Law first-digit analysis on transaction amounts.
    Returns chi-squared test, MAD, and per-digit anomaly flags.
    """
    result = analyzer.analyze(request.amounts, request.confidence_level)
    return result


@router.post("/benford/by-account")
async def benford_by_account(request: AccountTransactionsRequest):
    """Run per-account Benford's analysis."""
    results = analyzer.analyze_by_account(request.transactions)
    serialized = {}
    for acct, result in results.items():
        serialized[acct] = result.model_dump()
    return {
        "accounts_analyzed": len(serialized),
        "results": serialized,
        "flagged_accounts": [
            acct for acct, r in results.items() if r.is_anomalous
        ],
    }


@router.post("/structuring")
async def detect_structuring(request: StructuringRequest):
    """
    Detect transaction structuring (amounts clustered just below
    a reporting threshold).
    """
    result = analyzer.detect_structuring(
        request.amounts, request.threshold, request.margin
    )
    return result
