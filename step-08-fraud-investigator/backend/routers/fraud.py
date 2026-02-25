"""Fraud detection and graph operations endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field

from ...ai_engine.graph_schema import FraudGraphSchema, AccountNode, TransactionNode

router = APIRouter()
graph = FraudGraphSchema()


class AccountCreateRequest(BaseModel):
    account_id: str
    name: str
    country: str = "GB"
    account_type: str = "personal"


class TransactionCreateRequest(BaseModel):
    tx_id: str
    sender_id: str
    receiver_id: str
    amount: float = Field(gt=0)
    currency: str = "GBP"
    tx_type: str = "transfer"
    description: str = ""


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionCreateRequest]


@router.post("/initialize")
async def initialize_graph():
    """Initialize Neo4j schema with constraints and indexes."""
    result = graph.initialize_schema()
    return result


@router.post("/accounts")
async def create_account(request: AccountCreateRequest):
    """Create a new account node in the fraud graph."""
    account = AccountNode(
        account_id=request.account_id,
        name=request.name,
        country=request.country,
        account_type=request.account_type,
    )
    result = graph.create_account(account)
    return result


@router.post("/transactions")
async def create_transaction(request: TransactionCreateRequest):
    """Record a transaction between two accounts."""
    tx = TransactionNode(
        tx_id=request.tx_id,
        sender_id=request.sender_id,
        receiver_id=request.receiver_id,
        amount=request.amount,
        currency=request.currency,
        tx_type=request.tx_type,
        description=request.description,
    )
    result = graph.create_transaction(tx)
    return result


@router.post("/transactions/batch")
async def batch_create_transactions(request: BatchTransactionRequest):
    """Bulk-insert transactions."""
    results = []
    for tx_req in request.transactions:
        tx = TransactionNode(**tx_req.model_dump())
        results.append(graph.create_transaction(tx))
    return {"created": len(results), "results": results}


@router.get("/detect/circular")
async def detect_circular(min_depth: int = 3, max_depth: int = 6):
    """Detect circular money transfer patterns (layering)."""
    findings = graph.detect_circular_transfers(min_depth, max_depth)
    return {
        "pattern": "circular_transfer",
        "findings_count": len(findings),
        "findings": findings,
    }


@router.get("/detect/fan-out")
async def detect_fan_out(threshold: int = 10):
    """Detect fan-out structuring patterns (smurfing)."""
    findings = graph.detect_fan_out(threshold)
    return {
        "pattern": "fan_out_structuring",
        "findings_count": len(findings),
        "findings": findings,
    }


@router.post("/risk-scores")
async def compute_risk_scores():
    """Recompute all account risk scores via PageRank."""
    result = graph.compute_risk_scores()
    return result


@router.get("/subgraph/{account_id}")
async def get_subgraph(account_id: str, depth: int = 2):
    """Get transaction subgraph centered on an account."""
    subgraph = graph.get_subgraph(account_id, depth)
    return {"account_id": account_id, "depth": depth, "subgraph": subgraph}
