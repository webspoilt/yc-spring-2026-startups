from fastapi import APIRouter
from typing import List, Dict, Any
from datetime import datetime, timedelta
import random

router = APIRouter()


@router.get("/chain/blocks")
async def get_recent_blocks(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent Polygon blocks."""
    blocks = []
    for i in range(limit):
        blocks.append({
            "block_number": 50000000 - i * 1000,
            "timestamp": datetime.now() - timedelta(seconds=i*2),
            "transactions": random.randint(50, 200),
            "gas_used": random.randint(1000000, 5000000),
            "validator": f"0x{random.randint(0, 16**40):040x}"[:10]
        })
    return blocks


@router.get("/chain/gas")
async def get_gas_prices() -> Dict[str, Any]:
    """Get current gas prices."""
    return {
        "slow": random.uniform(20, 50),
        "standard": random.uniform(50, 100),
        "fast": random.uniform(100, 200),
        "unit": "gwei",
        "updated_at": datetime.now().isoformat()
    }


@router.get("/token/{token_address}/transfers")
async def get_token_transfers(
    token_address: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Get recent token transfers."""
    transfers = []
    for i in range(limit):
        transfers.append({
            "hash": f"0x{random.randint(0, 16**64):016x}",
            "from": f"0x{random.randint(0, 16**40):040x}"[:10],
            "to": f"0x{random.randint(0, 16**40):040x}"[:10],
            "value": random.uniform(100, 10000),
            "timestamp": datetime.now() - timedelta(minutes=i*5),
            "block": 50000000 - i * 100
        })
    return transfers


@router.get("/account/{address}/balance")
async def get_account_balance(address: str) -> Dict[str, Any]:
    """Get account token balances."""
    return {
        "address": address,
        "balances": [
            {"token": "USDC", "balance": random.uniform(1000, 100000)},
            {"token": "ETH", "balance": random.uniform(0.1, 10)},
            {"token": "MATIC", "balance": random.uniform(100, 10000)}
        ]
    }


@router.get("/events/alert")
async def get_alerts() -> List[Dict[str, Any]]:
    """Get monitoring alerts."""
    return [
        {
            "type": "price_deviation",
            "severity": "warning",
            "message": "USDC price deviation > 0.5%",
            "timestamp": datetime.now() - timedelta(minutes=5)
        },
        {
            "type": "large_transfer",
            "severity": "info",
            "message": "Large transfer detected: 50,000 USDC",
            "timestamp": datetime.now() - timedelta(minutes=15)
        }
    ]
