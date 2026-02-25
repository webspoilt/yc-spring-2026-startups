from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()


@router.get("/agents/status")
async def get_agent_status() -> Dict[str, Any]:
    """Get status of all trading agents."""
    return {
        "fundamental": {
            "status": "active",
            "model": "gpt-3.5-turbo",
            "data_sources": ["yfinance", "financial_ratios"]
        },
        "technical": {
            "status": "active", 
            "model": "gpt-3.5-turbo",
            "indicators": ["RSI", "MACD", "SMA", "EMA", "Bollinger"]
        },
        "sentiment": {
            "status": "active",
            "model": "gpt-3.5-turbo",
            "sources": ["news", "social_media"]
        },
        "risk_manager": {
            "status": "active",
            "max_daily_loss": 0.02,
            "max_position_size": 0.1
        }
    }


@router.get("/strategies")
async def list_strategies() -> Dict[str, Any]:
    """List available trading strategies."""
    return {
        "strategies": [
            {
                "name": "momentum",
                "description": "Follows price momentum using technical indicators",
                "agents": ["technical", "sentiment"]
            },
            {
                "name": "value",
                "description": "Fundamental value investing approach",
                "agents": ["fundamental"]
            },
            {
                "name": "multi_agent",
                "description": "Combines all agents for comprehensive analysis",
                "agents": ["fundamental", "technical", "sentiment", "risk_manager"]
            }
        ]
    }


@router.get("/history/{symbol}")
async def get_symbol_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
) -> Dict[str, Any]:
    """Get historical price data for a symbol."""
    import random
    
    # In production, use yfinance
    data = []
    base_price = random.uniform(100, 500)
    
    for i in range(250):  # ~1 year of trading days
        change = random.uniform(-0.05, 0.05)
        base_price *= (1 + change)
        data.append({
            "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "open": base_price * 0.99,
            "high": base_price * 1.02,
            "low": base_price * 0.98,
            "close": base_price,
            "volume": random.randint(1000000, 10000000)
        })
    
    return {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "data": data
    }
