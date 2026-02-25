from fastapi import APIRouter, HTTPException, status
from typing import List
from datetime import datetime
import uuid
import random

from ..models.trading import (
    TradeRequest, TradeResponse, TradeAction, OrderType,
    Portfolio, Position, AnalysisRequest, AnalysisResponse,
    AgentAnalysis, RiskMetrics, OrderBook
)
from ...ai_engine.swarm import QuantSwarm

router = APIRouter()

# Initialize the quant swarm
swarm = QuantSwarm()


# In-memory storage (replace with database in production)
trades_storage: List[TradeResponse] = []
portfolio = Portfolio(
    cash=100000.0,
    positions={},
    total_value=100000.0,
    daily_pnl=0.0,
    total_pnl=0.0
)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_symbol(request: AnalysisRequest):
    """
    Run multi-agent analysis on a symbol.
    Uses Fundamental, Technical, and Sentiment agents.
    """
    try:
        result = await swarm.analyze(
            symbol=request.symbol,
            include_fundamental=request.include_fundamental,
            include_technical=request.include_technical,
            include_sentiment=request.include_sentiment
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/trade", response_model=TradeResponse)
async def execute_trade(request: TradeRequest):
    """
    Execute a trade with risk management checks.
    """
    global portfolio
    
    # Check with risk manager
    risk_check = await swarm.risk_manager.check_trade(
        symbol=request.symbol,
        action=request.action,
        quantity=request.quantity,
        portfolio=portfolio
    )
    
    if not risk_check["approved"]:
        raise HTTPException(
            status_code=400,
            detail=f"Trade rejected by risk manager: {risk_check['reason']}"
        )
    
    # Simulate trade execution
    current_price = random.uniform(100, 500)  # In production, get real price
    
    trade = TradeResponse(
        trade_id=str(uuid.uuid4()),
        symbol=request.symbol,
        action=request.action,
        quantity=request.quantity,
        price=current_price,
        timestamp=datetime.now(),
        status="executed"
    )
    
    trades_storage.append(trade)
    
    # Update portfolio
    if request.action == TradeAction.BUY:
        portfolio.cash -= current_price * request.quantity
        portfolio.positions[request.symbol] = (
            portfolio.positions.get(request.symbol, 0) + request.quantity
        )
    elif request.action == TradeAction.SELL:
        portfolio.cash += current_price * request.quantity
        portfolio.positions[request.symbol] = (
            portfolio.positions.get(request.symbol, 0) - request.quantity
        )
    
    # Recalculate total value
    portfolio.total_value = portfolio.cash + sum(
        random.uniform(100, 500) * qty 
        for symbol, qty in portfolio.positions.items()
    )
    
    return trade


@router.get("/portfolio", response_model=Portfolio)
async def get_portfolio():
    """Get current portfolio status."""
    return portfolio


@router.get("/positions", response_model=List[Position])
async def get_positions():
    """Get all current positions."""
    positions = []
    for symbol, quantity in portfolio.positions.items():
        if quantity > 0:
            current_price = random.uniform(100, 500)  # In production, get real price
            avg_entry = current_price * 0.95  # Simulated
            unrealized = (current_price - avg_entry) * quantity
            
            positions.append(Position(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=avg_entry,
                current_price=current_price,
                unrealized_pnl=unrealized,
                weight=(current_price * quantity) / portfolio.total_value
            ))
    return positions


@router.get("/trades", response_model=List[TradeResponse])
async def get_trades(limit: int = 50):
    """Get recent trades."""
    return trades_storage[-limit:]


@router.get("/risk/metrics", response_model=RiskMetrics)
async def get_risk_metrics():
    """Get current risk metrics."""
    return await swarm.risk_manager.calculate_metrics(portfolio)


@router.get("/risk/limits")
async def get_risk_limits():
    """Get current risk limits."""
    return {
        "max_position_size": 0.1,  # 10% of portfolio
        "max_loss_per_day": 0.02,  # 2% of portfolio
        "max_leverage": 1.0,
        "stop_loss_percentage": 0.05  # 5% stop loss
    }


@router.get("/orderbook/{symbol}", response_model=OrderBook)
async def get_orderbook(symbol: str):
    """Get simulated orderbook for a symbol."""
    mid_price = random.uniform(100, 500)
    spread = mid_price * 0.001
    
    bids = [(mid_price - spread - i * 0.01, random.uniform(10, 100)) for i in range(5)]
    asks = [(mid_price + spread + i * 0.01, random.uniform(10, 100)) for i in range(5)]
    
    return OrderBook(
        symbol=symbol,
        bids=bids,
        asks=asks,
        spread=spread,
        mid_price=mid_price
    )
