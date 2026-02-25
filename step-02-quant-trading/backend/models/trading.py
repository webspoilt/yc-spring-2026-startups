from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"


class TradeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    action: TradeAction
    quantity: float = Field(..., gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


class TradeResponse(BaseModel):
    trade_id: str
    symbol: str
    action: TradeAction
    quantity: float
    price: float
    timestamp: datetime
    status: str


class Portfolio(BaseModel):
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    total_value: float
    daily_pnl: float
    total_pnl: float


class Position(BaseModel):
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    weight: float


class AnalysisRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    include_fundamental: bool = True
    include_technical: bool = True
    include_sentiment: bool = True


class AgentAnalysis(BaseModel):
    agent_name: str
    signal: TradeAction
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str
    data: Dict[str, Any]


class AnalysisResponse(BaseModel):
    symbol: str
    timestamp: datetime
    agents: List[AgentAnalysis]
    consensus: TradeAction
    risk_assessment: Dict[str, Any]
    recommended_action: TradeAction
    metadata: Dict[str, Any]


class RiskMetrics(BaseModel):
    portfolio_value: float
    daily_var: float  # Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    risk_score: int  # 1-10


class OrderBook(BaseModel):
    symbol: str
    bids: List[tuple[float, float]]  # price, quantity
    asks: List[tuple[float, float]]
    spread: float
    mid_price: float
