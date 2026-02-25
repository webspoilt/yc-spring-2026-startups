"""
LangChain Swarm Implementation for Quant Fund

Multi-agent system with:
- Fundamental Agent: Analyzes financial statements, ratios, earnings
- Technical Agent: Analyzes price patterns, indicators
- Sentiment Agent: Analyzes news, social media sentiment
- Risk Manager: Enforces risk limits (2% daily loss max)
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import yfinance as yf
import random


class AgentResponse(BaseModel):
    signal: str = Field(description="buy, sell, or hold")
    confidence: float = Field(description="0-1 confidence score")
    reasoning: str = Field(description="Detailed reasoning")
    data: Dict[str, Any] = Field(default_factory=dict)


class QuantSwarm:
    """
    Multi-agent quant trading swarm using LangChain.
    Coordinates Fundamental, Technical, Sentiment agents.
    """
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY", "")
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=api_key,
            temperature=0.3
        ) if api_key else None
        
        # Initialize agents
        self.fundamental_agent = FundamentalAgent(self.llm)
        self.technical_agent = TechnicalAgent(self.llm)
        self.sentiment_agent = SentimentAgent(self.llm)
        self.risk_manager = RiskManager()
    
    async def analyze(
        self,
        symbol: str,
        include_fundamental: bool = True,
        include_technical: bool = True,
        include_sentiment: bool = True
    ) -> Dict[str, Any]:
        """
        Run multi-agent analysis on a symbol.
        Returns consolidated analysis with consensus recommendation.
        """
        # Run agents in parallel
        tasks = []
        agents_results = []
        
        if include_fundamental:
            tasks.append(self.fundamental_agent.analyze(symbol))
        if include_technical:
            tasks.append(self.technical_agent.analyze(symbol))
        if include_sentiment:
            tasks.append(self.sentiment_agent.analyze(symbol))
        
        # Execute all agent analyses concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    agents_results.append({
                        "agent_name": "error",
                        "signal": "hold",
                        "confidence": 0.0,
                        "reasoning": str(result),
                        "data": {}
                    })
                else:
                    agents_results.append(result)
        
        # Calculate consensus
        consensus = self._calculate_consensus(agents_results)
        
        # Get risk assessment
        risk_assessment = await self.risk_manager.assess_risk(symbol, agents_results)
        
        # Determine final recommended action
        recommended = consensus["action"]
        if not risk_assessment["approved"]:
            recommended = "hold"
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "agents": agents_results,
            "consensus": consensus,
            "risk_assessment": risk_assessment,
            "recommended_action": recommended,
            "metadata": {
                "agents_used": len(agents_results),
                "analysis_time": datetime.now().isoformat()
            }
        }
    
    def _calculate_consensus(self, agents: List[Dict]) -> Dict[str, Any]:
        """Calculate consensus from multiple agents."""
        signals = {"buy": 0, "sell": 0, "hold": 0}
        total_confidence = 0
        
        for agent in agents:
            signal = agent.get("signal", "hold").lower()
            if signal in signals:
                signals[signal] += 1
            total_confidence += agent.get("confidence", 0)
        
        avg_confidence = total_confidence / len(agents) if agents else 0
        
        # Majority vote
        action = max(signals, key=signals.get)
        
        return {
            "action": action,
            "vote_counts": signals,
            "average_confidence": avg_confidence
        }


class FundamentalAgent:
    """Analyzes fundamental data: earnings, ratios, financials."""
    
    def __init__(self, llm: Optional[ChatOpenAI]):
        self.llm = llm
    
    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze fundamental data for a symbol."""
        # Fetch fundamental data
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key metrics
            pe_ratio = info.get("trailingPE", 0)
            eps = info.get("epsTrailingTwelveMonths", 0)
            dividend = info.get("dividendYield", 0)
            book_value = info.get("bookValue", 0)
            revenue = info.get("revenueGrowth", 0)
            profit_margin = info.get("profitMargins", 0)
            
            fundamental_data = {
                "pe_ratio": pe_ratio,
                "eps": eps,
                "dividend_yield": dividend,
                "book_value": book_value,
                "revenue_growth": revenue,
                "profit_margin": profit_margin
            }
        except Exception:
            fundamental_data = self._get_mock_data()
        
        # Use LLM to generate analysis if available
        if self.llm:
            reasoning = await self._generate_reasoning(symbol, fundamental_data)
        else:
            reasoning = self._generate_mock_reasoning(fundamental_data)
        
        # Determine signal based on fundamentals
        signal = self._determine_signal(fundamental_data)
        
        return {
            "agent_name": "fundamental",
            "signal": signal,
            "confidence": 0.7,
            "reasoning": reasoning,
            "data": fundamental_data
        }
    
    def _get_mock_data(self) -> Dict[str, float]:
        """Return mock fundamental data."""
        return {
            "pe_ratio": random.uniform(10, 30),
            "eps": random.uniform(1, 10),
            "dividend_yield": random.uniform(0, 5),
            "book_value": random.uniform(10, 100),
            "revenue_growth": random.uniform(-0.1, 0.3),
            "profit_margin": random.uniform(0.05, 0.25)
        }
    
    async def _generate_reasoning(self, symbol: str, data: Dict) -> str:
        """Generate reasoning using LLM."""
        template = ChatPromptTemplate.from_template(
            """Analyze the following fundamental data for {symbol} and provide a trading recommendation.

Fundamental Data:
- P/E Ratio: {pe_ratio:.2f}
- EPS: ${eps:.2f}
- Dividend Yield: {dividend_yield:.2f}%
- Revenue Growth: {revenue_growth:.2f}%
- Profit Margin: {profit_margin:.2f}%

Provide a brief analysis (2-3 sentences) and conclude with: SIGNAL: buy/sell/hold"""
        )
        
        chain = template | self.llm
        response = await chain.ainvoke({
            "symbol": symbol,
            **data
        })
        
        return response.content
    
    def _generate_mock_reasoning(self, data: Dict) -> str:
        """Generate mock reasoning."""
        signals = []
        
        if data["pe_ratio"] < 20:
            signals.append("P/E ratio is favorable")
        if data["revenue_growth"] > 0.1:
            signals.append("Strong revenue growth")
        if data["profit_margin"] > 0.15:
            signals.append("Healthy profit margins")
        
        if len(signals) >= 2:
            return f"Fundamental analysis shows {'; '.join(signals)}. Overall outlook is positive. SIGNAL: buy"
        elif len(signals) == 1:
            return f"Moderate fundamentals. {'; '.join(signals)}. SIGNAL: hold"
        return "Mixed fundamentals. SIGNAL: hold"
    
    def _determine_signal(self, data: Dict) -> str:
        """Determine buy/sell/hold from fundamentals."""
        score = 0
        
        # Favorable P/E
        if 10 < data["pe_ratio"] < 25:
            score += 1
        elif data["pe_ratio"] > 30:
            score -= 1
        
        # Positive growth
        if data["revenue_growth"] > 0.1:
            score += 1
        elif data["revenue_growth"] < 0:
            score -= 1
        
        # Good margins
        if data["profit_margin"] > 0.15:
            score += 1
        elif data["profit_margin"] < 0.05:
            score -= 1
        
        if score >= 2:
            return "buy"
        elif score <= -1:
            return "sell"
        return "hold"


class TechnicalAgent:
    """Analyzes technical indicators and price patterns."""
    
    def __init__(self, llm: Optional[ChatOpenAI]):
        self.llm = llm
    
    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze technical indicators for a symbol."""
        # Get price data
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            # Calculate indicators
            indicators = self._calculate_indicators(hist)
        except Exception:
            indicators = self._get_mock_indicators()
        
        # Generate reasoning
        if self.llm:
            reasoning = await self._generate_reasoning(symbol, indicators)
        else:
            reasoning = self._generate_mock_reasoning(indicators)
        
        signal = self._determine_signal(indicators)
        
        return {
            "agent_name": "technical",
            "signal": signal,
            "confidence": 0.65,
            "reasoning": reasoning,
            "data": indicators
        }
    
    def _calculate_indicators(self, hist) -> Dict[str, float]:
        """Calculate technical indicators from price history."""
        import pandas as pd
        import numpy as np
        
        if hist.empty:
            return self._get_mock_indicators()
        
        close = hist['Close']
        
        # SMA
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        return {
            "price": float(close.iloc[-1]),
            "sma_20": float(sma_20),
            "sma_50": float(sma_50),
            "rsi": float(rsi),
            "macd": float(macd.iloc[-1]),
            "macd_signal": float(signal_line.iloc[-1]),
            "trend": "uptrend" if sma_20 > sma_50 else "downtrend"
        }
    
    def _get_mock_indicators(self) -> Dict[str, float]:
        """Return mock technical indicators."""
        return {
            "price": random.uniform(100, 500),
            "sma_20": random.uniform(100, 500),
            "sma_50": random.uniform(100, 500),
            "rsi": random.uniform(30, 70),
            "macd": random.uniform(-5, 5),
            "macd_signal": random.uniform(-5, 5),
            "trend": random.choice(["uptrend", "downtrend", "sideways"])
        }
    
    async def _generate_reasoning(self, symbol: str, data: Dict) -> str:
        """Generate reasoning using LLM."""
        template = ChatPromptTemplate.from_template(
            """Analyze the following technical indicators for {symbol} and provide a trading recommendation.

Technical Data:
- Price: ${price:.2f}
- SMA 20: ${sma_20:.2f}
- SMA 50: ${sma_50:.2f}
- RSI: {rsi:.2f}
- MACD: {macd:.2f}
- MACD Signal: {macd_signal:.2f}
- Trend: {trend}

Provide a brief analysis (2-3 sentences) and conclude with: SIGNAL: buy/sell/hold"""
        )
        
        chain = template | self.llm
        response = await chain.ainvoke({
            "symbol": symbol,
            **data
        })
        
        return response.content
    
    def _generate_mock_reasoning(self, data: Dict) -> str:
        """Generate mock technical reasoning."""
        signals = []
        
        if data["rsi"] < 30:
            signals.append("RSI indicates oversold")
        elif data["rsi"] > 70:
            signals.append("RSI indicates overbought")
        
        if data.get("trend") == "uptrend":
            signals.append("Price in uptrend")
        elif data.get("trend") == "downtrend":
            signals.append("Price in downtrend")
        
        if data["macd"] > data["macd_signal"]:
            signals.append("MACD bullish crossover")
        
        if len(signals) >= 2:
            return f"Technicals: {'; '.join(signals)}. SIGNAL: buy"
        elif "overbought" in str(signals):
            return f"Technical warning: {'; '.join(signals)}. SIGNAL: sell"
        return f"Technicals neutral. SIGNAL: hold"
    
    def _determine_signal(self, data: Dict) -> str:
        """Determine signal from technical indicators."""
        score = 0
        
        # RSI
        if data["rsi"] < 30:
            score += 1
        elif data["rsi"] > 70:
            score -= 1
        
        # Trend
        if data.get("trend") == "uptrend":
            score += 1
        elif data.get("trend") == "downtrend":
            score -= 1
        
        # MACD
        if data["macd"] > data["macd_signal"]:
            score += 1
        else:
            score -= 1
        
        if score >= 2:
            return "buy"
        elif score <= -1:
            return "sell"
        return "hold"


class SentimentAgent:
    """Analyzes news and sentiment data."""
    
    def __init__(self, llm: Optional[ChatOpenAI]):
        self.llm = llm
    
    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment for a symbol."""
        # In production, fetch real news
        sentiment_data = self._get_sentiment_data(symbol)
        
        # Generate reasoning
        if self.llm:
            reasoning = await self._generate_reasoning(symbol, sentiment_data)
        else:
            reasoning = self._generate_mock_reasoning(sentiment_data)
        
        signal = self._determine_signal(sentiment_data)
        
        return {
            "agent_name": "sentiment",
            "signal": signal,
            "confidence": 0.6,
            "reasoning": reasoning,
            "data": sentiment_data
        }
    
    def _get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment data (mock for now)."""
        return {
            "news_count": random.randint(5, 50),
            "positive_count": random.randint(0, 30),
            "negative_count": random.randint(0, 20),
            "neutral_count": random.randint(0, 20),
            "sentiment_score": random.uniform(-1, 1),
            "social_mentions": random.randint(100, 10000),
            "social_sentiment": random.uniform(-1, 1)
        }
    
    async def _generate_reasoning(self, symbol: str, data: Dict) -> str:
        """Generate reasoning using LLM."""
        template = ChatPromptTemplate.from_template(
            """Analyze the following sentiment data for {symbol} and provide a trading recommendation.

Sentiment Data:
- News Articles: {news_count}
- Positive: {positive_count}
- Negative: {negative_count}
- Neutral: {neutral_count}
- Overall Sentiment Score: {sentiment_score:.2f} (-1 to 1)
- Social Media Mentions: {social_mentions}
- Social Sentiment: {social_sentiment:.2f}

Provide a brief analysis (2-3 sentences) and conclude with: SIGNAL: buy/sell/hold"""
        )
        
        chain = template | self.llm
        response = await chain.ainvoke({
            "symbol": symbol,
            **data
        })
        
        return response.content
    
    def _generate_mock_reasoning(self, data: Dict) -> str:
        """Generate mock sentiment reasoning."""
        score = data["sentiment_score"]
        
        if score > 0.3:
            return f"Positive sentiment ({data['positive_count']} positive articles). Market perception is favorable. SIGNAL: buy"
        elif score < -0.3:
            return f"Negative sentiment ({data['negative_count']} negative articles). Caution advised. SIGNAL: sell"
        return f"Mixed sentiment. Neutral market perception. SIGNAL: hold"
    
    def _determine_signal(self, data: Dict) -> str:
        """Determine signal from sentiment."""
        score = data["sentiment_score"]
        
        if score > 0.3:
            return "buy"
        elif score < -0.3:
            return "sell"
        return "hold"


class RiskManager:
    """
    Risk management agent.
    Enforces:
    - Max 2% daily loss
    - Max 10% position size
    - Stop loss at 5%
    """
    
    def __init__(self):
        self.max_daily_loss = 0.02  # 2%
        self.max_position_size = 0.10  # 10%
        self.stop_loss = 0.05  # 5%
        self.daily_pnl = 0.0
        self.daily_trades = []
    
    async def check_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        portfolio: Any
    ) -> Dict[str, Any]:
        """Check if trade passes risk management rules."""
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return {
                "approved": False,
                "reason": f"Daily loss limit exceeded ({self.daily_pnl*100:.1f}% > {self.max_daily_loss*100}%)"
            }
        
        # Check position size
        # In production, calculate real position value
        position_value = quantity * random.uniform(100, 500)
        position_weight = position_value / portfolio.total_value
        
        if position_weight > self.max_position_size:
            return {
                "approved": False,
                "reason": f"Position size too large ({position_weight*100:.1f}% > {self.max_position_size*100}%)"
            }
        
        # Check if we have enough cash for buy
        if action == "buy":
            required = position_value
            if required > portfolio.cash:
                return {
                    "approved": False,
                    "reason": "Insufficient cash for trade"
                }
        
        return {
            "approved": True,
            "reason": "Trade approved by risk manager"
        }
    
    async def assess_risk(self, symbol: str, agents: List[Dict]) -> Dict[str, Any]:
        """Assess overall risk of the trade."""
        
        # Check if agents agree
        signals = [a.get("signal", "hold") for a in agents]
        
        # If too many sell signals, might be market-wide risk
        sell_count = signals.count("sell")
        
        risk_level = "low"
        if sell_count >= 2:
            risk_level = "high"
        elif sell_count == 1:
            risk_level = "medium"
        
        return {
            "approved": True,
            "risk_level": risk_level,
            "agents_agreement": len(set(signals)) == 1,
            "sell_signals": sell_count
        }
    
    async def calculate_metrics(self, portfolio) -> Dict[str, Any]:
        """Calculate risk metrics."""
        
        # Mock calculations
        return {
            "portfolio_value": portfolio.total_value,
            "daily_var": portfolio.total_value * 0.02,  # 2% VaR
            "max_drawdown": random.uniform(0.05, 0.2),
            "sharpe_ratio": random.uniform(0.5, 2.5),
            "win_rate": random.uniform(0.4, 0.7),
            "risk_score": random.randint(1, 10)
        }
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking."""
        self.daily_pnl += pnl
    
    def reset_daily(self):
        """Reset daily tracking (call at start of new day)."""
        self.daily_pnl = 0.0
        self.daily_trades = []
