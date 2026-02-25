"""
Polygon Blockchain Monitor
Watches for events, transfers, and price changes.
"""

import asyncio
from typing import Dict, Any, List, Callable
from datetime import datetime
import random


class PolygonMonitor:
    """
    Monitors Polygon blockchain for events.
    """
    
    def __init__(self, rpc_url: str = None):
        self.rpc_url = rpc_url or "https://polygon-rpc.com"
        self.running = False
        self.subscribers: List[Callable] = []
    
    async def start(self):
        """Start monitoring."""
        self.running = True
        asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop monitoring."""
        self.running = False
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            # Check for new blocks
            block_data = await self._fetch_latest_block()
            
            # Emit events to subscribers
            for callback in self.subscribers:
                try:
                    callback(block_data)
                except Exception as e:
                    print(f"Subscriber error: {e}")
            
            await asyncio.sleep(2)  # Check every 2 seconds
    
    async def _fetch_latest_block(self) -> Dict[str, Any]:
        """Fetch latest block data (simulated)."""
        return {
            "block_number": random.randint(50000000, 51000000),
            "timestamp": datetime.now(),
            "transactions": random.randint(50, 200),
            "hash": f"0x{random.randint(0, 16**64):016x}"
        }
    
    def subscribe(self, callback: Callable):
        """Subscribe to block events."""
        self.subscribers.append(callback)
    
    async def get_token_transfers(
        self,
        token_address: str,
        from_block: int,
        to_block: int
    ) -> List[Dict[str, Any]]:
        """Get token transfers in a block range."""
        transfers = []
        for block in range(from_block, to_block + 1):
            transfers.append({
                "block": block,
                "hash": f"0x{random.randint(0, 16**64):016x}",
                "from": f"0x{random.randint(0, 16**40):040x}",
                "to": f"0x{random.randint(0, 16**40):040x}",
                "value": random.uniform(0, 100000)
            })
        return transfers
    
    async def watch_address(self, address: str) -> Dict[str, Any]:
        """Watch specific address for events."""
        return {
            "address": address,
            "incoming_transfers": random.randint(0, 50),
            "outgoing_transfers": random.randint(0, 30),
            "last_activity": datetime.now().isoformat()
        }


class PriceMonitor:
    """
    Monitors token prices for deviations.
    """
    
    def __init__(self):
        self.prices = {
            "USDC": 1.0,
            "USDT": 1.0,
            "DAI": 1.0,
            "ETH": random.uniform(2000, 3000),
            "MATIC": random.uniform(0.5, 1.5)
        }
        self.thresholds = {
            "stable": 0.005,  # 0.5% for stablecoins
            "volatile": 0.05  # 5% for volatile assets
        }
    
    async def get_price(self, token: str) -> float:
        """Get current price."""
        return self.prices.get(token, 0)
    
    async def check_price_deviation(
        self,
        token: str,
        old_price: float
    ) -> Dict[str, Any]:
        """Check if price deviated significantly."""
        current_price = await self.get_price(token)
        deviation = abs(current_price - old_price) / old_price
        
        threshold = self.thresholds["stable"] if token in ["USDC", "USDT", "DAI"] else self.thresholds["volatile"]
        
        return {
            "token": token,
            "old_price": old_price,
            "current_price": current_price,
            "deviation": deviation,
            "threshold": threshold,
            "alert": deviation > threshold
        }
    
    async def get_all_prices(self) -> Dict[str, float]:
        """Get all tracked prices."""
        return self.prices.copy()
