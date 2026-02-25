"""
Step 8: Fraud Detection System AI Engine
=========================================
- Neo4j graph schema for transaction networks
- Benford's Law anomaly detection
- GPT-4o FCA regulatory complaint generator
"""

from .graph_schema import FraudGraphSchema, TransactionNode, AccountNode
from .benford import BenfordAnalyzer
from .complaint_generator import FCAComplaintGenerator

__all__ = [
    "FraudGraphSchema",
    "TransactionNode",
    "AccountNode",
    "BenfordAnalyzer",
    "FCAComplaintGenerator",
]
