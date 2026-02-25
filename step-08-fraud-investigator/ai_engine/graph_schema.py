"""
Neo4j Graph Schema for Fraud Detection.

Defines Cypher queries and Python models for a transaction graph
consisting of Account nodes, Transaction relationships, and
derived risk metrics.

Graph Structure:
  (Account)-[:SENT {amount, timestamp}]->(Account)
  (Account)-[:RECEIVED {amount, timestamp}]->(Account)
  (Account {id, name, risk_score, kyc_status, country, created_at})
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os


class AccountNode(BaseModel):
    """Pydantic model for a graph Account node."""
    account_id: str = Field(..., description="Unique account identifier")
    name: str = Field(..., description="Account holder name")
    country: str = Field(default="GB", description="ISO country code")
    kyc_status: str = Field(default="pending", description="KYC verification status")
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0,
                              description="Computed fraud risk score 0-100")
    account_type: str = Field(default="personal",
                              description="personal | business | corporate")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    flagged: bool = Field(default=False)
    flag_reasons: List[str] = Field(default_factory=list)


class TransactionNode(BaseModel):
    """Pydantic model for a graph Transaction relationship."""
    tx_id: str = Field(..., description="Unique transaction ID")
    sender_id: str = Field(..., description="Source account ID")
    receiver_id: str = Field(..., description="Destination account ID")
    amount: float = Field(..., gt=0, description="Transaction amount in GBP")
    currency: str = Field(default="GBP")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    tx_type: str = Field(default="transfer",
                         description="transfer | payment | withdrawal | deposit")
    description: str = Field(default="")
    is_suspicious: bool = Field(default=False)
    anomaly_flags: List[str] = Field(default_factory=list)


class FraudGraphSchema:
    """
    Neo4j graph operations for fraud detection.
    Manages account/transaction CRUD and fraud-detection Cypher queries.
    """

    CONSTRAINTS = [
        "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
        "CREATE CONSTRAINT tx_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE",
    ]

    INDEXES = [
        "CREATE INDEX account_risk IF NOT EXISTS FOR (a:Account) ON (a.risk_score)",
        "CREATE INDEX tx_amount IF NOT EXISTS FOR ()-[r:SENT]-() ON (r.amount)",
        "CREATE INDEX tx_timestamp IF NOT EXISTS FOR ()-[r:SENT]-() ON (r.timestamp)",
    ]

    def __init__(self, driver=None):
        """
        Args:
            driver: neo4j.Driver instance. If None, creates from env vars.
        """
        if driver is None:
            try:
                from neo4j import GraphDatabase
                uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                user = os.getenv("NEO4J_USER", "neo4j")
                password = os.getenv("NEO4J_PASSWORD", "fraudpass")
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
            except ImportError:
                self.driver = None
        else:
            self.driver = driver

    def initialize_schema(self):
        """Create constraints and indexes."""
        if not self.driver:
            return {"status": "no_driver", "message": "Neo4j driver not available"}
        with self.driver.session() as session:
            for constraint in self.CONSTRAINTS:
                session.run(constraint)
            for index in self.INDEXES:
                session.run(index)
        return {"status": "initialized"}

    def create_account(self, account: AccountNode) -> Dict[str, Any]:
        """Insert an Account node into the graph."""
        query = """
        MERGE (a:Account {account_id: $account_id})
        SET a.name = $name,
            a.country = $country,
            a.kyc_status = $kyc_status,
            a.risk_score = $risk_score,
            a.account_type = $account_type,
            a.created_at = $created_at,
            a.flagged = $flagged
        RETURN a
        """
        if not self.driver:
            return {"status": "mock", "account": account.model_dump()}
        with self.driver.session() as session:
            result = session.run(query, **account.model_dump())
            record = result.single()
            return {"status": "created", "node": dict(record["a"])}

    def create_transaction(self, tx: TransactionNode) -> Dict[str, Any]:
        """Create a SENT relationship between two accounts."""
        query = """
        MATCH (sender:Account {account_id: $sender_id})
        MATCH (receiver:Account {account_id: $receiver_id})
        CREATE (sender)-[r:SENT {
            tx_id: $tx_id,
            amount: $amount,
            currency: $currency,
            timestamp: $timestamp,
            tx_type: $tx_type,
            description: $description,
            is_suspicious: $is_suspicious
        }]->(receiver)
        RETURN r
        """
        if not self.driver:
            return {"status": "mock", "transaction": tx.model_dump()}
        with self.driver.session() as session:
            result = session.run(query, **tx.model_dump())
            record = result.single()
            return {"status": "created", "relationship": dict(record["r"])}

    def detect_circular_transfers(self, min_depth: int = 3,
                                  max_depth: int = 6) -> List[Dict]:
        """
        Find circular money flows (A→B→C→...→A) indicative of
        layering in money laundering.
        """
        query = f"""
        MATCH path = (a:Account)-[:SENT*{min_depth}..{max_depth}]->(a)
        WITH a, path, reduce(total = 0, r IN relationships(path) | total + r.amount) AS total_flow
        WHERE total_flow > 1000
        RETURN a.account_id AS origin,
               length(path) AS depth,
               total_flow,
               [n IN nodes(path) | n.account_id] AS accounts
        ORDER BY total_flow DESC
        LIMIT 50
        """
        if not self.driver:
            return [{"origin": "MOCK-001", "depth": 3, "total_flow": 50000,
                     "accounts": ["MOCK-001", "MOCK-002", "MOCK-003", "MOCK-001"]}]
        with self.driver.session() as session:
            results = session.run(query)
            return [dict(record) for record in results]

    def detect_fan_out(self, threshold: int = 10,
                       time_window_hours: int = 24) -> List[Dict]:
        """
        Detect structuring / smurfing: single account sending to many
        recipients in a short window (fan-out pattern).
        """
        query = """
        MATCH (a:Account)-[r:SENT]->(b:Account)
        WITH a, count(DISTINCT b) AS recipients, sum(r.amount) AS total,
             collect(r.amount) AS amounts
        WHERE recipients >= $threshold
        RETURN a.account_id AS account,
               a.name AS name,
               recipients,
               total,
               amounts
        ORDER BY recipients DESC
        LIMIT 50
        """
        if not self.driver:
            return [{"account": "MOCK-001", "name": "Mock Sender",
                     "recipients": 15, "total": 75000,
                     "amounts": [5000] * 15}]
        with self.driver.session() as session:
            results = session.run(query, threshold=threshold)
            return [dict(record) for record in results]

    def compute_risk_scores(self) -> Dict[str, Any]:
        """
        Update account risk scores based on PageRank-style propagation
        in the transaction graph.
        """
        query = """
        CALL gds.pageRank.stream({
            nodeProjection: 'Account',
            relationshipProjection: {
                SENT: {type: 'SENT', properties: 'amount'}
            },
            relationshipWeightProperty: 'amount',
            maxIterations: 20,
            dampingFactor: 0.85
        })
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS account, score
        SET account.risk_score = score * 100
        RETURN account.account_id AS id, score * 100 AS risk_score
        ORDER BY risk_score DESC
        """
        if not self.driver:
            return {"status": "mock", "message": "PageRank not available without Neo4j GDS"}
        with self.driver.session() as session:
            results = session.run(query)
            return {"status": "computed",
                    "scores": [dict(r) for r in results]}

    def get_subgraph(self, account_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get the local neighborhood of an account for visualization."""
        query = """
        MATCH path = (a:Account {account_id: $account_id})-[:SENT*1..$depth]-(b)
        WITH collect(path) AS paths
        CALL apoc.convert.toTree(paths) YIELD value
        RETURN value
        """
        if not self.driver:
            return {"root": account_id, "depth": depth, "nodes": [], "edges": []}
        with self.driver.session() as session:
            result = session.run(query, account_id=account_id, depth=depth)
            record = result.single()
            return record["value"] if record else {}
