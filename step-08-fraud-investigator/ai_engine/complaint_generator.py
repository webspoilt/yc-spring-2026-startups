"""
FCA Regulatory Complaint Generator using GPT-4o.

Generates Financial Conduct Authority (FCA) compliant regulatory
complaint documents from fraud detection findings, including:
- Suspicious Activity Reports (SARs)
- Financial crime disclosures
- Evidence summaries with transaction details

Output follows FCA REP-CRIM reporting format.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class FraudEvidence(BaseModel):
    """Structured evidence package for a fraud case."""
    case_id: str = Field(..., description="Internal case reference")
    account_ids: List[str] = Field(..., description="Involved account IDs")
    total_suspicious_amount: float = Field(..., gt=0)
    currency: str = Field(default="GBP")
    detection_method: str = Field(..., description="benford | circular | fan_out | manual")
    anomaly_details: Dict[str, Any] = Field(default_factory=dict)
    transactions: List[Dict] = Field(default_factory=list,
                                     description="Supporting transaction records")
    risk_scores: Dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class FCAComplaint(BaseModel):
    """Generated FCA regulatory complaint document."""
    case_id: str
    report_type: str = Field(default="SAR", description="SAR | REP-CRIM | STR")
    subject: str
    summary: str
    evidence_narrative: str
    risk_assessment: str
    recommended_actions: List[str]
    regulatory_references: List[str]
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    confidence_score: float = Field(ge=0.0, le=1.0)


class FCAComplaintGenerator:
    """
    Uses GPT-4o to generate FCA-compliant regulatory complaints
    from structured fraud evidence.
    """

    SYSTEM_PROMPT = """You are a senior financial crime compliance officer at a UK-regulated 
financial institution. You generate FCA-compliant Suspicious Activity Reports (SARs) and 
regulatory complaint documents.

Your reports must:
1. Follow FCA REP-CRIM format guidelines
2. Include specific transaction details and amounts
3. Reference applicable regulations (POCA 2002, MLR 2017, FCA SYSC 6.3)
4. Provide clear risk assessment (Low/Medium/High/Critical)
5. List recommended enforcement actions
6. Be factual, precise, and free of speculation
7. Include relevant regulatory references

Output format: JSON matching the FCAComplaint schema with fields:
- subject: Brief case title
- summary: 2-3 paragraph executive summary
- evidence_narrative: Detailed evidence walkthrough
- risk_assessment: Risk level with justification
- recommended_actions: List of specific next steps
- regulatory_references: Applicable regulations
- confidence_score: 0-1 confidence in findings"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                return None
        return self._client

    def generate_complaint(self, evidence: FraudEvidence) -> FCAComplaint:
        """
        Generate an FCA regulatory complaint from fraud evidence.

        Args:
            evidence: structured fraud evidence package

        Returns:
            FCAComplaint with all generated sections
        """
        user_prompt = self._build_prompt(evidence)

        client = self._get_client()
        if client is None or not self.api_key:
            return self._generate_mock_complaint(evidence)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=4000,
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            return FCAComplaint(
                case_id=evidence.case_id,
                report_type=parsed.get("report_type", "SAR"),
                subject=parsed.get("subject", f"SAR - Case {evidence.case_id}"),
                summary=parsed.get("summary", ""),
                evidence_narrative=parsed.get("evidence_narrative", ""),
                risk_assessment=parsed.get("risk_assessment", "Medium"),
                recommended_actions=parsed.get("recommended_actions", []),
                regulatory_references=parsed.get("regulatory_references", []),
                confidence_score=parsed.get("confidence_score", 0.75),
            )

        except Exception as e:
            # Fallback to deterministic generation
            complaint = self._generate_mock_complaint(evidence)
            complaint.summary += f"\n\n[Note: AI generation failed ({str(e)}), using template-based report]"
            return complaint

    def _build_prompt(self, evidence: FraudEvidence) -> str:
        """Construct the GPT-4o prompt from structured evidence."""
        tx_summary = ""
        if evidence.transactions:
            tx_lines = []
            for i, tx in enumerate(evidence.transactions[:20], 1):
                tx_lines.append(
                    f"  {i}. {tx.get('sender_id', 'N/A')} → {tx.get('receiver_id', 'N/A')}: "
                    f"{tx.get('currency', 'GBP')} {tx.get('amount', 0):,.2f} "
                    f"on {tx.get('timestamp', 'N/A')}"
                    f"{' [FLAGGED]' if tx.get('is_suspicious') else ''}"
                )
            tx_summary = "\n".join(tx_lines)

        prompt = f"""Generate an FCA Suspicious Activity Report for the following case.

CASE REFERENCE: {evidence.case_id}
DETECTION METHOD: {evidence.detection_method}
TOTAL SUSPICIOUS AMOUNT: {evidence.currency} {evidence.total_suspicious_amount:,.2f}
ACCOUNTS INVOLVED: {', '.join(evidence.account_ids)}
DETECTION TIMESTAMP: {evidence.timestamp}

ANOMALY DETAILS:
{json.dumps(evidence.anomaly_details, indent=2)}

RISK SCORES:
{json.dumps(evidence.risk_scores, indent=2)}

SUPPORTING TRANSACTIONS:
{tx_summary if tx_summary else 'No individual transactions provided'}

Generate a complete SAR following FCA REP-CRIM guidelines. Return as JSON."""

        return prompt

    def _generate_mock_complaint(self, evidence: FraudEvidence) -> FCAComplaint:
        """
        Deterministic template-based complaint for when OpenAI is unavailable.
        Production-quality template following actual FCA SAR format.
        """
        detection_desc = {
            "benford": "statistical analysis using Benford's Law first-digit distribution test",
            "circular": "graph-based circular transfer pattern detection",
            "fan_out": "fan-out structuring pattern analysis",
            "manual": "manual compliance review",
        }.get(evidence.detection_method, evidence.detection_method)

        risk_level = "High" if evidence.total_suspicious_amount > 50000 else \
                     "Medium" if evidence.total_suspicious_amount > 10000 else "Low"

        return FCAComplaint(
            case_id=evidence.case_id,
            report_type="SAR",
            subject=f"Suspicious Activity Report - {evidence.detection_method.upper()} "
                    f"Detection - {evidence.currency} {evidence.total_suspicious_amount:,.2f}",
            summary=(
                f"This Suspicious Activity Report is submitted in accordance with the "
                f"Proceeds of Crime Act 2002 (POCA), Section 330. The automated fraud "
                f"detection system identified suspicious activity involving "
                f"{len(evidence.account_ids)} account(s) with a total suspicious value "
                f"of {evidence.currency} {evidence.total_suspicious_amount:,.2f}.\n\n"
                f"Detection was triggered by {detection_desc}. The activity patterns "
                f"are consistent with potential money laundering indicators as defined "
                f"in the Joint Money Laundering Steering Group (JMLSG) guidance."
            ),
            evidence_narrative=(
                f"Automated monitoring systems flagged anomalous activity across accounts: "
                f"{', '.join(evidence.account_ids)}. The detection method ({evidence.detection_method}) "
                f"identified the following anomalies:\n\n"
                f"{json.dumps(evidence.anomaly_details, indent=2)}\n\n"
                f"Risk scores for involved accounts: {json.dumps(evidence.risk_scores, indent=2)}\n\n"
                f"A total of {len(evidence.transactions)} transaction(s) have been flagged as "
                f"supporting evidence for this report."
            ),
            risk_assessment=(
                f"Risk Level: {risk_level}\n"
                f"Justification: Total suspicious amount of {evidence.currency} "
                f"{evidence.total_suspicious_amount:,.2f} detected through {detection_desc}. "
                f"{'Multiple accounts involved suggest coordinated activity.' if len(evidence.account_ids) > 1 else ''}"
            ),
            recommended_actions=[
                f"File SAR with NCA UK Financial Intelligence Unit (UKFIU)",
                f"Place enhanced monitoring on accounts: {', '.join(evidence.account_ids[:5])}",
                f"Conduct enhanced due diligence (EDD) review on flagged accounts",
                f"Suspend outbound transfers pending investigation" if risk_level == "High" else
                    f"Maintain enhanced monitoring for 90 days",
                f"Notify MLRO for case escalation",
                f"Preserve all relevant documentation for 5-year retention period",
            ],
            regulatory_references=[
                "Proceeds of Crime Act 2002 (POCA) - Section 330, 331, 332",
                "Money Laundering Regulations 2017 (MLR 2017) - Regulation 28",
                "FCA SYSC 6.3 - Financial crime",
                "FCA SUP 15.3 - Notification requirements",
                "JMLSG Guidance Part I, Chapter 6 - Suspicious Activity Reporting",
            ],
            confidence_score=0.85 if evidence.detection_method in ("benford", "circular") else 0.70,
        )

    def generate_batch(self, evidence_list: List[FraudEvidence]) -> List[FCAComplaint]:
        """Generate complaints for multiple fraud cases."""
        return [self.generate_complaint(e) for e in evidence_list]
