"""
Benford's Law Anomaly Detection for Financial Transactions.

Benford's Law states that in many naturally occurring datasets, the
leading digit d occurs with probability: P(d) = log10(1 + 1/d).

Expected distribution of leading digits:
  1: 30.1%, 2: 17.6%, 3: 12.5%, 4: 9.7%, 5: 7.9%
  6: 6.7%, 7: 5.8%, 8: 5.1%, 9: 4.6%

Deviations from this distribution in transaction amounts can indicate:
  - Fabricated invoices or round-number fraud
  - Structuring (amounts just under reporting thresholds)
  - Data manipulation / artificial entries
"""

import math
from collections import Counter
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np


class BenfordResult(BaseModel):
    """Result of Benford's Law analysis."""
    total_transactions: int
    digit_counts: Dict[int, int] = Field(default_factory=dict)
    observed_distribution: Dict[int, float] = Field(default_factory=dict)
    expected_distribution: Dict[int, float] = Field(default_factory=dict)
    chi_squared: float = 0.0
    p_value: float = 0.0
    mad: float = Field(0.0, description="Mean Absolute Deviation from Benford's")
    is_anomalous: bool = False
    anomaly_level: str = Field("normal",
                               description="normal | marginal | suspicious | critical")
    flagged_digits: List[int] = Field(default_factory=list)
    details: str = ""


class BenfordAnalyzer:
    """
    Statistical analyzer applying Benford's Law to financial transaction
    amounts. Computes chi-squared test, Mean Absolute Deviation, and
    per-digit z-scores to flag anomalous distributions.
    """

    # Benford's expected proportions for first digit
    EXPECTED = {d: math.log10(1 + 1 / d) for d in range(1, 10)}

    # MAD thresholds (Nigrini, 2012)
    MAD_THRESHOLDS = {
        "conforming": 0.006,
        "acceptable": 0.012,
        "marginal": 0.015,
        "suspicious": 0.022,
    }

    def __init__(self):
        self.expected_array = np.array([self.EXPECTED[d] for d in range(1, 10)])

    @staticmethod
    def extract_leading_digit(value: float) -> Optional[int]:
        """
        Extract the first significant (non-zero) digit from a number.

        Examples:
            1234.56 → 1
            0.00567 → 5
            -89.12  → 8
        """
        value = abs(value)
        if value == 0:
            return None
        # Normalize to get the leading digit
        while value < 1:
            value *= 10
        while value >= 10:
            value /= 10
        return int(value)

    def analyze(self, amounts: List[float],
                confidence_level: float = 0.05) -> BenfordResult:
        """
        Run full Benford's Law analysis on a list of transaction amounts.

        Args:
            amounts: list of transaction amounts (positive values)
            confidence_level: significance level for chi-squared test

        Returns:
            BenfordResult with all metrics and anomaly classification
        """
        # Filter and extract leading digits
        digits = []
        for amt in amounts:
            d = self.extract_leading_digit(amt)
            if d is not None and 1 <= d <= 9:
                digits.append(d)

        n = len(digits)
        if n < 50:
            return BenfordResult(
                total_transactions=n,
                details="Insufficient data: need at least 50 transactions for reliable analysis",
                anomaly_level="insufficient_data",
            )

        # Count digit frequencies
        digit_counts = Counter(digits)
        observed = {d: digit_counts.get(d, 0) / n for d in range(1, 10)}
        observed_array = np.array([observed.get(d, 0) for d in range(1, 10)])

        # Chi-squared statistic
        chi_squared = float(np.sum(
            n * (observed_array - self.expected_array) ** 2 / self.expected_array
        ))

        # P-value from chi-squared distribution (df=8)
        from scipy import stats
        p_value = float(1 - stats.chi2.cdf(chi_squared, df=8))

        # Mean Absolute Deviation
        mad = float(np.mean(np.abs(observed_array - self.expected_array)))

        # Per-digit z-scores to find which digits are anomalous
        flagged_digits = []
        for d in range(1, 10):
            expected_p = self.EXPECTED[d]
            observed_p = observed.get(d, 0)
            se = math.sqrt(expected_p * (1 - expected_p) / n)
            if se > 0:
                z = abs(observed_p - expected_p) / se
                if z > 2.576:  # 99% confidence
                    flagged_digits.append(d)

        # Classify anomaly level
        if mad <= self.MAD_THRESHOLDS["conforming"]:
            level = "normal"
        elif mad <= self.MAD_THRESHOLDS["acceptable"]:
            level = "acceptable"
        elif mad <= self.MAD_THRESHOLDS["marginal"]:
            level = "marginal"
        elif mad <= self.MAD_THRESHOLDS["suspicious"]:
            level = "suspicious"
        else:
            level = "critical"

        is_anomalous = p_value < confidence_level or level in ("suspicious", "critical")

        # Build detail message
        details_parts = []
        if is_anomalous:
            details_parts.append(f"ALERT: Transaction amounts deviate from Benford's Law (p={p_value:.4f})")
        if flagged_digits:
            details_parts.append(f"Anomalous leading digits: {flagged_digits}")
        if level in ("suspicious", "critical"):
            details_parts.append(f"MAD={mad:.4f} exceeds threshold for '{level}' classification")

        return BenfordResult(
            total_transactions=n,
            digit_counts=dict(digit_counts),
            observed_distribution={d: round(observed.get(d, 0), 4) for d in range(1, 10)},
            expected_distribution={d: round(self.EXPECTED[d], 4) for d in range(1, 10)},
            chi_squared=round(chi_squared, 4),
            p_value=round(p_value, 6),
            mad=round(mad, 6),
            is_anomalous=is_anomalous,
            anomaly_level=level,
            flagged_digits=flagged_digits,
            details=" | ".join(details_parts) if details_parts else "Distribution conforms to Benford's Law",
        )

    def analyze_by_account(self, transactions: List[Dict]) -> Dict[str, BenfordResult]:
        """
        Per-account Benford's analysis.

        Args:
            transactions: list of dicts with 'account_id' and 'amount' keys

        Returns:
            dict mapping account_id → BenfordResult
        """
        account_amounts: Dict[str, List[float]] = {}
        for tx in transactions:
            acct = tx.get("account_id") or tx.get("sender_id", "unknown")
            amount = tx.get("amount", 0)
            if amount > 0:
                account_amounts.setdefault(acct, []).append(amount)

        results = {}
        for acct, amounts in account_amounts.items():
            results[acct] = self.analyze(amounts)

        return results

    def detect_structuring(self, amounts: List[float],
                           threshold: float = 10000.0,
                           margin: float = 0.1) -> Dict:
        """
        Detect structuring: unusual clustering of amounts just below
        a reporting threshold (e.g., many transactions at $9,500-$9,999
        when the reporting threshold is $10,000).

        Args:
            amounts: transaction amounts
            threshold: regulatory reporting threshold
            margin: fraction below threshold to check (default 10%)

        Returns:
            dict with structuring analysis results
        """
        lower = threshold * (1 - margin)
        just_below = [a for a in amounts if lower <= a < threshold]
        pct = len(just_below) / max(len(amounts), 1) * 100

        # Under normal distribution, ~10% would fall in 10% band
        expected_pct = margin * 100
        is_suspicious = pct > expected_pct * 3  # 3x expected concentration

        return {
            "threshold": threshold,
            "check_range": f"{lower:.0f} - {threshold:.0f}",
            "transactions_in_range": len(just_below),
            "total_transactions": len(amounts),
            "percentage": round(pct, 2),
            "expected_percentage": expected_pct,
            "is_suspicious": is_suspicious,
            "detail": f"{'ALERT: ' if is_suspicious else ''}{pct:.1f}% of transactions "
                      f"cluster just below {threshold} threshold (expected ~{expected_pct:.0f}%)"
        }
