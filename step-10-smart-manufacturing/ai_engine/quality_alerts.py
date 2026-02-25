"""
ProductSpec Quality Inspection and Alert System.

Defines product quality specifications with tolerance ranges for
multiple dimensions and attributes. Inspects sensor data and
production outputs against specs, generating alerts for out-of-spec
conditions with severity classification and root cause hints.
"""

import statistics
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"


class ToleranceSpec(BaseModel):
    """Tolerance specification for a single quality dimension."""
    dimension: str = Field(..., description="Quality dimension name")
    unit: str = Field(default="mm")
    nominal: float = Field(..., description="Target value")
    upper_tolerance: float = Field(..., description="Maximum acceptable deviation above nominal")
    lower_tolerance: float = Field(..., description="Maximum acceptable deviation below nominal")
    upper_warning: Optional[float] = Field(None, description="Warning threshold above nominal")
    lower_warning: Optional[float] = Field(None, description="Warning threshold below nominal")
    measurement_method: str = Field(default="automatic", description="automatic | manual | cmm")
    critical: bool = Field(default=False, description="If true, out-of-spec triggers HALT")


class ProductSpec(BaseModel):
    """Complete product quality specification."""
    spec_id: str = Field(..., description="Specification identifier")
    product_name: str
    version: str = "1.0"
    tolerances: List[ToleranceSpec]
    material: str = Field(default="")
    process_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Required process conditions (temp, pressure, etc.)"
    )
    inspection_frequency: str = Field(
        default="every_part",
        description="every_part | every_nth | batch_sample | shift_start"
    )
    notes: str = ""


class QualityMeasurement(BaseModel):
    """A single quality measurement for inspection."""
    dimension: str
    measured_value: float
    unit: str = "mm"
    part_id: Optional[str] = None
    machine_id: Optional[str] = None
    operator_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class QualityAlert(BaseModel):
    """Alert triggered by out-of-spec quality measurement."""
    alert_id: str
    severity: AlertSeverity
    spec_id: str
    product_name: str
    dimension: str
    measured_value: float
    nominal_value: float
    tolerance_range: str
    deviation: float
    deviation_pct: float
    message: str
    root_cause_hints: List[str]
    recommended_actions: List[str]
    part_id: Optional[str] = None
    machine_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class InspectionResult(BaseModel):
    """Result of inspecting a set of measurements against a spec."""
    spec_id: str
    product_name: str
    total_measurements: int
    passed: int
    failed: int
    warnings: int
    pass_rate: float
    alerts: List[QualityAlert]
    dimension_summary: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    cpk_estimates: Dict[str, float] = Field(
        default_factory=dict,
        description="Process capability index estimates per dimension"
    )
    overall_status: str = Field(default="pass",
                                description="pass | warning | fail | halt")


class QualityInspector:
    """
    Inspects production measurements against product specifications
    and generates quality alerts.

    Features:
    - Multi-dimension tolerance checking with warning/critical thresholds
    - Process capability (Cpk) estimation
    - Root cause analysis hints based on deviation patterns
    - SPC (Statistical Process Control) trend detection
    - Automatic alert severity escalation
    """

    # Root cause hint database based on deviation direction and dimension type
    ROOT_CAUSE_DB = {
        "length": {
            "high": [
                "Tool wear causing insufficient material removal",
                "Incorrect G-code offset or tool length compensation",
                "Thermal expansion in machine or workpiece",
                "Work holding loosened during machining",
            ],
            "low": [
                "Excessive tool engagement or feed rate",
                "Tool breakage or chipping",
                "Material defect (voids, soft spots)",
                "Incorrect raw stock dimensions",
            ],
        },
        "diameter": {
            "high": [
                "Tool radius compensation error",
                "Spindle runout or bearing wear",
                "Incorrect tool selection",
            ],
            "low": [
                "Oversize tool wear",
                "Excessive cutting pressure causing deflection",
                "Material springback after machining",
            ],
        },
        "temperature": {
            "high": [
                "Insufficient cooling or coolant flow issue",
                "Excessive cutting speed",
                "Machine bearing degradation",
                "Ambient temperature exceeding spec",
            ],
            "low": [
                "Over-cooling causing thermal contraction",
                "Material not pre-heated per procedure",
            ],
        },
        "default": {
            "high": [
                "Process parameter drift — recalibrate equipment",
                "Raw material variation — verify incoming QC",
                "Environmental condition change",
            ],
            "low": [
                "Process parameter drift — recalibrate equipment",
                "Tooling wear or damage",
                "Measurement system error — verify gauge calibration",
            ],
        },
    }

    def inspect(self, spec: ProductSpec,
                measurements: List[QualityMeasurement]) -> InspectionResult:
        """
        Inspect a batch of measurements against the product specification.

        Args:
            spec: product quality specification
            measurements: list of quality measurements

        Returns:
            InspectionResult with pass/fail counts, alerts, and Cpk estimates
        """
        alerts = []
        passed = 0
        failed = 0
        warnings = 0
        overall_status = "pass"

        # Group measurements by dimension
        dim_measurements: Dict[str, List[QualityMeasurement]] = {}
        for m in measurements:
            dim_measurements.setdefault(m.dimension, []).append(m)

        # Check each tolerance
        tol_map = {t.dimension: t for t in spec.tolerances}

        for dim_name, dim_meas in dim_measurements.items():
            tol = tol_map.get(dim_name)
            if tol is None:
                continue

            for m in dim_meas:
                result, severity, deviation = self._check_tolerance(tol, m.measured_value)

                if result == "pass":
                    passed += 1
                elif result == "warning":
                    warnings += 1
                    passed += 1  # Warnings still pass
                    alert = self._create_alert(
                        spec, tol, m, severity, deviation
                    )
                    alerts.append(alert)
                    if overall_status == "pass":
                        overall_status = "warning"
                else:
                    failed += 1
                    alert = self._create_alert(
                        spec, tol, m, severity, deviation
                    )
                    alerts.append(alert)
                    if severity == AlertSeverity.HALT:
                        overall_status = "halt"
                    elif overall_status not in ("halt",):
                        overall_status = "fail"

        # Compute dimension summaries and Cpk
        dim_summary = {}
        cpk_estimates = {}
        for dim_name, dim_meas in dim_measurements.items():
            tol = tol_map.get(dim_name)
            values = [m.measured_value for m in dim_meas]
            if len(values) >= 2 and tol:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                usl = tol.nominal + tol.upper_tolerance
                lsl = tol.nominal - tol.lower_tolerance

                # Cpk = min((USL - mean) / 3σ, (mean - LSL) / 3σ)
                if std_val > 0:
                    cpk_upper = (usl - mean_val) / (3 * std_val)
                    cpk_lower = (mean_val - lsl) / (3 * std_val)
                    cpk = min(cpk_upper, cpk_lower)
                else:
                    cpk = float("inf")
                cpk_estimates[dim_name] = round(cpk, 3)

                dim_summary[dim_name] = {
                    "count": len(values),
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "nominal": tol.nominal,
                    "usl": usl,
                    "lsl": lsl,
                    "cpk": cpk_estimates[dim_name],
                    "capable": cpk >= 1.33,
                }

        total = passed + failed
        pass_rate = (passed / total * 100) if total > 0 else 0.0

        return InspectionResult(
            spec_id=spec.spec_id,
            product_name=spec.product_name,
            total_measurements=total,
            passed=passed,
            failed=failed,
            warnings=warnings,
            pass_rate=round(pass_rate, 2),
            alerts=alerts,
            dimension_summary=dim_summary,
            cpk_estimates=cpk_estimates,
            overall_status=overall_status,
        )

    def _check_tolerance(self, tol: ToleranceSpec,
                         value: float) -> Tuple[str, AlertSeverity, float]:
        """Check a single value against tolerance spec."""
        deviation = value - tol.nominal
        upper_limit = tol.nominal + tol.upper_tolerance
        lower_limit = tol.nominal - tol.lower_tolerance

        # Critical out-of-spec
        if value > upper_limit or value < lower_limit:
            severity = AlertSeverity.HALT if tol.critical else AlertSeverity.CRITICAL
            return "fail", severity, deviation

        # Warning zone
        if tol.upper_warning is not None and value > tol.nominal + tol.upper_warning:
            return "warning", AlertSeverity.WARNING, deviation
        if tol.lower_warning is not None and value < tol.nominal - tol.lower_warning:
            return "warning", AlertSeverity.WARNING, deviation

        return "pass", AlertSeverity.INFO, deviation

    def _create_alert(self, spec: ProductSpec, tol: ToleranceSpec,
                      measurement: QualityMeasurement,
                      severity: AlertSeverity,
                      deviation: float) -> QualityAlert:
        """Create a quality alert with root cause hints."""
        direction = "high" if deviation > 0 else "low"
        dim_type = self._classify_dimension(tol.dimension)
        hints = self.ROOT_CAUSE_DB.get(dim_type, self.ROOT_CAUSE_DB["default"])
        root_causes = hints.get(direction, hints.get("high", []))

        upper_limit = tol.nominal + tol.upper_tolerance
        lower_limit = tol.nominal - tol.lower_tolerance
        deviation_pct = (abs(deviation) / abs(tol.nominal)) * 100 if tol.nominal != 0 else 0

        actions = [
            f"Quarantine affected parts for review",
            f"Re-measure using {tol.measurement_method} method to confirm",
            f"Check machine calibration for {measurement.machine_id or 'assigned machine'}",
        ]
        if severity in (AlertSeverity.CRITICAL, AlertSeverity.HALT):
            actions.insert(0, "STOP PRODUCTION on this machine immediately")
            actions.append("Notify quality manager and production supervisor")
            actions.append("Initiate 8D problem-solving process")

        return QualityAlert(
            alert_id=f"QA-{spec.spec_id}-{tol.dimension}-{datetime.utcnow().strftime('%H%M%S')}",
            severity=severity,
            spec_id=spec.spec_id,
            product_name=spec.product_name,
            dimension=tol.dimension,
            measured_value=measurement.measured_value,
            nominal_value=tol.nominal,
            tolerance_range=f"{lower_limit:.4f} — {upper_limit:.4f}",
            deviation=round(deviation, 4),
            deviation_pct=round(deviation_pct, 2),
            message=(
                f"{severity.value.upper()}: {tol.dimension} = {measurement.measured_value:.4f} "
                f"(nominal {tol.nominal:.4f}, tolerance ±{tol.upper_tolerance:.4f}), "
                f"deviation {deviation:+.4f} ({deviation_pct:.1f}%)"
            ),
            root_cause_hints=root_causes[:3],
            recommended_actions=actions,
            part_id=measurement.part_id,
            machine_id=measurement.machine_id,
        )

    @staticmethod
    def _classify_dimension(dimension_name: str) -> str:
        """Classify dimension name to root cause category."""
        name_lower = dimension_name.lower()
        if any(k in name_lower for k in ("length", "width", "height", "depth", "thickness")):
            return "length"
        if any(k in name_lower for k in ("diameter", "bore", "radius", "od", "id")):
            return "diameter"
        if any(k in name_lower for k in ("temp", "thermal")):
            return "temperature"
        return "default"

    @staticmethod
    def example_spec() -> ProductSpec:
        """Generate an example product specification for testing."""
        return ProductSpec(
            spec_id="SPEC-BRACKET-001",
            product_name="Mounting Bracket Assembly",
            version="2.1",
            material="Aluminum 6061-T6",
            tolerances=[
                ToleranceSpec(
                    dimension="length", unit="mm", nominal=150.0,
                    upper_tolerance=0.05, lower_tolerance=0.05,
                    upper_warning=0.03, lower_warning=0.03,
                    critical=True,
                ),
                ToleranceSpec(
                    dimension="width", unit="mm", nominal=75.0,
                    upper_tolerance=0.1, lower_tolerance=0.1,
                    upper_warning=0.06, lower_warning=0.06,
                ),
                ToleranceSpec(
                    dimension="bore_diameter", unit="mm", nominal=12.0,
                    upper_tolerance=0.02, lower_tolerance=0.02,
                    upper_warning=0.01, lower_warning=0.01,
                    critical=True, measurement_method="cmm",
                ),
                ToleranceSpec(
                    dimension="surface_roughness_Ra", unit="µm", nominal=1.6,
                    upper_tolerance=0.4, lower_tolerance=0.4,
                    upper_warning=0.2, lower_warning=0.2,
                ),
            ],
            process_requirements={
                "machining_coolant": "flood",
                "spindle_speed_rpm": {"min": 2000, "max": 4000},
                "feed_rate_mm_per_rev": {"min": 0.1, "max": 0.3},
                "heat_treatment": "T6 temper verified",
            },
            inspection_frequency="every_part",
            notes="Critical safety component — 100% inspection required",
        )
