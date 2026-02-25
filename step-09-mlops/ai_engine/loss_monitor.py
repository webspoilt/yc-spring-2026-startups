"""
Loss Spike Monitor — Training Health Anomaly Detection.

Monitors training loss curves in real-time and detects:
- Sudden loss spikes (>2σ above rolling mean)
- Loss divergence (NaN / Inf)
- Learning rate anomalies
- Gradient norm explosions
- Plateau detection (no improvement for N steps)

Provides configurable alerting and automatic remediation suggestions.
"""

import math
import statistics
from datetime import datetime
from collections import deque
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SpikeAlert(BaseModel):
    """Alert triggered by the loss monitor."""
    alert_id: str
    severity: AlertSeverity
    alert_type: str = Field(..., description="spike | divergence | plateau | gradient_explosion")
    step: int
    message: str
    metric_value: float
    threshold: float
    rolling_mean: float
    rolling_std: float
    remediation: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class TrainingMetrics(BaseModel):
    """Snapshot of training metrics at a single step."""
    step: int
    loss: float
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    throughput_samples_per_sec: float = 0.0
    gpu_memory_used_mib: int = 0
    epoch: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class MonitorConfig(BaseModel):
    """Configuration for the loss spike monitor."""
    window_size: int = Field(default=100, description="Rolling window for statistics")
    spike_sigma_threshold: float = Field(default=3.0, description="Std devs for spike detection")
    divergence_threshold: float = Field(default=1e6, description="Loss considered diverged")
    plateau_patience: int = Field(default=500, description="Steps without improvement for plateau")
    plateau_min_delta: float = Field(default=1e-4, description="Minimum improvement to reset plateau")
    gradient_norm_threshold: float = Field(default=100.0, description="Max gradient norm before alert")
    lr_spike_ratio: float = Field(default=10.0, description="LR jump ratio for alert")
    max_alerts_per_type: int = Field(default=50, description="Max alerts to store per type")
    cooldown_steps: int = Field(default=20, description="Min steps between same-type alerts")


class MonitorState(BaseModel):
    """Serializable monitor state for persistence."""
    total_steps: int = 0
    best_loss: float = float("inf")
    best_loss_step: int = 0
    plateau_counter: int = 0
    alert_count: int = 0
    last_alert_step: Dict[str, int] = Field(default_factory=dict)


class LossSpikeMonitor:
    """
    Real-time training loss monitor with anomaly detection.

    Ingests per-step training metrics and checks for:
    1. Loss spikes: sudden jumps beyond rolling mean + k*sigma
    2. Loss divergence: NaN, Inf, or extremely large values
    3. Plateau: no improvement in loss for N consecutive steps
    4. Gradient explosion: gradient norm exceeding threshold
    5. LR anomalies: sudden learning rate jumps

    Usage:
        monitor = LossSpikeMonitor()
        for step, loss in training_loop:
            alerts = monitor.ingest(TrainingMetrics(step=step, loss=loss))
            if alerts:
                handle_alerts(alerts)
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.state = MonitorState()

        # Rolling windows
        self._loss_history = deque(maxlen=self.config.window_size)
        self._grad_history = deque(maxlen=self.config.window_size)
        self._lr_history = deque(maxlen=self.config.window_size)

        # All metrics for reporting
        self._all_metrics: List[TrainingMetrics] = []
        self._all_alerts: List[SpikeAlert] = []

    def ingest(self, metrics: TrainingMetrics) -> List[SpikeAlert]:
        """
        Process a single step's training metrics. Returns any triggered alerts.

        Args:
            metrics: TrainingMetrics for the current step

        Returns:
            List of SpikeAlert triggered (empty if healthy)
        """
        alerts = []
        self.state.total_steps += 1
        self._all_metrics.append(metrics)

        # 1. Check for divergence (NaN / Inf / extremely large)
        if math.isnan(metrics.loss) or math.isinf(metrics.loss):
            alerts.append(self._create_alert(
                "divergence", AlertSeverity.CRITICAL, metrics.step,
                metrics.loss, 0, 0, 0,
                f"Loss diverged at step {metrics.step}: {metrics.loss}",
                [
                    "Immediately reduce learning rate by 10x",
                    "Check for data corruption in current batch",
                    "Enable gradient clipping if not already active",
                    "Restore from last checkpoint and retry with lower LR",
                ],
            ))
            return alerts

        if metrics.loss > self.config.divergence_threshold:
            alerts.append(self._create_alert(
                "divergence", AlertSeverity.CRITICAL, metrics.step,
                metrics.loss, self.config.divergence_threshold, 0, 0,
                f"Loss {metrics.loss:.4e} exceeds divergence threshold {self.config.divergence_threshold:.4e}",
                [
                    "Reduce learning rate by 5-10x",
                    "Increase gradient clipping value",
                    "Check for numerical instability in model architecture",
                ],
            ))

        # 2. Check for loss spike
        if len(self._loss_history) >= 10:
            mean = statistics.mean(self._loss_history)
            std = statistics.stdev(self._loss_history) if len(self._loss_history) > 1 else 0
            threshold = mean + self.config.spike_sigma_threshold * max(std, 1e-8)

            if metrics.loss > threshold and self._can_alert("spike", metrics.step):
                severity = (AlertSeverity.CRITICAL
                            if metrics.loss > mean + 5 * max(std, 1e-8)
                            else AlertSeverity.WARNING)
                alerts.append(self._create_alert(
                    "spike", severity, metrics.step,
                    metrics.loss, threshold, mean, std,
                    f"Loss spike at step {metrics.step}: {metrics.loss:.6f} "
                    f"(mean={mean:.6f}, std={std:.6f}, threshold={threshold:.6f})",
                    [
                        "Monitor next 10 steps for recovery",
                        "If persistent, reduce learning rate",
                        "Check system logs for GPU errors or data issues",
                        "Consider reverting to previous checkpoint",
                    ],
                ))

        # 3. Check for plateau
        if metrics.loss < self.state.best_loss - self.config.plateau_min_delta:
            self.state.best_loss = metrics.loss
            self.state.best_loss_step = metrics.step
            self.state.plateau_counter = 0
        else:
            self.state.plateau_counter += 1

        if (self.state.plateau_counter >= self.config.plateau_patience and
                self._can_alert("plateau", metrics.step)):
            alerts.append(self._create_alert(
                "plateau", AlertSeverity.WARNING, metrics.step,
                metrics.loss, self.state.best_loss,
                float(self.state.plateau_counter), 0,
                f"Training plateau: no improvement for {self.state.plateau_counter} steps "
                f"(best={self.state.best_loss:.6f} at step {self.state.best_loss_step})",
                [
                    "Consider reducing learning rate (cosine annealing or step decay)",
                    "Try increasing model capacity or data augmentation",
                    "Evaluate if training has converged (check validation metrics)",
                    "Experiment with different optimizer (switch AdamW ↔ LAMB)",
                ],
            ))

        # 4. Check gradient norm
        if metrics.gradient_norm > 0:
            if (metrics.gradient_norm > self.config.gradient_norm_threshold and
                    self._can_alert("gradient_explosion", metrics.step)):
                alerts.append(self._create_alert(
                    "gradient_explosion", AlertSeverity.WARNING, metrics.step,
                    metrics.gradient_norm, self.config.gradient_norm_threshold,
                    statistics.mean(self._grad_history) if self._grad_history else 0, 0,
                    f"Gradient norm explosion at step {metrics.step}: "
                    f"{metrics.gradient_norm:.4f} > {self.config.gradient_norm_threshold}",
                    [
                        "Reduce gradient clipping value",
                        "Lower learning rate",
                        "Check for exploding activations in specific layers",
                    ],
                ))
            self._grad_history.append(metrics.gradient_norm)

        # 5. Check LR anomalies
        if metrics.learning_rate > 0 and len(self._lr_history) >= 5:
            avg_lr = statistics.mean(self._lr_history)
            if (metrics.learning_rate > avg_lr * self.config.lr_spike_ratio and
                    self._can_alert("lr_anomaly", metrics.step)):
                alerts.append(self._create_alert(
                    "lr_anomaly", AlertSeverity.WARNING, metrics.step,
                    metrics.learning_rate, avg_lr * self.config.lr_spike_ratio,
                    avg_lr, 0,
                    f"Learning rate jump at step {metrics.step}: "
                    f"{metrics.learning_rate:.2e} (avg={avg_lr:.2e})",
                    [
                        "Verify LR scheduler configuration",
                        "Check for warmup restart if using cosine with restarts",
                    ],
                ))
        if metrics.learning_rate > 0:
            self._lr_history.append(metrics.learning_rate)

        # Update rolling window
        self._loss_history.append(metrics.loss)

        # Store alerts
        self._all_alerts.extend(alerts)
        self.state.alert_count += len(alerts)

        return alerts

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the monitor's current state."""
        loss_list = list(self._loss_history)
        return {
            "total_steps": self.state.total_steps,
            "best_loss": self.state.best_loss,
            "best_loss_step": self.state.best_loss_step,
            "current_loss": loss_list[-1] if loss_list else None,
            "rolling_mean": statistics.mean(loss_list) if loss_list else None,
            "rolling_std": statistics.stdev(loss_list) if len(loss_list) > 1 else None,
            "plateau_counter": self.state.plateau_counter,
            "total_alerts": self.state.alert_count,
            "recent_alerts": [a.model_dump() for a in self._all_alerts[-10:]],
            "alert_counts_by_type": self._count_alerts_by_type(),
        }

    def get_loss_curve(self) -> Dict[str, Any]:
        """Return the full loss curve data for visualization."""
        return {
            "steps": [m.step for m in self._all_metrics],
            "losses": [m.loss for m in self._all_metrics],
            "learning_rates": [m.learning_rate for m in self._all_metrics],
            "gradient_norms": [m.gradient_norm for m in self._all_metrics],
            "spike_steps": [
                a.step for a in self._all_alerts if a.alert_type == "spike"
            ],
        }

    def reset(self):
        """Reset the monitor (e.g., after loading a checkpoint)."""
        self._loss_history.clear()
        self._grad_history.clear()
        self._lr_history.clear()
        self._all_metrics.clear()
        self._all_alerts.clear()
        self.state = MonitorState()

    def _create_alert(self, alert_type: str, severity: AlertSeverity,
                      step: int, value: float, threshold: float,
                      mean: float, std: float, message: str,
                      remediation: List[str]) -> SpikeAlert:
        alert = SpikeAlert(
            alert_id=f"{alert_type}_{step}_{datetime.utcnow().strftime('%H%M%S')}",
            severity=severity,
            alert_type=alert_type,
            step=step,
            message=message,
            metric_value=value,
            threshold=threshold,
            rolling_mean=mean,
            rolling_std=std,
            remediation=remediation,
        )
        self.state.last_alert_step[alert_type] = step
        return alert

    def _can_alert(self, alert_type: str, step: int) -> bool:
        """Enforce cooldown between same-type alerts."""
        last = self.state.last_alert_step.get(alert_type, -999)
        return step - last >= self.config.cooldown_steps

    def _count_alerts_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for a in self._all_alerts:
            counts[a.alert_type] = counts.get(a.alert_type, 0) + 1
        return counts
