"""
Statistical anomaly detection using rolling Z-scores.

Detects: cost spikes, latency regressions, quality drift, error bursts.

Usage:
    detector = AnomalyDetector(window=100, z_threshold=3.0)
    detector.observe("gpt-4o", "cost_usd", 0.002)
    anomaly = detector.check("gpt-4o", "cost_usd", 0.08)  # spike!
    if anomaly:
        alert_manager.send(anomaly)
"""

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.span import Span


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly."""
    model: str
    metric: str
    value: float
    baseline: float
    z_score: float
    severity: str  # low | medium | high | critical
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __str__(self):
        return (
            f"ANOMALY [{self.severity.upper()}] {self.model}/{self.metric}: "
            f"value={self.value:.4f} baseline={self.baseline:.4f} z={self.z_score:.2f}"
        )
    
    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "metric": self.metric,
            "value": self.value,
            "baseline": self.baseline,
            "z_score": self.z_score,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


class AnomalyDetector:
    """
    Statistical anomaly detection using rolling Z-scores.
    
    Maintains a rolling window of observations per model/metric
    and flags values that deviate significantly from the baseline.
    """
    
    def __init__(self, window: int = 200, z_threshold: float = 3.0):
        """
        Initialize the anomaly detector.
        
        Args:
            window: Size of rolling window for baseline computation
            z_threshold: Z-score threshold for anomaly detection
        """
        self.window = window
        self.z_threshold = z_threshold
        # {model: {metric: deque of values}}
        self._observations: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window))
        )
    
    def observe(self, model: str, metric: str, value: float) -> None:
        """
        Record a new observation.
        
        Call this for every span to build up the baseline.
        
        Args:
            model: Model name
            metric: Metric name
            value: Observed value
        """
        self._observations[model][metric].append(value)
    
    def check(
        self,
        model: str,
        metric: str,
        value: float
    ) -> Optional[AnomalyEvent]:
        """
        Check if a value is anomalous compared to historical baseline.
        
        Args:
            model: Model name
            metric: Metric name
            value: Value to check
            
        Returns:
            AnomalyEvent if anomalous, None otherwise
        """
        history = list(self._observations[model][metric])
        
        # Need minimum samples for baseline
        if len(history) < 10:
            return None
        
        mean = statistics.mean(history)
        stdev = statistics.stdev(history)
        
        if stdev == 0:
            return None
        
        z_score = abs((value - mean) / stdev)
        
        if z_score < self.z_threshold:
            return None
        
        # Determine severity
        if z_score > 6.0:
            severity = "critical"
        elif z_score > 5.0:
            severity = "high"
        elif z_score > 4.0:
            severity = "medium"
        else:
            severity = "low"
        
        return AnomalyEvent(
            model=model,
            metric=metric,
            value=value,
            baseline=mean,
            z_score=z_score,
            severity=severity,
        )
    
    def check_span(self, span: "Span") -> List[AnomalyEvent]:
        """
        Check all trackable metrics from a span at once.
        
        Args:
            span: Span to check
            
        Returns:
            List of detected anomalies
        """
        model = span.get_attribute("llm.model.name") or "unknown"
        events = []
        
        # Metrics to check
        metrics_to_check = {
            "cost_usd": span.get_attribute("cost.total_usd"),
            "duration_ms": span.duration_ms,
            "completion_tokens": span.get_attribute("llm.completion_tokens"),
            "quality_score": span.get_attribute("eval.quality"),
        }
        
        for metric, value in metrics_to_check.items():
            if value is not None:
                self.observe(model, metric, float(value))
                anomaly = self.check(model, metric, float(value))
                if anomaly:
                    events.append(anomaly)
        
        return events
    
    def get_baseline(self, model: str, metric: str) -> Optional[dict]:
        """
        Get baseline statistics for a model/metric.
        
        Args:
            model: Model name
            metric: Metric name
            
        Returns:
            Dict with mean, stdev, n or None if insufficient data
        """
        history = list(self._observations[model][metric])
        if len(history) < 2:
            return None
        
        return {
            "mean": statistics.mean(history),
            "stdev": statistics.stdev(history),
            "n": len(history),
        }
    
    def reset(self, model: Optional[str] = None, metric: Optional[str] = None) -> None:
        """
        Reset observations.
        
        Args:
            model: Model to reset (None = all)
            metric: Metric to reset (None = all for model)
        """
        if model is None:
            self._observations.clear()
        elif metric is None:
            self._observations.pop(model, None)
        else:
            if model in self._observations:
                self._observations[model].pop(metric, None)
