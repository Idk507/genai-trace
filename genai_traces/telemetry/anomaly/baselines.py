"""
Baseline computation for anomaly detection.

Provides per-model rolling baseline computation.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import statistics


@dataclass
class BaselineStats:
    """Statistics for a baseline."""
    mean: float
    std: float
    min_value: float
    max_value: float
    sample_count: int
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min_value,
            "max": self.max_value,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated.isoformat(),
        }


class RollingBaseline:
    """
    Computes rolling baseline statistics for a metric.
    
    Usage:
        baseline = RollingBaseline(window_size=1000)
        
        for latency in latencies:
            baseline.add(latency)
        
        stats = baseline.get_stats()
        print(f"Mean latency: {stats.mean:.2f}ms")
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        min_samples: int = 10,
    ):
        self._window_size = window_size
        self._min_samples = min_samples
        self._values: deque = deque(maxlen=window_size)
        self._sum = 0.0
        self._sum_sq = 0.0
    
    def add(self, value: float) -> None:
        """Add a value to the baseline."""
        if len(self._values) == self._window_size:
            old_value = self._values[0]
            self._sum -= old_value
            self._sum_sq -= old_value * old_value
        
        self._values.append(value)
        self._sum += value
        self._sum_sq += value * value
    
    def get_stats(self) -> Optional[BaselineStats]:
        """Get current baseline statistics."""
        n = len(self._values)
        
        if n < self._min_samples:
            return None
        
        mean = self._sum / n
        
        variance = (self._sum_sq / n) - (mean * mean)
        std = variance ** 0.5 if variance > 0 else 0.0
        
        return BaselineStats(
            mean=mean,
            std=std,
            min_value=min(self._values),
            max_value=max(self._values),
            sample_count=n,
        )
    
    def is_anomaly(self, value: float, z_threshold: float = 3.0) -> bool:
        """Check if a value is anomalous."""
        stats = self.get_stats()
        if stats is None or stats.std == 0:
            return False
        
        z_score = abs(value - stats.mean) / stats.std
        return z_score > z_threshold
    
    def get_z_score(self, value: float) -> Optional[float]:
        """Get z-score for a value."""
        stats = self.get_stats()
        if stats is None or stats.std == 0:
            return None
        
        return (value - stats.mean) / stats.std
    
    def clear(self) -> None:
        """Clear all values."""
        self._values.clear()
        self._sum = 0.0
        self._sum_sq = 0.0


class ModelBaseline:
    """
    Per-model baseline tracking.
    
    Tracks baselines for different metrics per model.
    
    Usage:
        baseline = ModelBaseline()
        
        baseline.add("gpt-4", "latency", 150.0)
        baseline.add("gpt-4", "tokens", 500)
        
        stats = baseline.get_stats("gpt-4", "latency")
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        min_samples: int = 10,
    ):
        self._window_size = window_size
        self._min_samples = min_samples
        self._baselines: Dict[str, Dict[str, RollingBaseline]] = {}
    
    def add(self, model: str, metric: str, value: float) -> None:
        """Add a value for a model/metric combination."""
        if model not in self._baselines:
            self._baselines[model] = {}
        
        if metric not in self._baselines[model]:
            self._baselines[model][metric] = RollingBaseline(
                window_size=self._window_size,
                min_samples=self._min_samples,
            )
        
        self._baselines[model][metric].add(value)
    
    def get_stats(self, model: str, metric: str) -> Optional[BaselineStats]:
        """Get baseline stats for a model/metric."""
        if model not in self._baselines:
            return None
        if metric not in self._baselines[model]:
            return None
        
        return self._baselines[model][metric].get_stats()
    
    def is_anomaly(
        self,
        model: str,
        metric: str,
        value: float,
        z_threshold: float = 3.0,
    ) -> bool:
        """Check if a value is anomalous for a model/metric."""
        if model not in self._baselines:
            return False
        if metric not in self._baselines[model]:
            return False
        
        return self._baselines[model][metric].is_anomaly(value, z_threshold)
    
    def get_all_stats(self) -> Dict[str, Dict[str, BaselineStats]]:
        """Get all baseline stats."""
        result = {}
        
        for model, metrics in self._baselines.items():
            result[model] = {}
            for metric, baseline in metrics.items():
                stats = baseline.get_stats()
                if stats:
                    result[model][metric] = stats
        
        return result
    
    def get_models(self) -> List[str]:
        """Get list of tracked models."""
        return list(self._baselines.keys())
    
    def get_metrics(self, model: str) -> List[str]:
        """Get list of tracked metrics for a model."""
        if model not in self._baselines:
            return []
        return list(self._baselines[model].keys())


_global_baseline: Optional[ModelBaseline] = None


def get_model_baseline() -> ModelBaseline:
    """Get the global model baseline tracker."""
    global _global_baseline
    if _global_baseline is None:
        _global_baseline = ModelBaseline()
    return _global_baseline


def record_baseline(model: str, metric: str, value: float) -> None:
    """Record a value to the global baseline."""
    get_model_baseline().add(model, metric, value)


def check_anomaly(
    model: str,
    metric: str,
    value: float,
    z_threshold: float = 3.0,
) -> bool:
    """Check if a value is anomalous using global baseline."""
    return get_model_baseline().is_anomaly(model, metric, value, z_threshold)
