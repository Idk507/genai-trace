"""
Latency tracking with percentile computation.

Provides P50/P90/P95/P99 latency metrics via rolling window.
"""

from typing import Dict, List, Optional, Any
from collections import deque, defaultdict
from dataclasses import dataclass
import statistics
import threading


@dataclass
class LatencyStats:
    """Latency statistics for a model/operation."""
    
    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
        }


class LatencyTracker:
    """
    Tracks latency metrics with rolling window percentile computation.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize the latency tracker.
        
        Args:
            window_size: Number of samples to keep in rolling window
        """
        self._window_size = window_size
        self._lock = threading.Lock()
        self._observations: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
    
    def record(self, key: str, latency_ms: float) -> None:
        """
        Record a latency observation.
        
        Args:
            key: Identifier (e.g., model name, operation type)
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            self._observations[key].append(latency_ms)
    
    def get_stats(self, key: str) -> Optional[LatencyStats]:
        """
        Get latency statistics for a key.
        
        Args:
            key: Identifier to get stats for
            
        Returns:
            LatencyStats if data exists, None otherwise
        """
        with self._lock:
            if key not in self._observations:
                return None
            
            values = list(self._observations[key])
        
        if not values:
            return None
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return LatencyStats(
            count=n,
            mean_ms=statistics.mean(values),
            min_ms=min(values),
            max_ms=max(values),
            p50_ms=self._percentile(sorted_values, 50),
            p90_ms=self._percentile(sorted_values, 90),
            p95_ms=self._percentile(sorted_values, 95),
            p99_ms=self._percentile(sorted_values, 99),
            std_dev_ms=statistics.stdev(values) if n > 1 else 0.0,
        )
    
    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Get statistics for all tracked keys."""
        with self._lock:
            keys = list(self._observations.keys())
        
        return {key: self.get_stats(key) for key in keys if self.get_stats(key)}
    
    def is_slow(
        self, 
        key: str, 
        latency_ms: float, 
        percentile: int = 95
    ) -> bool:
        """
        Check if a latency is considered slow (above percentile threshold).
        
        Args:
            key: Identifier
            latency_ms: Latency to check
            percentile: Percentile threshold (default P95)
            
        Returns:
            True if latency exceeds the percentile threshold
        """
        stats = self.get_stats(key)
        if not stats:
            return False
        
        threshold = getattr(stats, f"p{percentile}_ms", stats.p95_ms)
        return latency_ms > threshold
    
    def clear(self, key: Optional[str] = None) -> None:
        """Clear observations for a key or all keys."""
        with self._lock:
            if key:
                self._observations.pop(key, None)
            else:
                self._observations.clear()
    
    def _percentile(self, sorted_values: List[float], p: int) -> float:
        """Calculate percentile from sorted values."""
        n = len(sorted_values)
        if n == 0:
            return 0.0
        
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f
        
        if f == c:
            return sorted_values[f]
        
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


_tracker = LatencyTracker()


def get_latency_tracker() -> LatencyTracker:
    """Get the global latency tracker instance."""
    return _tracker


def record_latency(key: str, latency_ms: float) -> None:
    """Convenience function to record latency."""
    _tracker.record(key, latency_ms)
