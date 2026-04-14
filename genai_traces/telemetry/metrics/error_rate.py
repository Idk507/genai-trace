"""
Error rate tracking for GenAI-Traces.

Tracks rolling error rates with exponential decay.
"""

from typing import Dict, Optional, Any, List
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime
import threading
import time


@dataclass
class ErrorSample:
    """A single error/success sample."""
    timestamp: float
    is_error: bool
    error_type: Optional[str] = None


@dataclass
class ErrorRateStats:
    """Error rate statistics."""
    
    error_rate: float
    total_requests: int
    error_count: int
    success_count: int
    window_seconds: float
    error_types: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_rate": round(self.error_rate, 4),
            "error_rate_pct": round(self.error_rate * 100, 2),
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "window_seconds": self.window_seconds,
            "error_types": self.error_types,
        }


class ErrorRateTracker:
    """
    Tracks error rates over a sliding time window.
    """
    
    def __init__(self, window_seconds: float = 300.0):
        """
        Initialize the error rate tracker.
        
        Args:
            window_seconds: Time window for error rate calculation (default 5 min)
        """
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._samples: Dict[str, deque] = defaultdict(deque)
    
    def record_success(self, key: str) -> None:
        """Record a successful request."""
        self._record(key, is_error=False)
    
    def record_error(self, key: str, error_type: Optional[str] = None) -> None:
        """
        Record an error.
        
        Args:
            key: Identifier (e.g., model name)
            error_type: Type of error (e.g., "rate_limit", "timeout")
        """
        self._record(key, is_error=True, error_type=error_type)
    
    def _record(
        self, 
        key: str, 
        is_error: bool, 
        error_type: Optional[str] = None
    ) -> None:
        """Record a sample."""
        sample = ErrorSample(
            timestamp=time.time(),
            is_error=is_error,
            error_type=error_type,
        )
        
        with self._lock:
            self._samples[key].append(sample)
            self._cleanup(key)
    
    def get_stats(self, key: str) -> Optional[ErrorRateStats]:
        """
        Get error rate statistics for a key.
        
        Args:
            key: Identifier to get stats for
            
        Returns:
            ErrorRateStats if data exists, None otherwise
        """
        with self._lock:
            if key not in self._samples:
                return None
            
            self._cleanup(key)
            samples = list(self._samples[key])
        
        if not samples:
            return None
        
        error_count = sum(1 for s in samples if s.is_error)
        success_count = len(samples) - error_count
        
        error_types: Dict[str, int] = defaultdict(int)
        for s in samples:
            if s.is_error and s.error_type:
                error_types[s.error_type] += 1
        
        if len(samples) > 1:
            time_span = samples[-1].timestamp - samples[0].timestamp
        else:
            time_span = self._window_seconds
        
        return ErrorRateStats(
            error_rate=error_count / len(samples) if samples else 0.0,
            total_requests=len(samples),
            error_count=error_count,
            success_count=success_count,
            window_seconds=time_span,
            error_types=dict(error_types),
        )
    
    def get_all_stats(self) -> Dict[str, ErrorRateStats]:
        """Get statistics for all tracked keys."""
        with self._lock:
            keys = list(self._samples.keys())
        
        return {key: self.get_stats(key) for key in keys if self.get_stats(key)}
    
    def get_error_rate(self, key: str) -> float:
        """Get current error rate (0.0 to 1.0)."""
        stats = self.get_stats(key)
        return stats.error_rate if stats else 0.0
    
    def is_healthy(self, key: str, threshold: float = 0.1) -> bool:
        """
        Check if error rate is below threshold.
        
        Args:
            key: Identifier
            threshold: Maximum acceptable error rate (default 10%)
            
        Returns:
            True if error rate is below threshold
        """
        return self.get_error_rate(key) < threshold
    
    def get_unhealthy_keys(self, threshold: float = 0.1) -> List[str]:
        """Get all keys with error rate above threshold."""
        unhealthy = []
        with self._lock:
            keys = list(self._samples.keys())
        
        for key in keys:
            if not self.is_healthy(key, threshold):
                unhealthy.append(key)
        
        return unhealthy
    
    def clear(self, key: Optional[str] = None) -> None:
        """Clear samples for a key or all keys."""
        with self._lock:
            if key:
                self._samples.pop(key, None)
            else:
                self._samples.clear()
    
    def _cleanup(self, key: str) -> None:
        """Remove samples outside the time window."""
        cutoff = time.time() - self._window_seconds
        samples = self._samples[key]
        
        while samples and samples[0].timestamp < cutoff:
            samples.popleft()


_tracker = ErrorRateTracker()


def get_error_rate_tracker() -> ErrorRateTracker:
    """Get the global error rate tracker instance."""
    return _tracker


def record_success(key: str) -> None:
    """Convenience function to record success."""
    _tracker.record_success(key)


def record_error(key: str, error_type: Optional[str] = None) -> None:
    """Convenience function to record error."""
    _tracker.record_error(key, error_type)
