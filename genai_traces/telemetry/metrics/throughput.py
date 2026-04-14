"""
Throughput tracking for GenAI-Traces.

Tracks tokens per second and requests per second.
"""

from typing import Dict, Optional, Any
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time


@dataclass
class ThroughputStats:
    """Throughput statistics."""
    
    tokens_per_second: float
    requests_per_second: float
    total_tokens: int
    total_requests: int
    window_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens_per_second": round(self.tokens_per_second, 2),
            "requests_per_second": round(self.requests_per_second, 2),
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "window_seconds": self.window_seconds,
        }


@dataclass
class ThroughputSample:
    """A single throughput sample."""
    timestamp: float
    tokens: int
    requests: int = 1


class ThroughputTracker:
    """
    Tracks throughput metrics over a sliding time window.
    """
    
    def __init__(self, window_seconds: float = 60.0):
        """
        Initialize the throughput tracker.
        
        Args:
            window_seconds: Time window for throughput calculation
        """
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._samples: Dict[str, deque] = defaultdict(deque)
    
    def record(self, key: str, tokens: int, requests: int = 1) -> None:
        """
        Record a throughput sample.
        
        Args:
            key: Identifier (e.g., model name)
            tokens: Number of tokens processed
            requests: Number of requests (default 1)
        """
        sample = ThroughputSample(
            timestamp=time.time(),
            tokens=tokens,
            requests=requests,
        )
        
        with self._lock:
            self._samples[key].append(sample)
            self._cleanup(key)
    
    def get_stats(self, key: str) -> Optional[ThroughputStats]:
        """
        Get throughput statistics for a key.
        
        Args:
            key: Identifier to get stats for
            
        Returns:
            ThroughputStats if data exists, None otherwise
        """
        with self._lock:
            if key not in self._samples:
                return None
            
            self._cleanup(key)
            samples = list(self._samples[key])
        
        if not samples:
            return None
        
        total_tokens = sum(s.tokens for s in samples)
        total_requests = sum(s.requests for s in samples)
        
        if len(samples) > 1:
            time_span = samples[-1].timestamp - samples[0].timestamp
        else:
            time_span = self._window_seconds
        
        time_span = max(time_span, 0.001)
        
        return ThroughputStats(
            tokens_per_second=total_tokens / time_span,
            requests_per_second=total_requests / time_span,
            total_tokens=total_tokens,
            total_requests=total_requests,
            window_seconds=time_span,
        )
    
    def get_all_stats(self) -> Dict[str, ThroughputStats]:
        """Get statistics for all tracked keys."""
        with self._lock:
            keys = list(self._samples.keys())
        
        return {key: self.get_stats(key) for key in keys if self.get_stats(key)}
    
    def get_current_rate(self, key: str) -> float:
        """Get current tokens per second rate."""
        stats = self.get_stats(key)
        return stats.tokens_per_second if stats else 0.0
    
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


_tracker = ThroughputTracker()


def get_throughput_tracker() -> ThroughputTracker:
    """Get the global throughput tracker instance."""
    return _tracker


def record_throughput(key: str, tokens: int) -> None:
    """Convenience function to record throughput."""
    _tracker.record(key, tokens)
