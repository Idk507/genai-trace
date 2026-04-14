"""
Adaptive sampling for trace collection.
"""

import random
from typing import Optional


class AdaptiveSampler:
    """
    Intelligent sampling that prioritizes errors and slow requests.
    
    - Always samples errors (regardless of sample_rate)
    - Always samples slow requests (above threshold)
    - Samples normal requests at base_rate
    
    Usage:
        sampler = AdaptiveSampler(base_rate=0.1, slow_threshold_ms=5000)
        if sampler.should_sample("my_span", is_error=False, duration_ms=100):
            # Export the span
            pass
    """
    
    def __init__(
        self,
        base_rate: float = 0.1,
        error_rate: float = 1.0,
        slow_threshold_ms: float = 5000.0,
        slow_rate: float = 1.0,
    ):
        """
        Initialize the sampler.
        
        Args:
            base_rate: Sampling rate for normal requests (0.0-1.0)
            error_rate: Sampling rate for error requests (0.0-1.0)
            slow_threshold_ms: Threshold for slow requests in milliseconds
            slow_rate: Sampling rate for slow requests (0.0-1.0)
        """
        self.base_rate = max(0.0, min(1.0, base_rate))
        self.error_rate = max(0.0, min(1.0, error_rate))
        self.slow_threshold = slow_threshold_ms
        self.slow_rate = max(0.0, min(1.0, slow_rate))
    
    def should_sample(
        self,
        span_name: Optional[str] = None,
        is_error: bool = False,
        duration_ms: Optional[float] = None,
    ) -> bool:
        """
        Determine if a span should be sampled.
        
        Args:
            span_name: Name of the span (currently unused, for future pattern matching)
            is_error: Whether the span represents an error
            duration_ms: Duration of the span in milliseconds
            
        Returns:
            True if the span should be sampled, False otherwise
        """
        # Always sample errors at error_rate
        if is_error:
            return random.random() < self.error_rate
        
        # Always sample slow requests at slow_rate
        if duration_ms and duration_ms > self.slow_threshold:
            return random.random() < self.slow_rate
        
        # Sample normal requests at base_rate
        return random.random() < self.base_rate
    
    def update_rates(
        self,
        base_rate: Optional[float] = None,
        error_rate: Optional[float] = None,
        slow_rate: Optional[float] = None,
    ) -> None:
        """
        Update sampling rates dynamically.
        
        Args:
            base_rate: New base sampling rate
            error_rate: New error sampling rate
            slow_rate: New slow request sampling rate
        """
        if base_rate is not None:
            self.base_rate = max(0.0, min(1.0, base_rate))
        if error_rate is not None:
            self.error_rate = max(0.0, min(1.0, error_rate))
        if slow_rate is not None:
            self.slow_rate = max(0.0, min(1.0, slow_rate))


class RateLimitingSampler:
    """
    Sampler that limits the number of spans per time window.
    
    Useful for preventing trace explosion during high-traffic periods.
    """
    
    def __init__(
        self,
        max_spans_per_second: int = 100,
        burst_size: int = 200,
    ):
        """
        Initialize the rate-limiting sampler.
        
        Args:
            max_spans_per_second: Maximum spans to sample per second
            burst_size: Maximum burst size for handling traffic spikes
        """
        self.max_rate = max_spans_per_second
        self.burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = 0.0
    
    def should_sample(self) -> bool:
        """
        Determine if a span should be sampled based on rate limits.
        
        Returns:
            True if the span should be sampled, False otherwise
        """
        import time
        
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now
        
        # Replenish tokens
        self._tokens = min(
            self.burst_size,
            self._tokens + elapsed * self.max_rate
        )
        
        # Check if we have tokens available
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        
        return False
