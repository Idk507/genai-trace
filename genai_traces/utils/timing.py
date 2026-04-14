"""
Timing utilities for performance measurement.
"""

import time
from typing import Optional
from contextlib import contextmanager


class Timer:
    """
    High-precision timer for measuring elapsed time.
    
    Usage:
        timer = Timer()
        timer.start()
        # ... do work ...
        elapsed_ms = timer.stop()
        
        # Or as context manager:
        with Timer() as t:
            # ... do work ...
        print(f"Elapsed: {t.elapsed_ms}ms")
    """
    
    def __init__(self):
        self._start_ns: Optional[int] = None
        self._end_ns: Optional[int] = None
    
    def start(self) -> "Timer":
        """Start the timer."""
        self._start_ns = time.perf_counter_ns()
        self._end_ns = None
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds."""
        self._end_ns = time.perf_counter_ns()
        return self.elapsed_ms
    
    @property
    def elapsed_ns(self) -> int:
        """Get elapsed time in nanoseconds."""
        if self._start_ns is None:
            return 0
        end = self._end_ns or time.perf_counter_ns()
        return end - self._start_ns
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ns / 1_000_000
    
    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_ns / 1_000_000_000
    
    def __enter__(self) -> "Timer":
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()


@contextmanager
def timed_block(name: str = "block"):
    """
    Context manager for timing a block of code.
    
    Usage:
        with timed_block("my_operation") as timer:
            # ... do work ...
        print(f"{timer.name}: {timer.elapsed_ms}ms")
    """
    timer = Timer()
    timer.name = name
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds since epoch."""
    return int(time.time() * 1000)


def get_timestamp_ns() -> int:
    """Get current timestamp in nanoseconds since epoch."""
    return time.time_ns()
