"""
Resource usage monitoring for GenAI-Traces.

Tracks CPU and memory usage via psutil.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""
    
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    
    process_cpu_percent: float
    process_memory_mb: float
    process_threads: int
    
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": {
                "cpu_percent": round(self.cpu_percent, 1),
                "memory_percent": round(self.memory_percent, 1),
                "memory_used_mb": round(self.memory_used_mb, 1),
                "memory_available_mb": round(self.memory_available_mb, 1),
            },
            "process": {
                "cpu_percent": round(self.process_cpu_percent, 1),
                "memory_mb": round(self.process_memory_mb, 1),
                "threads": self.process_threads,
            },
            "timestamp": self.timestamp,
        }


def get_resource_usage() -> ResourceUsage:
    """
    Get current resource usage.
    
    Returns:
        ResourceUsage object with current metrics
    """
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        process = psutil.Process()
        process_cpu = process.cpu_percent(interval=0.1)
        process_memory = process.memory_info().rss / (1024 * 1024)
        process_threads = process.num_threads()
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            process_cpu_percent=process_cpu,
            process_memory_mb=process_memory,
            process_threads=process_threads,
            timestamp=time.time(),
        )
        
    except ImportError:
        return ResourceUsage(
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_used_mb=0.0,
            memory_available_mb=0.0,
            process_cpu_percent=0.0,
            process_memory_mb=0.0,
            process_threads=1,
            timestamp=time.time(),
        )


class ResourceMonitor:
    """
    Monitors resource usage over time.
    """
    
    def __init__(self, sample_interval_seconds: float = 1.0):
        """
        Initialize the resource monitor.
        
        Args:
            sample_interval_seconds: Interval between samples
        """
        self._interval = sample_interval_seconds
        self._samples: list = []
        self._max_samples = 1000
        self._running = False
        self._thread: Optional[Any] = None
    
    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return
        
        import threading
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                usage = get_resource_usage()
                self._samples.append(usage)
                
                if len(self._samples) > self._max_samples:
                    self._samples = self._samples[-self._max_samples:]
                
            except Exception:
                pass
            
            time.sleep(self._interval)
    
    def get_current(self) -> ResourceUsage:
        """Get current resource usage."""
        return get_resource_usage()
    
    def get_average(self, last_n: int = 60) -> Optional[ResourceUsage]:
        """Get average resource usage over last N samples."""
        if not self._samples:
            return None
        
        samples = self._samples[-last_n:]
        n = len(samples)
        
        return ResourceUsage(
            cpu_percent=sum(s.cpu_percent for s in samples) / n,
            memory_percent=sum(s.memory_percent for s in samples) / n,
            memory_used_mb=sum(s.memory_used_mb for s in samples) / n,
            memory_available_mb=sum(s.memory_available_mb for s in samples) / n,
            process_cpu_percent=sum(s.process_cpu_percent for s in samples) / n,
            process_memory_mb=sum(s.process_memory_mb for s in samples) / n,
            process_threads=int(sum(s.process_threads for s in samples) / n),
            timestamp=time.time(),
        )
    
    def get_peak(self, last_n: int = 60) -> Optional[Dict[str, float]]:
        """Get peak resource usage over last N samples."""
        if not self._samples:
            return None
        
        samples = self._samples[-last_n:]
        
        return {
            "peak_cpu_percent": max(s.cpu_percent for s in samples),
            "peak_memory_percent": max(s.memory_percent for s in samples),
            "peak_process_cpu_percent": max(s.process_cpu_percent for s in samples),
            "peak_process_memory_mb": max(s.process_memory_mb for s in samples),
        }


_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = ResourceMonitor()
    return _monitor
