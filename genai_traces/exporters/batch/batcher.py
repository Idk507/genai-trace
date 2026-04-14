"""
Batch exporter with backpressure support.
"""

import threading
import time
from typing import List, Any, Callable, Optional
from queue import Queue, Full, Empty

from ..base import BaseExporter
from .buffer import CircularBuffer


class BatchExporter(BaseExporter):
    """
    Exporter that batches spans before sending to reduce overhead.
    
    Features:
    - Configurable batch size and flush interval
    - Backpressure handling
    - Background thread for async export
    """
    
    def __init__(
        self,
        export_fn: Callable[[List[Any]], bool],
        batch_size: int = 100,
        flush_interval_ms: float = 5000,
        max_queue_size: int = 10000,
        max_export_attempts: int = 3,
    ):
        """
        Initialize the batch exporter.
        
        Args:
            export_fn: Function to export a batch of spans
            batch_size: Maximum spans per batch
            flush_interval_ms: Max time to wait before flushing
            max_queue_size: Maximum queue size (backpressure)
            max_export_attempts: Retry attempts for failed exports
        """
        self._export_fn = export_fn
        self._batch_size = batch_size
        self._flush_interval = flush_interval_ms / 1000
        self._max_queue_size = max_queue_size
        self._max_attempts = max_export_attempts
        
        self._buffer = CircularBuffer(max_queue_size)
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._export_thread: Optional[threading.Thread] = None
        
        self._exported_count = 0
        self._dropped_count = 0
        self._failed_count = 0
        
        self._start_export_thread()
    
    def export_span(self, span: Any) -> None:
        """
        Add a span to the export queue.
        
        Non-blocking. Drops span if queue is full.
        """
        if self._shutdown.is_set():
            return
        
        success = self._buffer.push(span)
        if not success:
            self._dropped_count += 1
    
    def export_batch(self, spans: List[Any]) -> None:
        """Add multiple spans to the queue."""
        for span in spans:
            self.export_span(span)
    
    def flush(self) -> None:
        """Force flush all pending spans."""
        self._flush_buffer()
    
    def shutdown(self) -> None:
        """Shutdown the exporter and flush remaining spans."""
        self._shutdown.set()
        
        self._flush_buffer()
        
        if self._export_thread and self._export_thread.is_alive():
            self._export_thread.join(timeout=5.0)
    
    def get_stats(self) -> dict:
        """Get export statistics."""
        return {
            "exported": self._exported_count,
            "dropped": self._dropped_count,
            "failed": self._failed_count,
            "pending": len(self._buffer),
        }
    
    def _start_export_thread(self) -> None:
        """Start the background export thread."""
        self._export_thread = threading.Thread(
            target=self._export_loop,
            daemon=True,
        )
        self._export_thread.start()
    
    def _export_loop(self) -> None:
        """Background export loop."""
        last_flush = time.time()
        
        while not self._shutdown.is_set():
            current_time = time.time()
            
            should_flush = (
                len(self._buffer) >= self._batch_size or
                current_time - last_flush >= self._flush_interval
            )
            
            if should_flush and len(self._buffer) > 0:
                self._flush_buffer()
                last_flush = current_time
            
            time.sleep(0.1)
    
    def _flush_buffer(self) -> None:
        """Flush the buffer to the export function."""
        batch = []
        
        while len(batch) < self._batch_size:
            span = self._buffer.pop()
            if span is None:
                break
            batch.append(span)
        
        if not batch:
            return
        
        for attempt in range(self._max_attempts):
            try:
                success = self._export_fn(batch)
                if success:
                    self._exported_count += len(batch)
                    return
            except Exception:
                pass
            
            if attempt < self._max_attempts - 1:
                time.sleep(0.1 * (attempt + 1))
        
        self._failed_count += len(batch)


class AsyncBatchExporter:
    """
    Async version of batch exporter.
    """
    
    def __init__(
        self,
        export_fn: Callable[[List[Any]], Any],
        batch_size: int = 100,
        flush_interval_ms: float = 5000,
        max_queue_size: int = 10000,
    ):
        import asyncio
        
        self._export_fn = export_fn
        self._batch_size = batch_size
        self._flush_interval = flush_interval_ms / 1000
        self._max_queue_size = max_queue_size
        
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._shutdown = False
        self._task: Optional[asyncio.Task] = None
        
        self._exported_count = 0
        self._dropped_count = 0
    
    async def start(self) -> None:
        """Start the async export loop."""
        import asyncio
        self._task = asyncio.create_task(self._export_loop())
    
    async def stop(self) -> None:
        """Stop the exporter."""
        self._shutdown = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        await self._flush()
    
    async def export_span(self, span: Any) -> None:
        """Add a span to the queue."""
        try:
            self._queue.put_nowait(span)
        except asyncio.QueueFull:
            self._dropped_count += 1
    
    async def _export_loop(self) -> None:
        """Async export loop."""
        import asyncio
        
        while not self._shutdown:
            await asyncio.sleep(self._flush_interval)
            await self._flush()
    
    async def _flush(self) -> None:
        """Flush pending spans."""
        batch = []
        
        while len(batch) < self._batch_size:
            try:
                span = self._queue.get_nowait()
                batch.append(span)
            except asyncio.QueueEmpty:
                break
        
        if batch:
            try:
                await self._export_fn(batch)
                self._exported_count += len(batch)
            except Exception:
                pass
