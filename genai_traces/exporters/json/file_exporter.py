"""
JSON file exporter with rotation support.
"""

import json
import asyncio
import gzip
from pathlib import Path
from datetime import datetime
from collections import deque
from threading import Thread, Lock
from typing import TYPE_CHECKING, Optional

from ..base import BaseExporter
from ...utils.serialization import span_to_jsonable

if TYPE_CHECKING:
    from ...core.span import Span


class JSONFileExporter(BaseExporter):
    """
    Writes spans as JSONL (one JSON object per line).
    
    Features:
    - File rotation (daily, hourly, or size-based)
    - Optional gzip compression
    - Thread-safe with background writer
    - Non-blocking export
    
    Usage:
        exporter = JSONFileExporter(
            output_dir="./traces",
            rotation="daily",
            compress=True,
        )
        tracer = init_tracer(
            service_name="my-app",
            exporters=[exporter],
        )
    """
    
    def __init__(
        self,
        output_dir: str = "./traces",
        rotation: str = "daily",  # daily | hourly | size
        max_size_mb: int = 100,
        compress: bool = False,
        flush_interval_seconds: float = 1.0,
    ):
        """
        Initialize the JSON file exporter.
        
        Args:
            output_dir: Directory to write trace files
            rotation: Rotation strategy (daily, hourly, size)
            max_size_mb: Maximum file size in MB (for size-based rotation)
            compress: Whether to gzip compress files
            flush_interval_seconds: How often to flush the buffer
        """
        self.output_dir = Path(output_dir)
        self.rotation = rotation
        self.max_size_mb = max_size_mb
        self.compress = compress
        self.flush_interval = flush_interval_seconds
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe queue
        self._queue: deque = deque()
        self._lock: Lock = Lock()
        self._running = True
        
        # Start background writer thread
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        
        # Track current file for size-based rotation
        self._current_file: Optional[Path] = None
        self._current_size: int = 0
    
    def export_span(self, span: "Span") -> None:
        """Queue a span for export."""
        with self._lock:
            self._queue.append(span)
    
    def _writer_loop(self):
        """Background thread that writes spans to disk."""
        import time
        
        while self._running:
            batch = []
            with self._lock:
                while self._queue:
                    batch.append(self._queue.popleft())
            
            if batch:
                self._write_batch(batch)
            
            time.sleep(self.flush_interval)
        
        # Final flush on shutdown
        batch = []
        with self._lock:
            while self._queue:
                batch.append(self._queue.popleft())
        if batch:
            self._write_batch(batch)
    
    def _write_batch(self, spans: list):
        """Write a batch of spans to the current file."""
        filepath = self._get_current_filepath()
        
        lines = []
        for span in spans:
            data = span_to_jsonable(span)
            lines.append(json.dumps(data, default=str))
        
        content = "\n".join(lines) + "\n"
        content_bytes = content.encode("utf-8")
        
        if self.compress:
            # Write to gzip file
            with gzip.open(filepath, "at", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(content)
        
        # Update size tracking for size-based rotation
        self._current_size += len(content_bytes)
    
    def _get_current_filepath(self) -> Path:
        """Get the current file path based on rotation strategy."""
        now = datetime.utcnow()
        
        if self.rotation == "daily":
            suffix = now.strftime("%Y-%m-%d")
        elif self.rotation == "hourly":
            suffix = now.strftime("%Y-%m-%d_%H")
        elif self.rotation == "size":
            # Check if we need to rotate
            if self._current_file and self._current_size >= self.max_size_mb * 1024 * 1024:
                self._current_file = None
                self._current_size = 0
            
            if self._current_file is None:
                suffix = now.strftime("%Y-%m-%d_%H%M%S")
                self._current_file = self._make_filepath(suffix)
                self._current_size = 0
            
            return self._current_file
        else:
            suffix = "current"
        
        return self._make_filepath(suffix)
    
    def _make_filepath(self, suffix: str) -> Path:
        """Create a file path with the given suffix."""
        ext = ".jsonl.gz" if self.compress else ".jsonl"
        return self.output_dir / f"traces_{suffix}{ext}"
    
    async def flush(self) -> None:
        """Flush all pending spans."""
        import time
        
        deadline = time.time() + 5.0
        while self._queue and time.time() < deadline:
            await asyncio.sleep(0.05)
    
    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._running = False
        await self.flush()
        
        # Wait for writer thread to finish
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
