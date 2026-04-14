"""
CSV file exporter with functionality-based organization.

Exports traces to CSV files organized by span type (llm, agent, tool, etc.)
"""

import csv
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from threading import Thread, Lock
from typing import TYPE_CHECKING, Optional, Dict, List, Any, Set
from collections import deque

from ..base import BaseExporter

if TYPE_CHECKING:
    from ...core.span import Span


@dataclass
class CSVConfig:
    """Configuration for CSV exporter."""
    output_dir: str = "./outputs/csv"
    rotation: str = "daily"  # daily | hourly | none
    separate_by_type: bool = True  # Create separate files per span_type
    separate_by_status: bool = False  # Create separate files for ok/error
    include_attributes: bool = True  # Flatten attributes into columns
    max_attribute_columns: int = 50  # Limit attribute columns
    flush_interval_seconds: float = 1.0
    
    @classmethod
    def from_env(cls) -> "CSVConfig":
        """Create config from environment variables."""
        return cls(
            output_dir=os.getenv("GENAI_CSV_OUTPUT_DIR", "./outputs/csv"),
            rotation=os.getenv("GENAI_CSV_ROTATION", "daily"),
            separate_by_type=os.getenv("GENAI_CSV_SEPARATE_BY_TYPE", "true").lower() == "true",
            separate_by_status=os.getenv("GENAI_CSV_SEPARATE_BY_STATUS", "false").lower() == "true",
            include_attributes=os.getenv("GENAI_CSV_INCLUDE_ATTRIBUTES", "true").lower() == "true",
            max_attribute_columns=int(os.getenv("GENAI_CSV_MAX_ATTR_COLUMNS", "50")),
            flush_interval_seconds=float(os.getenv("GENAI_CSV_FLUSH_INTERVAL", "1.0")),
        )


class CSVExporter(BaseExporter):
    """
    Exports spans to CSV files organized by functionality.
    
    Features:
    - Separate CSV files per span type (llm, agent, tool, retrieval, etc.)
    - Optional separation by status (ok/error)
    - Automatic header management
    - File rotation (daily/hourly)
    - Thread-safe with background writer
    
    Usage:
        config = CSVConfig(output_dir="./traces/csv", separate_by_type=True)
        exporter = CSVExporter(config)
        tracer = init_tracer("my-app", exporters=[exporter])
    """
    
    # Base columns that are always present
    BASE_COLUMNS = [
        "trace_id", "span_id", "parent_span_id", "root_span_id",
        "name", "span_type", "start_time", "end_time", "duration_ms",
        "status", "status_message", "prompt_name", "prompt_version",
        "experiment_id", "variant_id", "injection_detected", "injection_type"
    ]
    
    def __init__(self, config: Optional[CSVConfig] = None):
        """
        Initialize the CSV exporter.
        
        Args:
            config: CSVConfig instance or None to use defaults/env vars
        """
        self.config = config or CSVConfig.from_env()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track known columns per file type
        self._columns: Dict[str, List[str]] = defaultdict(lambda: list(self.BASE_COLUMNS))
        self._known_attrs: Dict[str, Set[str]] = defaultdict(set)
        
        # Thread-safe queue
        self._queue: deque = deque()
        self._lock = Lock()
        self._running = True
        
        # Track which files have headers written
        self._headers_written: Set[str] = set()
        
        # Start background writer
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
    
    def export_span(self, span: "Span") -> None:
        """Queue a span for export."""
        with self._lock:
            self._queue.append(span)
    
    def _writer_loop(self):
        """Background thread that writes spans to CSV."""
        import time
        
        while self._running:
            batch = []
            with self._lock:
                while self._queue:
                    batch.append(self._queue.popleft())
            
            if batch:
                self._write_batch(batch)
            
            time.sleep(self.config.flush_interval_seconds)
        
        # Final flush
        batch = []
        with self._lock:
            while self._queue:
                batch.append(self._queue.popleft())
        if batch:
            self._write_batch(batch)
    
    def _write_batch(self, spans: list):
        """Write a batch of spans to appropriate CSV files."""
        # Group spans by their target file
        grouped: Dict[str, List["Span"]] = defaultdict(list)
        
        for span in spans:
            file_key = self._get_file_key(span)
            grouped[file_key].append(span)
        
        # Write each group
        for file_key, file_spans in grouped.items():
            self._write_to_file(file_key, file_spans)
    
    def _get_file_key(self, span: "Span") -> str:
        """Get the file key for a span based on config."""
        parts = []
        
        if self.config.separate_by_type:
            parts.append(span.span_type.value)
        
        if self.config.separate_by_status:
            parts.append(span.status.value)
        
        if self.config.rotation == "daily":
            parts.append(datetime.utcnow().strftime("%Y-%m-%d"))
        elif self.config.rotation == "hourly":
            parts.append(datetime.utcnow().strftime("%Y-%m-%d_%H"))
        
        return "_".join(parts) if parts else "traces"
    
    def _get_filepath(self, file_key: str) -> Path:
        """Get the file path for a file key."""
        return self.output_dir / f"{file_key}.csv"
    
    def _write_to_file(self, file_key: str, spans: List["Span"]):
        """Write spans to a specific CSV file."""
        filepath = self._get_filepath(file_key)
        
        # Collect all attribute keys from spans
        if self.config.include_attributes:
            for span in spans:
                for attr_key in span.attributes.keys():
                    if len(self._known_attrs[file_key]) < self.config.max_attribute_columns:
                        if attr_key not in self._known_attrs[file_key]:
                            self._known_attrs[file_key].add(attr_key)
                            self._columns[file_key].append(f"attr.{attr_key}")
        
        columns = self._columns[file_key]
        
        # Check if we need to write headers
        write_header = file_key not in self._headers_written and not filepath.exists()
        
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            
            if write_header:
                writer.writeheader()
                self._headers_written.add(file_key)
            
            for span in spans:
                row = self._span_to_row(span, columns)
                writer.writerow(row)
    
    def _span_to_row(self, span: "Span", columns: List[str]) -> Dict[str, Any]:
        """Convert a span to a CSV row dict."""
        row = {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id or "",
            "root_span_id": span.root_span_id or "",
            "name": span.name,
            "span_type": span.span_type.value,
            "start_time": span.start_time.isoformat() if span.start_time else "",
            "end_time": span.end_time.isoformat() if span.end_time else "",
            "duration_ms": span.duration_ms or "",
            "status": span.status.value,
            "status_message": span.status_message or "",
            "prompt_name": span.prompt_name or "",
            "prompt_version": span.prompt_version or "",
            "experiment_id": span.experiment_id or "",
            "variant_id": span.variant_id or "",
            "injection_detected": span.injection_detected,
            "injection_type": span.injection_type or "",
        }
        
        # Add attribute columns
        if self.config.include_attributes:
            for col in columns:
                if col.startswith("attr."):
                    attr_key = col[5:]  # Remove "attr." prefix
                    value = span.attributes.get(attr_key, "")
                    # Handle complex values
                    if isinstance(value, (dict, list)):
                        value = str(value)
                    row[col] = value
        
        return row
    
    async def flush(self) -> None:
        """Flush all pending spans."""
        import asyncio
        import time
        
        deadline = time.time() + 5.0
        while self._queue and time.time() < deadline:
            await asyncio.sleep(0.05)
    
    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._running = False
        await self.flush()
        
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
    
    def get_exported_files(self) -> List[Path]:
        """Get list of all exported CSV files."""
        return list(self.output_dir.glob("*.csv"))
    
    def get_file_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about exported files."""
        stats = {}
        for filepath in self.get_exported_files():
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                stats[filepath.name] = {
                    "path": str(filepath),
                    "rows": len(rows) - 1 if rows else 0,  # Exclude header
                    "columns": len(rows[0]) if rows else 0,
                    "size_bytes": filepath.stat().st_size,
                }
        return stats
