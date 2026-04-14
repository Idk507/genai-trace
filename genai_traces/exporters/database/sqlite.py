"""
SQLite exporter for GenAI-Traces.

Lightweight exporter for development and testing.
"""

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..base import BaseExporter


class SQLiteExporter(BaseExporter):
    """
    SQLite exporter for traces (development/testing).
    
    Usage:
        exporter = SQLiteExporter("./traces.db")
        exporter.export_span(span)
        exporter.close()
    """
    
    def __init__(
        self,
        db_path: str = "./genai_traces.db",
        auto_create_tables: bool = True,
    ):
        self._db_path = db_path
        self._auto_create = auto_create_tables
        self._local = threading.local()
        
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        if auto_create_tables:
            self.create_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self._db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def close(self) -> None:
        """Close the connection."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection
    
    def export_span(self, span: Any) -> None:
        """Export a span to SQLite."""
        span_dict = span.to_dict() if hasattr(span, "to_dict") else span
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO spans 
            (trace_id, span_id, parent_span_id, name, span_type, status,
             start_time, end_time, duration_ms, attributes, events)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                span_dict.get("trace_id"),
                span_dict.get("span_id"),
                span_dict.get("parent_span_id"),
                span_dict.get("name"),
                span_dict.get("span_type"),
                span_dict.get("status"),
                span_dict.get("start_time"),
                span_dict.get("end_time"),
                span_dict.get("duration_ms"),
                json.dumps(span_dict.get("attributes", {})),
                json.dumps(span_dict.get("events", [])),
            ),
        )
        conn.commit()
    
    def export_batch(self, spans: List[Any]) -> None:
        """Export multiple spans in a batch."""
        records = []
        for span in spans:
            span_dict = span.to_dict() if hasattr(span, "to_dict") else span
            records.append((
                span_dict.get("trace_id"),
                span_dict.get("span_id"),
                span_dict.get("parent_span_id"),
                span_dict.get("name"),
                span_dict.get("span_type"),
                span_dict.get("status"),
                span_dict.get("start_time"),
                span_dict.get("end_time"),
                span_dict.get("duration_ms"),
                json.dumps(span_dict.get("attributes", {})),
                json.dumps(span_dict.get("events", [])),
            ))
        
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO spans 
            (trace_id, span_id, parent_span_id, name, span_type, status,
             start_time, end_time, duration_ms, attributes, events)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
    
    def create_tables(self) -> None:
        """Create the necessary tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                parent_span_id TEXT,
                name TEXT NOT NULL,
                span_type TEXT,
                status TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_ms REAL,
                attributes TEXT,
                events TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time)
        """)
        
        conn.commit()
    
    def query_spans(
        self,
        trace_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query spans from the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if trace_id:
            cursor.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time LIMIT ?",
                (trace_id, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM spans ORDER BY start_time DESC LIMIT ?",
                (limit,),
            )
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def flush(self) -> None:
        """Flush pending exports."""
        conn = self._get_connection()
        conn.commit()
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self.close()
