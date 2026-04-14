"""
MySQL exporter for GenAI-Traces.
"""

import json
from typing import Any, Dict, List, Optional

from ..base import BaseExporter


class MySQLExporter(BaseExporter):
    """
    MySQL exporter for traces.
    
    Usage:
        exporter = MySQLExporter(
            host="localhost",
            port=3306,
            database="traces",
            user="root",
            password="secret",
        )
        exporter.connect()
        exporter.export_span(span)
        exporter.close()
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "genai_traces",
        user: str = "root",
        password: str = "",
        pool_size: int = 5,
    ):
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._pool_size = pool_size
        self._connection = None
    
    def connect(self) -> None:
        """Connect to MySQL."""
        try:
            import mysql.connector
            from mysql.connector import pooling
            
            self._pool = pooling.MySQLConnectionPool(
                pool_name="genai_traces_pool",
                pool_size=self._pool_size,
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password,
            )
        except ImportError:
            raise ImportError("mysql-connector-python is required for MySQL export")
    
    def close(self) -> None:
        """Close the connection."""
        pass
    
    def export_span(self, span: Any) -> None:
        """Export a span to MySQL."""
        if not hasattr(self, "_pool"):
            self.connect()
        
        span_dict = span.to_dict() if hasattr(span, "to_dict") else span
        
        conn = self._pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO spans 
                (trace_id, span_id, parent_span_id, name, span_type, status,
                 start_time, end_time, duration_ms, attributes, events)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        finally:
            cursor.close()
            conn.close()
    
    def export_batch(self, spans: List[Any]) -> None:
        """Export multiple spans in a batch."""
        if not hasattr(self, "_pool"):
            self.connect()
        
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
        
        conn = self._pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO spans 
                (trace_id, span_id, parent_span_id, name, span_type, status,
                 start_time, end_time, duration_ms, attributes, events)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                records,
            )
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def create_tables(self) -> None:
        """Create the necessary tables."""
        if not hasattr(self, "_pool"):
            self.connect()
        
        conn = self._pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    trace_id VARCHAR(64) NOT NULL,
                    span_id VARCHAR(64) NOT NULL,
                    parent_span_id VARCHAR(64),
                    name VARCHAR(255) NOT NULL,
                    span_type VARCHAR(50),
                    status VARCHAR(20),
                    start_time DATETIME(6),
                    end_time DATETIME(6),
                    duration_ms FLOAT,
                    attributes JSON,
                    events JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_trace_id (trace_id),
                    INDEX idx_start_time (start_time)
                )
            """)
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def flush(self) -> None:
        """Flush pending exports."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self.close()
