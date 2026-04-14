"""
PostgreSQL exporter for GenAI-Traces.

Provides async PostgreSQL export using asyncpg.
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..base import BaseExporter


class PostgresExporter(BaseExporter):
    """
    Async PostgreSQL exporter for traces.
    
    Usage:
        exporter = PostgresExporter(
            host="localhost",
            port=5432,
            database="traces",
            user="postgres",
            password="secret",
        )
        await exporter.connect()
        await exporter.export_span(span)
        await exporter.close()
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "genai_traces",
        user: str = "postgres",
        password: str = "",
        schema: str = "public",
        pool_size: int = 10,
    ):
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._schema = schema
        self._pool_size = pool_size
        self._pool = None
    
    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password,
                min_size=1,
                max_size=self._pool_size,
            )
        except ImportError:
            raise ImportError("asyncpg is required for PostgreSQL export")
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
    
    def export_span(self, span: Any) -> None:
        """Sync export (not recommended for Postgres)."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.export_span_async(span))
    
    async def export_span_async(self, span: Any) -> None:
        """Export a span to PostgreSQL."""
        if not self._pool:
            await self.connect()
        
        span_dict = span.to_dict() if hasattr(span, "to_dict") else span
        
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._schema}.spans 
                (trace_id, span_id, parent_span_id, name, span_type, status, 
                 start_time, end_time, duration_ms, attributes, events)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
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
            )
    
    async def export_batch_async(self, spans: List[Any]) -> None:
        """Export multiple spans in a batch."""
        if not self._pool:
            await self.connect()
        
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
        
        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self._schema}.spans 
                (trace_id, span_id, parent_span_id, name, span_type, status,
                 start_time, end_time, duration_ms, attributes, events)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                records,
            )
    
    async def create_tables(self) -> None:
        """Create the necessary tables."""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._schema}.spans (
                    id SERIAL PRIMARY KEY,
                    trace_id VARCHAR(64) NOT NULL,
                    span_id VARCHAR(64) NOT NULL,
                    parent_span_id VARCHAR(64),
                    name VARCHAR(255) NOT NULL,
                    span_type VARCHAR(50),
                    status VARCHAR(20),
                    start_time TIMESTAMP WITH TIME ZONE,
                    end_time TIMESTAMP WITH TIME ZONE,
                    duration_ms FLOAT,
                    attributes JSONB,
                    events JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_spans_trace_id 
                ON {self._schema}.spans(trace_id);
                
                CREATE INDEX IF NOT EXISTS idx_spans_start_time 
                ON {self._schema}.spans(start_time);
            """)
    
    def flush(self) -> None:
        """Flush pending exports."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        import asyncio
        if self._pool:
            asyncio.get_event_loop().run_until_complete(self.close())
