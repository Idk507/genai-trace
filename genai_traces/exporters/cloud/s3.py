"""
AWS S3 exporter for GenAI-Traces.
"""

import json
import gzip
from typing import Any, Dict, List, Optional
from datetime import datetime
from io import BytesIO

from ..base import BaseExporter


class S3Exporter(BaseExporter):
    """
    AWS S3 exporter for traces.
    
    Usage:
        exporter = S3Exporter(
            bucket="my-traces-bucket",
            prefix="traces/",
            region="us-east-1",
        )
        exporter.export_span(span)
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "traces/",
        region: str = "us-east-1",
        compress: bool = True,
        batch_size: int = 100,
    ):
        self._bucket = bucket
        self._prefix = prefix
        self._region = region
        self._compress = compress
        self._batch_size = batch_size
        self._buffer: List[Dict] = []
        self._client = None
    
    def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("s3", region_name=self._region)
            except ImportError:
                raise ImportError("boto3 is required for S3 export")
        return self._client
    
    def export_span(self, span: Any) -> None:
        """Export a span to S3."""
        span_dict = span.to_dict() if hasattr(span, "to_dict") else span
        self._buffer.append(span_dict)
        
        if len(self._buffer) >= self._batch_size:
            self.flush()
    
    def export_batch(self, spans: List[Any]) -> None:
        """Export multiple spans."""
        for span in spans:
            span_dict = span.to_dict() if hasattr(span, "to_dict") else span
            self._buffer.append(span_dict)
        
        self.flush()
    
    def flush(self) -> None:
        """Flush buffer to S3."""
        if not self._buffer:
            return
        
        client = self._get_client()
        
        timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H%M%S")
        key = f"{self._prefix}{timestamp}.jsonl"
        
        if self._compress:
            key += ".gz"
        
        content = "\n".join(json.dumps(span) for span in self._buffer)
        
        if self._compress:
            buffer = BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                gz.write(content.encode("utf-8"))
            body = buffer.getvalue()
        else:
            body = content.encode("utf-8")
        
        client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType="application/x-ndjson",
        )
        
        self._buffer.clear()
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self.flush()
