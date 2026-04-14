"""
Google Cloud Storage exporter for GenAI-Traces.
"""

import json
import gzip
from typing import Any, Dict, List, Optional
from datetime import datetime
from io import BytesIO

from ..base import BaseExporter


class GCSExporter(BaseExporter):
    """
    Google Cloud Storage exporter for traces.
    
    Usage:
        exporter = GCSExporter(
            bucket="my-traces-bucket",
            prefix="traces/",
        )
        exporter.export_span(span)
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "traces/",
        compress: bool = True,
        batch_size: int = 100,
        credentials_path: Optional[str] = None,
    ):
        self._bucket_name = bucket
        self._prefix = prefix
        self._compress = compress
        self._batch_size = batch_size
        self._credentials_path = credentials_path
        self._buffer: List[Dict] = []
        self._client = None
        self._bucket = None
    
    def _get_bucket(self):
        """Get or create GCS bucket reference."""
        if self._bucket is None:
            try:
                from google.cloud import storage
                
                if self._credentials_path:
                    self._client = storage.Client.from_service_account_json(
                        self._credentials_path
                    )
                else:
                    self._client = storage.Client()
                
                self._bucket = self._client.bucket(self._bucket_name)
            except ImportError:
                raise ImportError("google-cloud-storage is required for GCS export")
        return self._bucket
    
    def export_span(self, span: Any) -> None:
        """Export a span to GCS."""
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
        """Flush buffer to GCS."""
        if not self._buffer:
            return
        
        bucket = self._get_bucket()
        
        timestamp = datetime.utcnow().strftime("%Y/%m/%d/%H%M%S")
        blob_name = f"{self._prefix}{timestamp}.jsonl"
        
        if self._compress:
            blob_name += ".gz"
        
        content = "\n".join(json.dumps(span) for span in self._buffer)
        
        if self._compress:
            buffer = BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                gz.write(content.encode("utf-8"))
            data = buffer.getvalue()
        else:
            data = content.encode("utf-8")
        
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type="application/x-ndjson")
        
        self._buffer.clear()
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self.flush()
