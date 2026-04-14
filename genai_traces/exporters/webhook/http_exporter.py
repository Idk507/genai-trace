"""
HTTP/Webhook exporter for GenAI-Traces.
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..base import BaseExporter


class HTTPExporter(BaseExporter):
    """
    HTTP exporter for sending traces to a webhook endpoint.
    
    Usage:
        exporter = HTTPExporter(
            endpoint="https://api.example.com/traces",
            headers={"Authorization": "Bearer token"},
        )
        exporter.export_span(span)
    """
    
    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        batch_size: int = 100,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        self._endpoint = endpoint
        self._headers = headers or {}
        self._timeout = timeout
        self._batch_size = batch_size
        self._retry_count = retry_count
        self._retry_delay = retry_delay
        self._buffer: List[Dict] = []
        self._session = None
    
    def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
                self._session.headers.update(self._headers)
                self._session.headers["Content-Type"] = "application/json"
            except ImportError:
                raise ImportError("requests is required for HTTP export")
        return self._session
    
    def export_span(self, span: Any) -> None:
        """Export a span via HTTP."""
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
        """Flush buffer to HTTP endpoint."""
        if not self._buffer:
            return
        
        session = self._get_session()
        
        payload = {
            "spans": self._buffer,
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(self._buffer),
        }
        
        import time
        for attempt in range(self._retry_count):
            try:
                response = session.post(
                    self._endpoint,
                    json=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                self._buffer.clear()
                return
            except Exception as e:
                if attempt < self._retry_count - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    raise
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self.flush()
        if self._session:
            self._session.close()


class WebhookExporter(HTTPExporter):
    """Alias for HTTPExporter."""
    pass
