"""
Azure Blob Storage exporter for GenAI-Traces.
"""

import json
import gzip
from typing import Any, Dict, List, Optional
from datetime import datetime
from io import BytesIO

from ..base import BaseExporter


class AzureBlobExporter(BaseExporter):
    """
    Azure Blob Storage exporter for traces.
    
    Usage:
        exporter = AzureBlobExporter(
            connection_string="DefaultEndpointsProtocol=https;...",
            container="traces",
            prefix="traces/",
        )
        exporter.export_span(span)
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
        container: str = "traces",
        prefix: str = "traces/",
        compress: bool = True,
        batch_size: int = 100,
    ):
        self._connection_string = connection_string
        self._account_url = account_url
        self._container_name = container
        self._prefix = prefix
        self._compress = compress
        self._batch_size = batch_size
        self._buffer: List[Dict] = []
        self._client = None
        self._container = None
    
    def _get_container(self):
        """Get or create container client."""
        if self._container is None:
            try:
                from azure.storage.blob import BlobServiceClient
                
                if self._connection_string:
                    self._client = BlobServiceClient.from_connection_string(
                        self._connection_string
                    )
                elif self._account_url:
                    from azure.identity import DefaultAzureCredential
                    self._client = BlobServiceClient(
                        self._account_url,
                        credential=DefaultAzureCredential(),
                    )
                else:
                    raise ValueError(
                        "Either connection_string or account_url is required"
                    )
                
                self._container = self._client.get_container_client(
                    self._container_name
                )
            except ImportError:
                raise ImportError(
                    "azure-storage-blob is required for Azure Blob export"
                )
        return self._container
    
    def export_span(self, span: Any) -> None:
        """Export a span to Azure Blob Storage."""
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
        """Flush buffer to Azure Blob Storage."""
        if not self._buffer:
            return
        
        container = self._get_container()
        
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
        
        blob_client = container.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True)
        
        self._buffer.clear()
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self.flush()
