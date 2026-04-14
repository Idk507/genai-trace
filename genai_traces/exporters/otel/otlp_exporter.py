"""
OTLP exporter for GenAI-Traces.

Exports traces via gRPC or HTTP OTLP protocol.
"""

from typing import Any, Dict, List, Optional
from enum import Enum

from ..base import BaseExporter
from .mapper import SpanMapper


class OTLPProtocol(Enum):
    """OTLP protocol options."""
    GRPC = "grpc"
    HTTP = "http"


class OTLPExporter(BaseExporter):
    """
    OTLP exporter for OpenTelemetry backends.
    
    Usage:
        exporter = OTLPExporter(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
        )
        exporter.export_span(span)
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        protocol: OTLPProtocol = OTLPProtocol.GRPC,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        insecure: bool = True,
    ):
        self._endpoint = endpoint
        self._protocol = protocol
        self._headers = headers or {}
        self._timeout = timeout
        self._insecure = insecure
        self._exporter = None
        self._mapper = SpanMapper()
    
    def _get_exporter(self):
        """Get or create the OTLP exporter."""
        if self._exporter is not None:
            return self._exporter
        
        try:
            if self._protocol == OTLPProtocol.GRPC:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                self._exporter = OTLPSpanExporter(
                    endpoint=self._endpoint,
                    headers=self._headers,
                    timeout=self._timeout,
                    insecure=self._insecure,
                )
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                self._exporter = OTLPSpanExporter(
                    endpoint=self._endpoint,
                    headers=self._headers,
                    timeout=self._timeout,
                )
        except ImportError:
            raise ImportError(
                "opentelemetry-exporter-otlp is required for OTLP export"
            )
        
        return self._exporter
    
    def export_span(self, span: Any) -> None:
        """Export a span via OTLP."""
        otel_span = self._mapper.to_otel_span(span)
        if otel_span:
            exporter = self._get_exporter()
            exporter.export([otel_span])
    
    def export_batch(self, spans: List[Any]) -> None:
        """Export multiple spans."""
        otel_spans = [
            self._mapper.to_otel_span(span)
            for span in spans
        ]
        otel_spans = [s for s in otel_spans if s is not None]
        
        if otel_spans:
            exporter = self._get_exporter()
            exporter.export(otel_spans)
    
    def flush(self) -> None:
        """Flush pending exports."""
        if self._exporter:
            self._exporter.force_flush()
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        if self._exporter:
            self._exporter.shutdown()
