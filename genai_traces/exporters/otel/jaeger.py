"""
Jaeger exporter for GenAI-Traces.
"""

from typing import Any, Dict, List, Optional

from ..base import BaseExporter
from .mapper import SpanMapper


class JaegerExporter(BaseExporter):
    """
    Jaeger exporter for traces.
    
    Usage:
        exporter = JaegerExporter(
            agent_host="localhost",
            agent_port=6831,
        )
        exporter.export_span(span)
    """
    
    def __init__(
        self,
        agent_host: str = "localhost",
        agent_port: int = 6831,
        collector_endpoint: Optional[str] = None,
        service_name: str = "genai-traces",
    ):
        self._agent_host = agent_host
        self._agent_port = agent_port
        self._collector_endpoint = collector_endpoint
        self._service_name = service_name
        self._exporter = None
        self._mapper = SpanMapper()
    
    def _get_exporter(self):
        """Get or create the Jaeger exporter."""
        if self._exporter is not None:
            return self._exporter
        
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter as OTelJaeger
            
            if self._collector_endpoint:
                self._exporter = OTelJaeger(
                    collector_endpoint=self._collector_endpoint,
                )
            else:
                self._exporter = OTelJaeger(
                    agent_host_name=self._agent_host,
                    agent_port=self._agent_port,
                )
        except ImportError:
            raise ImportError(
                "opentelemetry-exporter-jaeger is required for Jaeger export"
            )
        
        return self._exporter
    
    def export_span(self, span: Any) -> None:
        """Export a span to Jaeger."""
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
