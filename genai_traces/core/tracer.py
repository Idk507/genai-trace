"""
Main Tracer class for GenAI-Traces.
"""

from __future__ import annotations
import contextlib
from datetime import datetime
from typing import Any, AsyncGenerator, Generator, List, Optional

from .span import Span
from .context import _current_span, inject_context_into_span
from .types import SpanType, SpanStatus
from ..utils.id_generator import generate_trace_id, generate_span_id
from ..config.settings import TracerConfig

# Global tracer instance
_global_tracer: Optional["Tracer"] = None


def init_tracer(
    service_name: str,
    environment: str = "development",
    exporters: Optional[List[Any]] = None,
    config: Optional[TracerConfig] = None,
    **kwargs
) -> "Tracer":
    """
    Initialize the global tracer. Call once at app startup.
    
    Args:
        service_name: Name of the service being traced
        environment: Environment (development, staging, production)
        exporters: List of exporter instances to use
        config: Optional TracerConfig instance
        **kwargs: Additional config options
        
    Returns:
        Initialized Tracer instance
        
    Example:
        tracer = init_tracer(
            service_name="my-ai-app",
            environment="production",
            exporters=[JSONFileExporter()],
        )
    """
    global _global_tracer
    
    if config is None:
        config = TracerConfig(
            service_name=service_name,
            environment=environment,
            **kwargs
        )
    else:
        config.service_name = service_name
        config.environment = environment
    
    _global_tracer = Tracer(config=config, exporters=exporters or [])
    return _global_tracer


def get_tracer() -> "Tracer":
    """
    Get the global tracer instance.
    
    Raises:
        RuntimeError: If tracer has not been initialized
        
    Returns:
        The global Tracer instance
    """
    if _global_tracer is None:
        raise RuntimeError("Tracer not initialized. Call init_tracer() first.")
    return _global_tracer


def _get_tracer_safe() -> Optional["Tracer"]:
    """Get the global tracer instance without raising an error."""
    return _global_tracer


class Tracer:
    """
    Main tracer class for creating and managing spans.
    
    The tracer is responsible for:
    - Creating new spans with proper parent-child relationships
    - Managing the span lifecycle (start, end, export)
    - Applying sampling decisions
    - Exporting spans to configured exporters
    """
    
    def __init__(
        self,
        config: TracerConfig,
        exporters: Optional[List[Any]] = None
    ):
        """
        Initialize the tracer.
        
        Args:
            config: TracerConfig instance
            exporters: List of exporter instances
        """
        self.config = config
        self.exporters = exporters or []
        self._sampler = None
        
        if config.enable_adaptive_sampling:
            from .sampling import AdaptiveSampler
            self._sampler = AdaptiveSampler(base_rate=config.sample_rate)
    
    @contextlib.contextmanager
    def start_as_current_span(
        self,
        name: str,
        span_type: SpanType = SpanType.LLM,
        attributes: Optional[dict] = None,
    ) -> Generator[Span, None, None]:
        """
        Create a new span and set it as the current span (sync version).
        
        Args:
            name: Name of the span
            span_type: Type of span (LLM, AGENT, TOOL, etc.)
            attributes: Initial attributes to set on the span
            
        Yields:
            The created Span instance
            
        Example:
            with tracer.start_as_current_span("llm_call", SpanType.LLM) as span:
                response = llm.generate(prompt)
                span.record_response(response)
        """
        span = self._create_span(name, span_type, attributes)
        token = _current_span.set(span)
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.status = SpanStatus.OK
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            self._finish_span(span)
            _current_span.reset(token)
    
    @contextlib.asynccontextmanager
    async def start_as_current_span_async(
        self,
        name: str,
        span_type: SpanType = SpanType.LLM,
        attributes: Optional[dict] = None,
    ) -> AsyncGenerator[Span, None]:
        """
        Create a new span and set it as the current span (async version).
        
        Args:
            name: Name of the span
            span_type: Type of span (LLM, AGENT, TOOL, etc.)
            attributes: Initial attributes to set on the span
            
        Yields:
            The created Span instance
            
        Example:
            async with tracer.start_as_current_span_async("llm_call") as span:
                response = await llm.agenerate(prompt)
                span.record_response(response)
        """
        span = self._create_span(name, span_type, attributes)
        token = _current_span.set(span)
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.status = SpanStatus.OK
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            self._finish_span(span)
            _current_span.reset(token)
    
    def get_current_span(self) -> Optional[Span]:
        """Get the currently active span, if any."""
        return _current_span.get(None)
    
    def start_span(
        self,
        name: str,
        span_type: SpanType = SpanType.LLM,
        attributes: Optional[dict] = None,
    ) -> Span:
        """
        Create a new span without setting it as current.
        
        Use this for manual span management. Remember to call end_span()
        when the span is complete.
        
        Args:
            name: Name of the span
            span_type: Type of span
            attributes: Initial attributes
            
        Returns:
            The created Span instance
        """
        return self._create_span(name, span_type, attributes)
    
    def end_span(self, span: Span) -> None:
        """
        End a span and export it.
        
        Args:
            span: The span to end
        """
        self._finish_span(span)
    
    def _create_span(
        self,
        name: str,
        span_type: SpanType,
        attributes: Optional[dict]
    ) -> Span:
        """Create a new span with proper parent-child linking."""
        parent = _current_span.get(None)
        trace_id = parent.trace_id if parent else generate_trace_id()
        
        span = Span(
            trace_id=trace_id,
            span_id=generate_span_id(),
            parent_span_id=parent.span_id if parent else None,
            root_span_id=parent.root_span_id if parent else None,
            name=name,
            span_type=span_type,
        )
        
        # Set root_span_id for the first span in a trace
        if span.root_span_id is None and parent is None:
            span.root_span_id = span.span_id
        elif parent and parent.root_span_id:
            span.root_span_id = parent.root_span_id
        elif parent:
            span.root_span_id = parent.span_id
        
        # Add service metadata
        span.set_attribute("service.name", self.config.service_name)
        span.set_attribute("service.environment", self.config.environment)
        if self.config.version:
            span.set_attribute("service.version", self.config.version)
        
        # Inject context (conversation, experiment, etc.)
        inject_context_into_span(span)
        
        # Add custom attributes
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        
        return span
    
    def _finish_span(self, span: Span) -> None:
        """Finish a span and export it."""
        span.end_time = datetime.utcnow()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        
        # Check sampling
        if self._sampler:
            is_error = span.status == SpanStatus.ERROR
            if not self._sampler.should_sample(span.name, is_error, span.duration_ms):
                return
        
        # Export to all registered exporters
        for exporter in self.exporters:
            try:
                exporter.export_span(span)
            except Exception:
                pass  # Never let exporter failure break user code
    
    def add_exporter(self, exporter: Any) -> None:
        """Add an exporter to the tracer."""
        self.exporters.append(exporter)
    
    def remove_exporter(self, exporter: Any) -> None:
        """Remove an exporter from the tracer."""
        if exporter in self.exporters:
            self.exporters.remove(exporter)
    
    async def flush(self) -> None:
        """Flush all exporters."""
        for exporter in self.exporters:
            if hasattr(exporter, "flush"):
                try:
                    result = exporter.flush()
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    pass
    
    async def shutdown(self) -> None:
        """Shutdown the tracer and all exporters."""
        await self.flush()
        for exporter in self.exporters:
            if hasattr(exporter, "shutdown"):
                try:
                    result = exporter.shutdown()
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    pass
