"""
Base exporter interface for GenAI-Traces.
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.span import Span


class BaseExporter(ABC):
    """
    Abstract base class for all exporters.
    
    All exporters must implement this interface. Exporters are responsible
    for persisting or forwarding span data to external systems.
    
    Implementations should be non-blocking where possible - use internal
    queues and background threads/tasks for I/O operations.
    """
    
    @abstractmethod
    def export_span(self, span: "Span") -> None:
        """
        Export a single span.
        
        This method must be non-blocking. Implementations should queue
        the span internally and return immediately.
        
        Args:
            span: The span to export
        """
        pass
    
    def export_batch(self, spans: List["Span"]) -> None:
        """
        Export a batch of spans.
        
        Default implementation calls export_span for each span.
        Override for more efficient batch processing.
        
        Args:
            spans: List of spans to export
        """
        for span in spans:
            self.export_span(span)
    
    async def flush(self) -> None:
        """
        Flush all pending spans.
        
        Called on shutdown to ensure all spans are exported.
        Default implementation does nothing.
        """
        pass
    
    async def shutdown(self) -> None:
        """
        Shutdown the exporter.
        
        Called when the tracer is being shut down. Should flush
        pending spans and release resources.
        """
        await self.flush()
    
    def export_feedback(self, feedback) -> None:
        """
        Export feedback data.
        
        Optional method for exporters that support feedback.
        Default implementation does nothing.
        
        Args:
            feedback: FeedbackRecord to export
        """
        pass
