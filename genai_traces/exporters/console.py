"""
Console exporter for debugging and development.
"""

import json
from typing import TYPE_CHECKING

from .base import BaseExporter
from ..utils.serialization import span_to_jsonable

if TYPE_CHECKING:
    from ..core.span import Span


class ConsoleExporter(BaseExporter):
    """
    Exports spans to the console (stdout).
    
    Useful for debugging and development. Not recommended for production.
    
    Usage:
        tracer = init_tracer(
            service_name="my-app",
            exporters=[ConsoleExporter()],
        )
    """
    
    def __init__(
        self,
        pretty: bool = True,
        include_attributes: bool = True,
        color: bool = True,
    ):
        """
        Initialize the console exporter.
        
        Args:
            pretty: Whether to pretty-print JSON
            include_attributes: Whether to include all attributes
            color: Whether to use ANSI colors
        """
        self.pretty = pretty
        self.include_attributes = include_attributes
        self.color = color
    
    def export_span(self, span: "Span") -> None:
        """Export a span to the console."""
        data = span_to_jsonable(span)
        
        if not self.include_attributes:
            data.pop("attributes", None)
            data.pop("events", None)
        
        # Format output
        if self.pretty:
            output = json.dumps(data, indent=2, default=str)
        else:
            output = json.dumps(data, default=str)
        
        # Add color if enabled
        if self.color:
            status = span.status.value
            if status == "ok":
                color_code = "\033[92m"  # Green
            elif status == "error":
                color_code = "\033[91m"  # Red
            elif status == "blocked":
                color_code = "\033[93m"  # Yellow
            else:
                color_code = "\033[94m"  # Blue
            reset_code = "\033[0m"
            
            header = f"{color_code}[SPAN] {span.name} ({span.span_type.value}) - {status}{reset_code}"
            print(header)
            print(output)
            print()
        else:
            print(f"[SPAN] {span.name}")
            print(output)
            print()
    
    async def flush(self) -> None:
        """Console exporter has no buffering, so flush is a no-op."""
        pass
