"""
Example custom exporter plugin for GenAI-Traces.

Shows how to create a custom exporter plugin.
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from genai_traces.exporters.base import BaseExporter


class CustomExporter(BaseExporter):
    """
    Example custom exporter that writes to a custom format.
    
    Usage:
        from genai_traces.plugins.examples.custom_exporter import CustomExporter
        from genai_traces.plugins import get_plugin_registry
        
        exporter = CustomExporter(output_path="./custom_traces.log")
        get_plugin_registry().register("exporter", "custom", exporter)
    """
    
    name = "custom_exporter"
    version = "1.0.0"
    
    def __init__(
        self,
        output_path: str = "./traces.custom",
        include_metadata: bool = True,
        format_type: str = "detailed",
    ):
        self._output_path = output_path
        self._include_metadata = include_metadata
        self._format_type = format_type
        self._buffer: List[Dict] = []
        self._batch_size = 10
    
    def export_span(self, span: Any) -> None:
        """
        Export a span.
        
        Args:
            span: The span to export
        """
        span_dict = span.to_dict() if hasattr(span, "to_dict") else span
        formatted = self._format_span(span_dict)
        self._buffer.append(formatted)
        
        if len(self._buffer) >= self._batch_size:
            self.flush()
    
    def _format_span(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """Format a span for export."""
        if self._format_type == "minimal":
            return {
                "id": span.get("span_id"),
                "name": span.get("name"),
                "duration": span.get("duration_ms"),
            }
        
        elif self._format_type == "detailed":
            result = {
                "span": {
                    "id": span.get("span_id"),
                    "trace_id": span.get("trace_id"),
                    "name": span.get("name"),
                    "type": span.get("span_type"),
                    "status": span.get("status"),
                    "duration_ms": span.get("duration_ms"),
                },
                "timing": {
                    "start": span.get("start_time"),
                    "end": span.get("end_time"),
                },
                "attributes": span.get("attributes", {}),
            }
            
            if self._include_metadata:
                result["metadata"] = {
                    "exported_at": datetime.utcnow().isoformat(),
                    "exporter": self.name,
                    "version": self.version,
                }
            
            return result
        
        else:
            return span
    
    def flush(self) -> None:
        """Flush buffered spans to file."""
        if not self._buffer:
            return
        
        with open(self._output_path, "a") as f:
            for span in self._buffer:
                f.write(json.dumps(span) + "\n")
        
        self._buffer.clear()
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self.flush()


class WebhookNotifier(BaseExporter):
    """
    Example exporter that sends notifications for specific events.
    
    Usage:
        notifier = WebhookNotifier(
            webhook_url="https://hooks.example.com/notify",
            notify_on_error=True,
        )
    """
    
    name = "webhook_notifier"
    version = "1.0.0"
    
    def __init__(
        self,
        webhook_url: str,
        notify_on_error: bool = True,
        notify_on_slow: bool = True,
        slow_threshold_ms: float = 5000.0,
    ):
        self._webhook_url = webhook_url
        self._notify_on_error = notify_on_error
        self._notify_on_slow = notify_on_slow
        self._slow_threshold_ms = slow_threshold_ms
    
    def export_span(self, span: Any) -> None:
        """Export a span (send notification if criteria met)."""
        span_dict = span.to_dict() if hasattr(span, "to_dict") else span
        
        should_notify = False
        reason = ""
        
        if self._notify_on_error and span_dict.get("status") == "ERROR":
            should_notify = True
            reason = "error"
        
        if self._notify_on_slow:
            duration = span_dict.get("duration_ms", 0)
            if duration > self._slow_threshold_ms:
                should_notify = True
                reason = "slow"
        
        if should_notify:
            self._send_notification(span_dict, reason)
    
    def _send_notification(self, span: Dict, reason: str) -> None:
        """Send a notification webhook."""
        try:
            import requests
            
            payload = {
                "event": "span_alert",
                "reason": reason,
                "span_id": span.get("span_id"),
                "trace_id": span.get("trace_id"),
                "name": span.get("name"),
                "duration_ms": span.get("duration_ms"),
                "status": span.get("status"),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            requests.post(
                self._webhook_url,
                json=payload,
                timeout=5,
            )
        except Exception:
            pass
    
    def flush(self) -> None:
        """Flush (no-op for notifier)."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown (no-op for notifier)."""
        pass


def register_plugin():
    """Register this plugin with the registry."""
    from genai_traces.plugins import get_plugin_registry
    
    registry = get_plugin_registry()
    
    exporter = CustomExporter()
    registry.register("exporter", "custom", exporter)
    
    return exporter
