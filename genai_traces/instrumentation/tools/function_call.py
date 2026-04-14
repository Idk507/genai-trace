"""
Function call and tool use tracing.

Traces OpenAI tool_calls and Anthropic tool_use.
"""

import contextlib
import json
from typing import Any, Dict, List, Optional, Generator
from datetime import datetime

from ...core.tracer import get_tracer
from ...core.types import SpanType


class FunctionCallTracer:
    """
    Traces function/tool calls from LLM responses.
    """
    
    def __init__(self, tracer: Any = None):
        self._tracer = tracer
    
    @property
    def tracer(self):
        if self._tracer is None:
            return get_tracer()
        return self._tracer
    
    def trace_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        parent_span: Any = None,
    ) -> List[Any]:
        """
        Create spans for tool calls from an LLM response.
        
        Args:
            tool_calls: List of tool call dicts from LLM response
            parent_span: Optional parent span
            
        Returns:
            List of created spans
        """
        spans = []
        
        for tool_call in tool_calls:
            span = self._create_tool_span(tool_call, parent_span)
            spans.append(span)
        
        return spans
    
    def _create_tool_span(
        self,
        tool_call: Dict[str, Any],
        parent_span: Any = None,
    ) -> Any:
        """Create a span for a single tool call."""
        if hasattr(tool_call, 'function'):
            name = tool_call.function.name
            arguments = tool_call.function.arguments
            tool_id = tool_call.id
        else:
            name = tool_call.get("name") or tool_call.get("function", {}).get("name", "unknown")
            arguments = tool_call.get("arguments") or tool_call.get("function", {}).get("arguments", "{}")
            tool_id = tool_call.get("id", "")
        
        span = self.tracer.start_span(
            f"tool.{name}",
            SpanType.FUNCTION_CALL,
        )
        
        span.set_attribute("tool.name", name)
        span.set_attribute("tool.id", tool_id)
        
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        
        span.set_attribute("tool.arguments", arguments)
        
        return span
    
    def record_tool_result(
        self,
        span: Any,
        result: Any,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Record the result of a tool call.
        
        Args:
            span: The tool span
            result: Tool execution result
            error: Exception if tool failed
        """
        if error:
            span.record_exception(error)
            span.set_attribute("tool.error", str(error))
        else:
            if isinstance(result, (dict, list)):
                span.set_attribute("tool.result", json.dumps(result)[:1000])
            else:
                span.set_attribute("tool.result", str(result)[:1000])
            
            from ...core.types import SpanStatus
            span.status = SpanStatus.OK
        
        span.end_time = datetime.utcnow()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000


@contextlib.contextmanager
def trace_function_call(
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    tool_id: str = "",
) -> Generator[Any, None, None]:
    """
    Context manager for tracing function/tool calls.
    
    Usage:
        with trace_function_call("get_weather", {"city": "NYC"}) as span:
            result = get_weather("NYC")
            span.set_attribute("tool.result", result)
    
    Args:
        name: Function/tool name
        arguments: Function arguments
        tool_id: Optional tool call ID
        
    Yields:
        Span for the function call
    """
    tracer = get_tracer()
    
    with tracer.start_as_current_span(f"tool.{name}", SpanType.FUNCTION_CALL) as span:
        span.set_attribute("tool.name", name)
        span.set_attribute("tool.id", tool_id)
        
        if arguments:
            span.set_attribute("tool.arguments", arguments)
        
        try:
            yield span
        except Exception as e:
            span.set_attribute("tool.error", str(e))
            raise


def extract_tool_calls_openai(response: Any) -> List[Dict[str, Any]]:
    """
    Extract tool calls from OpenAI response.
    
    Args:
        response: OpenAI chat completion response
        
    Returns:
        List of tool call dicts
    """
    tool_calls = []
    
    if hasattr(response, 'choices') and response.choices:
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })
    
    return tool_calls


def extract_tool_use_anthropic(response: Any) -> List[Dict[str, Any]]:
    """
    Extract tool use from Anthropic response.
    
    Args:
        response: Anthropic message response
        
    Returns:
        List of tool use dicts
    """
    tool_uses = []
    
    if hasattr(response, 'content'):
        for block in response.content:
            if hasattr(block, 'type') and block.type == 'tool_use':
                tool_uses.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
    
    return tool_uses
