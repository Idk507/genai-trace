"""
LangChain integration for GenAI-Traces.

Provides a callback handler for automatic tracing of LangChain operations.
"""

import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


class LangChainCallbackHandler:
    """
    LangChain callback handler for GenAI-Traces.
    
    Usage:
        from genai_traces.instrumentation.frameworks.langchain import LangChainCallbackHandler
        
        handler = LangChainCallbackHandler()
        chain = LLMChain(..., callbacks=[handler])
        chain.run("Hello")
    """
    
    def __init__(self, tracer=None):
        self._tracer = tracer or get_tracer()
        self._spans: Dict[str, Any] = {}
        self._start_times: Dict[str, float] = {}
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        run_id_str = str(run_id)
        
        model_name = serialized.get("name", "unknown")
        span = self._tracer.start_span(f"langchain.llm.{model_name}", SpanType.LLM)
        
        span.set_attribute("llm.provider", "langchain")
        span.set_attribute("llm.model.name", model_name)
        span.set_attribute("llm.prompts", prompts)
        
        if tags:
            span.set_attribute("langchain.tags", tags)
        if metadata:
            span.set_attribute("langchain.metadata", metadata)
        
        self._spans[run_id_str] = span
        self._start_times[run_id_str] = time.perf_counter()
    
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends running."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        start_time = self._start_times.pop(run_id_str, None)
        
        if span:
            if start_time:
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
            
            if hasattr(response, "generations"):
                generations = response.generations
                if generations and generations[0]:
                    span.set_attribute("llm.completion", generations[0][0].text[:1000])
            
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                if token_usage:
                    span.set_attribute("llm.prompt.tokens", token_usage.get("prompt_tokens", 0))
                    span.set_attribute("llm.completion.tokens", token_usage.get("completion_tokens", 0))
                    span.set_attribute("llm.total_tokens", token_usage.get("total_tokens", 0))
            
            span.status = SpanStatus.OK
            span.end()
    
    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        self._start_times.pop(run_id_str, None)
        
        if span:
            span.record_exception(error)
            span.end()
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts running."""
        run_id_str = str(run_id)
        
        chain_name = serialized.get("name", "chain")
        span = self._tracer.start_span(f"langchain.chain.{chain_name}", SpanType.CHAIN)
        
        span.set_attribute("langchain.chain.name", chain_name)
        span.set_attribute("langchain.chain.inputs", inputs)
        
        if tags:
            span.set_attribute("langchain.tags", tags)
        
        self._spans[run_id_str] = span
        self._start_times[run_id_str] = time.perf_counter()
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain ends running."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        start_time = self._start_times.pop(run_id_str, None)
        
        if span:
            if start_time:
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("langchain.chain.duration_ms", duration_ms)
            
            span.set_attribute("langchain.chain.outputs", outputs)
            span.status = SpanStatus.OK
            span.end()
    
    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        self._start_times.pop(run_id_str, None)
        
        if span:
            span.record_exception(error)
            span.end()
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts running."""
        run_id_str = str(run_id)
        
        tool_name = serialized.get("name", "tool")
        span = self._tracer.start_span(f"langchain.tool.{tool_name}", SpanType.TOOL)
        
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.input", input_str[:1000])
        
        self._spans[run_id_str] = span
        self._start_times[run_id_str] = time.perf_counter()
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends running."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        start_time = self._start_times.pop(run_id_str, None)
        
        if span:
            if start_time:
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("tool.duration_ms", duration_ms)
            
            span.set_attribute("tool.output", str(output)[:1000])
            span.status = SpanStatus.OK
            span.end()
    
    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        self._start_times.pop(run_id_str, None)
        
        if span:
            span.record_exception(error)
            span.end()
    
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever starts running."""
        run_id_str = str(run_id)
        
        retriever_name = serialized.get("name", "retriever")
        span = self._tracer.start_span(f"langchain.retriever.{retriever_name}", SpanType.RETRIEVAL)
        
        span.set_attribute("retriever.name", retriever_name)
        span.set_attribute("retriever.query", query[:1000])
        
        self._spans[run_id_str] = span
        self._start_times[run_id_str] = time.perf_counter()
    
    def on_retriever_end(
        self,
        documents: List[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever ends running."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        start_time = self._start_times.pop(run_id_str, None)
        
        if span:
            if start_time:
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("retriever.duration_ms", duration_ms)
            
            span.set_attribute("retriever.document_count", len(documents))
            span.status = SpanStatus.OK
            span.end()
    
    def on_retriever_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever errors."""
        run_id_str = str(run_id)
        span = self._spans.pop(run_id_str, None)
        self._start_times.pop(run_id_str, None)
        
        if span:
            span.record_exception(error)
            span.end()


def instrument_langchain() -> LangChainCallbackHandler:
    """
    Create and return a LangChain callback handler for tracing.
    
    Returns:
        LangChainCallbackHandler instance
    """
    return LangChainCallbackHandler()
