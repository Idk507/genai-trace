"""
Streaming token accumulator for GenAI-Traces.

Accumulates tokens from streaming LLM responses for accurate counting.
"""

from typing import Any, Optional, Iterator, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class StreamingStats:
    """Statistics from a streaming response."""
    
    chunks_received: int = 0
    total_content: str = ""
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    start_time: float = field(default_factory=time.time)
    
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    
    @property
    def time_to_first_token_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000
    
    @property
    def total_duration_ms(self) -> Optional[float]:
        """Total streaming duration in milliseconds."""
        if self.last_token_time is None:
            return None
        return (self.last_token_time - self.start_time) * 1000
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Tokens per second throughput."""
        if self.completion_tokens is None or self.total_duration_ms is None:
            return None
        if self.total_duration_ms == 0:
            return None
        return self.completion_tokens / (self.total_duration_ms / 1000)


class StreamingAccumulator:
    """
    Accumulates content and statistics from streaming LLM responses.
    
    Usage:
        accumulator = StreamingAccumulator()
        
        for chunk in stream:
            content = accumulator.process_chunk(chunk)
            yield content
        
        stats = accumulator.finalize()
        span.set_attribute("llm.completion", stats.total_content)
        span.set_attribute("llm.ttft_ms", stats.time_to_first_token_ms)
    """
    
    def __init__(
        self,
        content_extractor: Optional[Callable[[Any], str]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        """
        Initialize the accumulator.
        
        Args:
            content_extractor: Function to extract content from a chunk
            token_counter: Function to count tokens in text
        """
        self._content_extractor = content_extractor or self._default_extractor
        self._token_counter = token_counter
        self._stats = StreamingStats()
        self._finalized = False
    
    def process_chunk(self, chunk: Any) -> str:
        """
        Process a streaming chunk and return the content.
        
        Args:
            chunk: A chunk from the streaming response
            
        Returns:
            The content extracted from the chunk
        """
        if self._finalized:
            raise RuntimeError("Accumulator has been finalized")
        
        content = self._content_extractor(chunk)
        
        if content:
            current_time = time.time()
            
            if self._stats.first_token_time is None:
                self._stats.first_token_time = current_time
            
            self._stats.last_token_time = current_time
            self._stats.total_content += content
        
        self._stats.chunks_received += 1
        
        if hasattr(chunk, 'usage') and chunk.usage:
            usage = chunk.usage
            if hasattr(usage, 'completion_tokens'):
                self._stats.completion_tokens = usage.completion_tokens
            if hasattr(usage, 'prompt_tokens'):
                self._stats.prompt_tokens = usage.prompt_tokens
        
        return content
    
    def finalize(self) -> StreamingStats:
        """
        Finalize the accumulator and return statistics.
        
        Returns:
            StreamingStats with accumulated data
        """
        if self._finalized:
            return self._stats
        
        self._finalized = True
        
        if self._stats.completion_tokens is None and self._token_counter:
            self._stats.completion_tokens = self._token_counter(
                self._stats.total_content
            )
        
        return self._stats
    
    @property
    def content(self) -> str:
        """Get the accumulated content."""
        return self._stats.total_content
    
    @property
    def stats(self) -> StreamingStats:
        """Get current statistics."""
        return self._stats
    
    def _default_extractor(self, chunk: Any) -> str:
        """Default content extractor for OpenAI/Anthropic format."""
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                return delta.content
        
        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
            return chunk.delta.text or ""
        
        if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'text'):
            return chunk.content_block.text or ""
        
        return ""


def wrap_streaming_response(
    stream: Iterator[Any],
    span: Any,
    content_extractor: Optional[Callable[[Any], str]] = None,
) -> Iterator[Any]:
    """
    Wrap a streaming response to accumulate statistics.
    
    Args:
        stream: The streaming iterator
        span: The span to update with statistics
        content_extractor: Optional custom content extractor
        
    Yields:
        Original chunks from the stream
    """
    accumulator = StreamingAccumulator(content_extractor=content_extractor)
    
    try:
        for chunk in stream:
            accumulator.process_chunk(chunk)
            yield chunk
    finally:
        stats = accumulator.finalize()
        
        span.set_attribute("llm.completion", stats.total_content)
        span.set_attribute("llm.streaming", True)
        
        if stats.time_to_first_token_ms is not None:
            span.set_attribute("llm.ttft_ms", stats.time_to_first_token_ms)
        
        if stats.tokens_per_second is not None:
            span.set_attribute("llm.tokens_per_second", stats.tokens_per_second)
        
        if stats.completion_tokens is not None:
            span.set_attribute("llm.completion.tokens", stats.completion_tokens)


async def wrap_async_streaming_response(
    stream: AsyncIterator[Any],
    span: Any,
    content_extractor: Optional[Callable[[Any], str]] = None,
) -> AsyncIterator[Any]:
    """
    Wrap an async streaming response to accumulate statistics.
    
    Args:
        stream: The async streaming iterator
        span: The span to update with statistics
        content_extractor: Optional custom content extractor
        
    Yields:
        Original chunks from the stream
    """
    accumulator = StreamingAccumulator(content_extractor=content_extractor)
    
    try:
        async for chunk in stream:
            accumulator.process_chunk(chunk)
            yield chunk
    finally:
        stats = accumulator.finalize()
        
        span.set_attribute("llm.completion", stats.total_content)
        span.set_attribute("llm.streaming", True)
        
        if stats.time_to_first_token_ms is not None:
            span.set_attribute("llm.ttft_ms", stats.time_to_first_token_ms)
        
        if stats.tokens_per_second is not None:
            span.set_attribute("llm.tokens_per_second", stats.tokens_per_second)
        
        if stats.completion_tokens is not None:
            span.set_attribute("llm.completion.tokens", stats.completion_tokens)
