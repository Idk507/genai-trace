"""
Base instrumentation class for GenAI-Traces.

All LLM provider and framework instrumentations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Dict, List
import functools


class BaseInstrumentation(ABC):
    """
    Abstract base class for all instrumentation implementations.
    
    Provides a common interface for instrumenting LLM providers and frameworks.
    """
    
    def __init__(self, tracer: Any = None):
        """
        Initialize the instrumentation.
        
        Args:
            tracer: Optional tracer instance. If not provided, will use global tracer.
        """
        self._tracer = tracer
        self._original_methods: Dict[str, Callable] = {}
        self._instrumented = False
    
    @property
    def tracer(self):
        """Get the tracer instance."""
        if self._tracer is None:
            from ..core.tracer import get_tracer
            return get_tracer()
        return self._tracer
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider being instrumented."""
        pass
    
    @abstractmethod
    def instrument(self) -> None:
        """
        Apply instrumentation to the target library.
        
        This method should monkey-patch or wrap the relevant methods
        to capture trace data.
        """
        pass
    
    @abstractmethod
    def uninstrument(self) -> None:
        """
        Remove instrumentation and restore original methods.
        """
        pass
    
    def is_instrumented(self) -> bool:
        """Check if instrumentation is currently active."""
        return self._instrumented
    
    def _store_original(self, key: str, method: Callable) -> None:
        """Store an original method for later restoration."""
        if key not in self._original_methods:
            self._original_methods[key] = method
    
    def _restore_original(self, key: str) -> Optional[Callable]:
        """Restore and return an original method."""
        return self._original_methods.pop(key, None)
    
    def _wrap_method(
        self,
        original: Callable,
        span_name: str,
        extract_attributes: Optional[Callable] = None,
        extract_response: Optional[Callable] = None,
    ) -> Callable:
        """
        Create a wrapped version of a method that adds tracing.
        
        Args:
            original: The original method to wrap
            span_name: Name for the span
            extract_attributes: Function to extract attributes from args/kwargs
            extract_response: Function to extract attributes from response
        """
        from ..core.types import SpanType
        
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            attributes = {}
            if extract_attributes:
                try:
                    attributes = extract_attributes(*args, **kwargs)
                except Exception:
                    pass
            
            with self.tracer.start_as_current_span(
                span_name,
                SpanType.LLM,
                attributes=attributes
            ) as span:
                span.set_attribute("llm.provider", self.provider_name)
                
                try:
                    response = original(*args, **kwargs)
                    
                    if extract_response:
                        try:
                            response_attrs = extract_response(response)
                            for key, value in response_attrs.items():
                                span.set_attribute(key, value)
                        except Exception:
                            pass
                    
                    span.record_response(response)
                    return response
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        return wrapper
    
    def _wrap_async_method(
        self,
        original: Callable,
        span_name: str,
        extract_attributes: Optional[Callable] = None,
        extract_response: Optional[Callable] = None,
    ) -> Callable:
        """
        Create a wrapped version of an async method that adds tracing.
        """
        from ..core.types import SpanType
        
        @functools.wraps(original)
        async def wrapper(*args, **kwargs):
            attributes = {}
            if extract_attributes:
                try:
                    attributes = extract_attributes(*args, **kwargs)
                except Exception:
                    pass
            
            async with self.tracer.start_as_current_span_async(
                span_name,
                SpanType.LLM,
                attributes=attributes
            ) as span:
                span.set_attribute("llm.provider", self.provider_name)
                
                try:
                    response = await original(*args, **kwargs)
                    
                    if extract_response:
                        try:
                            response_attrs = extract_response(response)
                            for key, value in response_attrs.items():
                                span.set_attribute(key, value)
                        except Exception:
                            pass
                    
                    span.record_response(response)
                    return response
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        return wrapper


class InstrumentationRegistry:
    """Registry for managing multiple instrumentations."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._instrumentations: Dict[str, BaseInstrumentation] = {}
        return cls._instance
    
    def register(self, name: str, instrumentation: BaseInstrumentation) -> None:
        """Register an instrumentation."""
        self._instrumentations[name] = instrumentation
    
    def get(self, name: str) -> Optional[BaseInstrumentation]:
        """Get an instrumentation by name."""
        return self._instrumentations.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered instrumentation names."""
        return list(self._instrumentations.keys())
    
    def instrument_all(self) -> None:
        """Apply all registered instrumentations."""
        for inst in self._instrumentations.values():
            if not inst.is_instrumented():
                inst.instrument()
    
    def uninstrument_all(self) -> None:
        """Remove all instrumentations."""
        for inst in self._instrumentations.values():
            if inst.is_instrumented():
                inst.uninstrument()


_registry = InstrumentationRegistry()


def get_instrumentation_registry() -> InstrumentationRegistry:
    """Get the global instrumentation registry."""
    return _registry
