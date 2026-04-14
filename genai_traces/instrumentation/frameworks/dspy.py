"""
DSPy integration for GenAI-Traces.

Provides tracing for DSPy modules and optimizers.
"""

import functools
import time
from typing import Any, Dict, Optional, Callable

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


_instrumented = False


def instrument_dspy() -> None:
    """
    Instrument DSPy for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.frameworks.dspy import instrument_dspy
        instrument_dspy()
        
        # All DSPy module calls are now traced
    """
    global _instrumented
    
    if _instrumented:
        return
    
    try:
        import dspy
    except ImportError:
        return
    
    original_forward = dspy.Module.forward if hasattr(dspy.Module, "forward") else None
    
    if original_forward:
        @functools.wraps(original_forward)
        def traced_forward(self, *args, **kwargs):
            return _trace_module_forward(original_forward, self, *args, **kwargs)
        
        dspy.Module.forward = traced_forward
    
    _instrumented = True


def _trace_module_forward(original_fn, self, *args, **kwargs):
    """Wrap a DSPy module forward call with tracing."""
    tracer = get_tracer()
    
    module_name = self.__class__.__name__
    
    with tracer.start_as_current_span(f"dspy.module.{module_name}", SpanType.CHAIN) as span:
        span.set_attribute("dspy.module.name", module_name)
        span.set_attribute("dspy.module.class", self.__class__.__module__ + "." + self.__class__.__name__)
        
        if args:
            span.set_attribute("dspy.module.input", str(args[0])[:1000])
        
        start_time = time.perf_counter()
        
        try:
            result = original_fn(self, *args, **kwargs)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("dspy.module.duration_ms", duration_ms)
            span.set_attribute("dspy.module.output", str(result)[:1000])
            span.status = SpanStatus.OK
            
            return result
            
        except Exception as e:
            span.record_exception(e)
            raise


def trace_dspy_module(name: Optional[str] = None) -> Callable:
    """
    Decorator to trace DSPy modules.
    
    Usage:
        @trace_dspy_module("my_module")
        class MyModule(dspy.Module):
            def forward(self, question):
                return self.predict(question=question)
    """
    def decorator(cls):
        original_forward = cls.forward
        module_name = name or cls.__name__
        
        @functools.wraps(original_forward)
        def traced_forward(self, *args, **kwargs):
            tracer = get_tracer()
            
            with tracer.start_as_current_span(f"dspy.module.{module_name}", SpanType.CHAIN) as span:
                span.set_attribute("dspy.module.name", module_name)
                
                start_time = time.perf_counter()
                
                try:
                    result = original_forward(self, *args, **kwargs)
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("dspy.module.duration_ms", duration_ms)
                    span.status = SpanStatus.OK
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        cls.forward = traced_forward
        return cls
    
    return decorator


def trace_optimizer(optimizer_name: str = "optimizer") -> Callable:
    """
    Decorator to trace DSPy optimizer runs.
    
    Usage:
        @trace_optimizer("bootstrap_few_shot")
        def optimize_module(module, trainset):
            optimizer = BootstrapFewShot()
            return optimizer.compile(module, trainset=trainset)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            with tracer.start_as_current_span(f"dspy.optimizer.{optimizer_name}", SpanType.CHAIN) as span:
                span.set_attribute("dspy.optimizer.name", optimizer_name)
                
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("dspy.optimizer.duration_ms", duration_ms)
                    span.status = SpanStatus.OK
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        return wrapper
    
    return decorator
