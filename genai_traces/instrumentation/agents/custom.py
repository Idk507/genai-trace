"""
Custom agent framework support for GenAI-Traces.

Provides generic tracing utilities for any agentic framework.
"""

import functools
import time
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, field

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


@dataclass
class AgentStep:
    """Represents a single step in an agent's execution."""
    step_name: str
    step_type: str = "generic"
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CustomAgentTracer:
    """
    Generic tracer for custom agent frameworks.
    
    Usage:
        tracer = CustomAgentTracer("my_agent")
        
        with tracer.trace_execution() as execution:
            with execution.trace_step("planning") as step:
                plan = create_plan(task)
                step.record_output(plan)
            
            with execution.trace_step("execution") as step:
                result = execute_plan(plan)
                step.record_output(result)
    """
    
    def __init__(self, agent_name: str, agent_type: str = "custom"):
        self._tracer = get_tracer()
        self._agent_name = agent_name
        self._agent_type = agent_type
    
    def trace_execution(self, task: Optional[str] = None):
        """Start tracing an agent execution."""
        return AgentExecutionContext(
            self._tracer,
            self._agent_name,
            self._agent_type,
            task,
        )
    
    def trace_step(self, step_name: str, step_type: str = "generic"):
        """Trace a single agent step (standalone, not within execution)."""
        return StandaloneStepContext(
            self._tracer,
            self._agent_name,
            step_name,
            step_type,
        )


class AgentExecutionContext:
    """Context manager for tracing a full agent execution."""
    
    def __init__(self, tracer, agent_name: str, agent_type: str, task: Optional[str]):
        self._tracer = tracer
        self._agent_name = agent_name
        self._agent_type = agent_type
        self._task = task
        self._span = None
        self._steps: List[AgentStep] = []
        self._start_time = None
    
    def __enter__(self):
        self._span = self._tracer.start_span(f"agent.{self._agent_name}", SpanType.AGENT)
        self._span.set_attribute("agent.name", self._agent_name)
        self._span.set_attribute("agent.type", self._agent_type)
        
        if self._task:
            self._span.set_attribute("agent.task", self._task[:500])
        
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute("agent.duration_ms", duration_ms)
            self._span.set_attribute("agent.step_count", len(self._steps))
            
            if exc_type:
                self._span.record_exception(exc_val)
            else:
                self._span.status = SpanStatus.OK
            
            self._span.end()
        return False
    
    def trace_step(self, step_name: str, step_type: str = "generic"):
        """Trace a step within this execution."""
        return AgentStepContext(
            self._tracer,
            self._agent_name,
            step_name,
            step_type,
            self._steps,
        )
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the execution span."""
        if self._span:
            self._span.set_attribute(f"agent.{key}", value)


class AgentStepContext:
    """Context manager for tracing a single agent step."""
    
    def __init__(
        self,
        tracer,
        agent_name: str,
        step_name: str,
        step_type: str,
        steps_list: List[AgentStep],
    ):
        self._tracer = tracer
        self._agent_name = agent_name
        self._step_name = step_name
        self._step_type = step_type
        self._steps_list = steps_list
        self._span = None
        self._step = AgentStep(step_name=step_name, step_type=step_type)
        self._start_time = None
    
    def __enter__(self):
        self._span = self._tracer.start_span(
            f"agent.step.{self._agent_name}.{self._step_name}",
            SpanType.AGENT
        )
        self._span.set_attribute("agent.step.name", self._step_name)
        self._span.set_attribute("agent.step.type", self._step_type)
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._step.duration_ms = duration_ms
            self._span.set_attribute("agent.step.duration_ms", duration_ms)
            
            if exc_type:
                self._span.record_exception(exc_val)
            else:
                self._span.status = SpanStatus.OK
            
            self._span.end()
        
        self._steps_list.append(self._step)
        return False
    
    def record_input(self, input_data: Any) -> None:
        """Record the input to this step."""
        self._step.input_data = input_data
        if self._span:
            self._span.set_attribute("agent.step.input", str(input_data)[:500])
    
    def record_output(self, output_data: Any) -> None:
        """Record the output from this step."""
        self._step.output_data = output_data
        if self._span:
            self._span.set_attribute("agent.step.output", str(output_data)[:500])
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata on this step."""
        self._step.metadata[key] = value
        if self._span:
            self._span.set_attribute(f"agent.step.metadata.{key}", value)


class StandaloneStepContext:
    """Context manager for a standalone step (not within an execution)."""
    
    def __init__(self, tracer, agent_name: str, step_name: str, step_type: str):
        self._tracer = tracer
        self._agent_name = agent_name
        self._step_name = step_name
        self._step_type = step_type
        self._span = None
        self._start_time = None
    
    def __enter__(self):
        self._span = self._tracer.start_span(
            f"agent.step.{self._agent_name}.{self._step_name}",
            SpanType.AGENT
        )
        self._span.set_attribute("agent.name", self._agent_name)
        self._span.set_attribute("agent.step.name", self._step_name)
        self._span.set_attribute("agent.step.type", self._step_type)
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute("agent.step.duration_ms", duration_ms)
            
            if exc_type:
                self._span.record_exception(exc_val)
            else:
                self._span.status = SpanStatus.OK
            
            self._span.end()
        return False
    
    def record_input(self, input_data: Any) -> None:
        """Record the input to this step."""
        if self._span:
            self._span.set_attribute("agent.step.input", str(input_data)[:500])
    
    def record_output(self, output_data: Any) -> None:
        """Record the output from this step."""
        if self._span:
            self._span.set_attribute("agent.step.output", str(output_data)[:500])


def trace_agent_step(
    step_name: Optional[str] = None,
    step_type: str = "generic",
    record_args: bool = True,
    record_result: bool = True,
) -> Callable:
    """
    Decorator to trace an agent step function.
    
    Usage:
        @trace_agent_step("planning", step_type="planning")
        def plan_task(task_description):
            return create_plan(task_description)
    """
    def decorator(func: Callable) -> Callable:
        name = step_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            with tracer.start_as_current_span(f"agent.step.{name}", SpanType.AGENT) as span:
                span.set_attribute("agent.step.name", name)
                span.set_attribute("agent.step.type", step_type)
                
                if record_args and args:
                    span.set_attribute("agent.step.input", str(args[0])[:500])
                
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("agent.step.duration_ms", duration_ms)
                    
                    if record_result and result is not None:
                        span.set_attribute("agent.step.output", str(result)[:500])
                    
                    span.status = SpanStatus.OK
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            async with tracer.start_as_current_span_async(f"agent.step.{name}", SpanType.AGENT) as span:
                span.set_attribute("agent.step.name", name)
                span.set_attribute("agent.step.type", step_type)
                
                if record_args and args:
                    span.set_attribute("agent.step.input", str(args[0])[:500])
                
                start_time = time.perf_counter()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("agent.step.duration_ms", duration_ms)
                    
                    if record_result and result is not None:
                        span.set_attribute("agent.step.output", str(result)[:500])
                    
                    span.status = SpanStatus.OK
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator
