"""
ReAct (Reasoning + Acting) agent tracing for GenAI-Traces.

Traces the thought-action-observation loop in ReAct-style agents.
"""

import functools
import time
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, field

from ...core.tracer import get_tracer
from ...core.types import SpanType, SpanStatus


@dataclass
class ReActStep:
    """Represents a single ReAct reasoning step."""
    step_number: int
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[Any] = None
    observation: Optional[str] = None
    duration_ms: Optional[float] = None


class ReActTracer:
    """
    Tracer for ReAct-style agent reasoning loops.
    
    Usage:
        tracer = ReActTracer()
        
        with tracer.trace_loop("my_agent", max_steps=10) as loop:
            for step_num in range(10):
                with loop.trace_step(step_num) as step:
                    step.record_thought("I need to search for...")
                    step.record_action("search", {"query": "..."})
                    result = search(...)
                    step.record_observation(result)
                    
                    if is_final_answer(result):
                        break
    """
    
    def __init__(self):
        self._tracer = get_tracer()
    
    def trace_loop(self, agent_name: str, max_steps: int = 10):
        """Start tracing a ReAct loop."""
        return ReActLoopContext(self._tracer, agent_name, max_steps)


class ReActLoopContext:
    """Context manager for a ReAct reasoning loop."""
    
    def __init__(self, tracer, agent_name: str, max_steps: int):
        self._tracer = tracer
        self._agent_name = agent_name
        self._max_steps = max_steps
        self._span = None
        self._steps: List[ReActStep] = []
        self._start_time = None
    
    def __enter__(self):
        self._span = self._tracer.start_span(f"react.loop.{self._agent_name}", SpanType.AGENT)
        self._span.set_attribute("agent.name", self._agent_name)
        self._span.set_attribute("agent.type", "react")
        self._span.set_attribute("agent.max_steps", self._max_steps)
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute("agent.duration_ms", duration_ms)
            self._span.set_attribute("agent.total_steps", len(self._steps))
            
            if exc_type:
                self._span.record_exception(exc_val)
            else:
                self._span.status = SpanStatus.OK
            
            self._span.end()
        return False
    
    def trace_step(self, step_number: int):
        """Start tracing a single ReAct step."""
        return ReActStepContext(self._tracer, self._agent_name, step_number, self._steps)


class ReActStepContext:
    """Context manager for a single ReAct step."""
    
    def __init__(self, tracer, agent_name: str, step_number: int, steps_list: List[ReActStep]):
        self._tracer = tracer
        self._agent_name = agent_name
        self._step_number = step_number
        self._steps_list = steps_list
        self._span = None
        self._step = ReActStep(step_number=step_number)
        self._start_time = None
    
    def __enter__(self):
        self._span = self._tracer.start_span(
            f"react.step.{self._agent_name}.{self._step_number}",
            SpanType.AGENT
        )
        self._span.set_attribute("agent.step.number", self._step_number)
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
    
    def record_thought(self, thought: str) -> None:
        """Record the agent's thought/reasoning."""
        self._step.thought = thought
        if self._span:
            self._span.set_attribute("agent.step.thought", thought[:1000])
    
    def record_action(self, action: str, action_input: Any = None) -> None:
        """Record the action the agent decided to take."""
        self._step.action = action
        self._step.action_input = action_input
        if self._span:
            self._span.set_attribute("agent.step.action", action)
            if action_input:
                self._span.set_attribute("agent.step.action_input", str(action_input)[:500])
    
    def record_observation(self, observation: str) -> None:
        """Record the observation from the action."""
        self._step.observation = observation
        if self._span:
            self._span.set_attribute("agent.step.observation", str(observation)[:1000])


def trace_react_step(
    step_name: str = "react_step",
    record_thought: bool = True,
    record_action: bool = True,
    record_observation: bool = True,
) -> Callable:
    """
    Decorator to trace a ReAct step function.
    
    Usage:
        @trace_react_step("reasoning")
        def reason_and_act(state):
            thought = generate_thought(state)
            action = decide_action(thought)
            observation = execute_action(action)
            return {"thought": thought, "action": action, "observation": observation}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            with tracer.start_as_current_span(f"react.{step_name}", SpanType.AGENT) as span:
                span.set_attribute("agent.type", "react")
                span.set_attribute("agent.step.name", step_name)
                
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("agent.step.duration_ms", duration_ms)
                    
                    if isinstance(result, dict):
                        if record_thought and "thought" in result:
                            span.set_attribute("agent.step.thought", str(result["thought"])[:1000])
                        if record_action and "action" in result:
                            span.set_attribute("agent.step.action", str(result["action"]))
                        if record_observation and "observation" in result:
                            span.set_attribute("agent.step.observation", str(result["observation"])[:1000])
                    
                    span.status = SpanStatus.OK
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        return wrapper
    
    return decorator
