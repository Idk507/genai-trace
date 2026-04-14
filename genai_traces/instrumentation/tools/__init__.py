"""
Tool and function call instrumentation.
"""

from .function_call import trace_function_call, FunctionCallTracer

__all__ = ["trace_function_call", "FunctionCallTracer"]
