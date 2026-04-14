"""
AWS Bedrock instrumentation for GenAI-Traces.

Traces boto3 bedrock-runtime invoke_model calls.
"""

import functools
import json
import time
from typing import Any, Optional

from ...core.tracer import get_tracer
from ...core.types import SpanType

_original_invoke_model = None
_instrumented = False


def instrument_bedrock() -> None:
    """
    Instrument AWS Bedrock for automatic tracing.
    
    Usage:
        from genai_traces.instrumentation.llm.bedrock import instrument_bedrock
        instrument_bedrock()
        
        # All subsequent Bedrock calls are automatically traced
        client = boto3.client('bedrock-runtime')
        response = client.invoke_model(...)
    """
    global _original_invoke_model, _instrumented
    
    if _instrumented:
        return
    
    try:
        import botocore.client
    except ImportError:
        return
    
    original_make_api_call = botocore.client.BaseClient._make_api_call
    
    @functools.wraps(original_make_api_call)
    def traced_make_api_call(self, operation_name, api_params):
        if operation_name == "InvokeModel" and self._service_model.service_name == "bedrock-runtime":
            return _trace_bedrock_call(original_make_api_call, self, operation_name, api_params)
        return original_make_api_call(self, operation_name, api_params)
    
    botocore.client.BaseClient._make_api_call = traced_make_api_call
    _original_invoke_model = original_make_api_call
    _instrumented = True


def uninstrument_bedrock() -> None:
    """Remove Bedrock instrumentation."""
    global _original_invoke_model, _instrumented
    
    if not _instrumented:
        return
    
    try:
        import botocore.client
        if _original_invoke_model:
            botocore.client.BaseClient._make_api_call = _original_invoke_model
    except ImportError:
        pass
    
    _instrumented = False


def _trace_bedrock_call(original_fn, self, operation_name, api_params):
    """Wrap a Bedrock invoke_model call with tracing."""
    tracer = get_tracer()
    model_id = api_params.get("modelId", "unknown")
    
    with tracer.start_as_current_span(f"bedrock.{model_id}", SpanType.LLM) as span:
        span.set_attribute("llm.provider", "bedrock")
        span.set_attribute("llm.model.name", model_id)
        
        body = api_params.get("body", "{}")
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        
        try:
            body_json = json.loads(body)
            
            if "anthropic" in model_id.lower():
                _extract_anthropic_attributes(span, body_json)
            elif "amazon" in model_id.lower():
                _extract_titan_attributes(span, body_json)
            elif "meta" in model_id.lower():
                _extract_llama_attributes(span, body_json)
            elif "cohere" in model_id.lower():
                _extract_cohere_attributes(span, body_json)
        except json.JSONDecodeError:
            pass
        
        start_time = time.perf_counter()
        
        try:
            response = original_fn(self, operation_name, api_params)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("llm.duration_ms", duration_ms)
            
            response_body = response.get("body")
            if response_body:
                try:
                    response_json = json.loads(response_body.read())
                    response["body"] = response_json
                    _extract_response_attributes(span, response_json, model_id)
                except Exception:
                    pass
            
            return response
            
        except Exception as e:
            span.record_exception(e)
            raise


def _extract_anthropic_attributes(span, body: dict) -> None:
    """Extract attributes from Anthropic-format request."""
    if "prompt" in body:
        span.set_attribute("llm.prompt", body["prompt"][:1000])
    if "messages" in body:
        span.set_attribute("llm.messages", body["messages"])
    if "max_tokens_to_sample" in body:
        span.set_attribute("llm.request.max_tokens", body["max_tokens_to_sample"])
    if "temperature" in body:
        span.set_attribute("llm.request.temperature", body["temperature"])


def _extract_titan_attributes(span, body: dict) -> None:
    """Extract attributes from Amazon Titan request."""
    if "inputText" in body:
        span.set_attribute("llm.prompt", body["inputText"][:1000])
    config = body.get("textGenerationConfig", {})
    if "maxTokenCount" in config:
        span.set_attribute("llm.request.max_tokens", config["maxTokenCount"])
    if "temperature" in config:
        span.set_attribute("llm.request.temperature", config["temperature"])


def _extract_llama_attributes(span, body: dict) -> None:
    """Extract attributes from Meta Llama request."""
    if "prompt" in body:
        span.set_attribute("llm.prompt", body["prompt"][:1000])
    if "max_gen_len" in body:
        span.set_attribute("llm.request.max_tokens", body["max_gen_len"])
    if "temperature" in body:
        span.set_attribute("llm.request.temperature", body["temperature"])


def _extract_cohere_attributes(span, body: dict) -> None:
    """Extract attributes from Cohere request."""
    if "prompt" in body:
        span.set_attribute("llm.prompt", body["prompt"][:1000])
    if "max_tokens" in body:
        span.set_attribute("llm.request.max_tokens", body["max_tokens"])
    if "temperature" in body:
        span.set_attribute("llm.request.temperature", body["temperature"])


def _extract_response_attributes(span, response: dict, model_id: str) -> None:
    """Extract attributes from response."""
    if "anthropic" in model_id.lower():
        if "completion" in response:
            span.set_attribute("llm.completion", response["completion"][:1000])
    elif "amazon" in model_id.lower():
        results = response.get("results", [{}])
        if results:
            span.set_attribute("llm.completion", results[0].get("outputText", "")[:1000])
    elif "meta" in model_id.lower():
        if "generation" in response:
            span.set_attribute("llm.completion", response["generation"][:1000])
    elif "cohere" in model_id.lower():
        generations = response.get("generations", [{}])
        if generations:
            span.set_attribute("llm.completion", generations[0].get("text", "")[:1000])
