"""
Pydantic-based configuration validation for GenAI-Traces.
"""

from typing import List, Optional, Any, Dict
from dataclasses import dataclass


def validate_config(config: Any) -> List[str]:
    """
    Validate a TracerConfig object and return a list of validation errors.
    Returns empty list if config is valid.
    """
    errors = []
    
    if not config.service_name:
        errors.append("service_name is required")
    elif len(config.service_name) > 255:
        errors.append("service_name must be <= 255 characters")
    
    if config.sample_rate < 0.0 or config.sample_rate > 1.0:
        errors.append("sample_rate must be between 0.0 and 1.0")
    
    if config.batch_size < 1:
        errors.append("batch_size must be >= 1")
    
    if config.flush_interval_ms < 100:
        errors.append("flush_interval_ms must be >= 100")
    
    if config.max_queue_size < 100:
        errors.append("max_queue_size must be >= 100")
    
    valid_environments = ["development", "staging", "production", "test"]
    if config.environment not in valid_environments:
        errors.append(f"environment must be one of: {valid_environments}")
    
    valid_actions = ["block", "flag", "log"]
    if config.guardrail_action not in valid_actions:
        errors.append(f"guardrail_action must be one of: {valid_actions}")
    
    return errors


def validate_span_attributes(attributes: Dict[str, Any]) -> List[str]:
    """Validate span attributes for type correctness."""
    errors = []
    
    type_requirements = {
        "llm.prompt.tokens": (int, float),
        "llm.completion.tokens": (int, float),
        "llm.total_tokens": (int, float),
        "cost.total_usd": (int, float),
        "llm.duration_ms": (int, float),
        "eval.relevance": (int, float),
        "eval.quality": (int, float),
        "feedback.score": (int, float),
    }
    
    for key, expected_types in type_requirements.items():
        if key in attributes:
            value = attributes[key]
            if value is not None and not isinstance(value, expected_types):
                errors.append(f"Attribute '{key}' must be numeric, got {type(value).__name__}")
    
    string_attributes = [
        "llm.model.name", "llm.provider", "llm.prompt", "llm.completion",
        "service.name", "service.environment"
    ]
    for key in string_attributes:
        if key in attributes:
            value = attributes[key]
            if value is not None and not isinstance(value, str):
                errors.append(f"Attribute '{key}' must be string, got {type(value).__name__}")
    
    return errors


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    
    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(valid=True, errors=[], warnings=[])
    
    @classmethod
    def failure(cls, errors: List[str], warnings: List[str] = None) -> "ValidationResult":
        return cls(valid=False, errors=errors, warnings=warnings or [])


def validate_exporter_config(exporter_type: str, config: Dict[str, Any]) -> ValidationResult:
    """Validate exporter-specific configuration."""
    errors = []
    warnings = []
    
    if exporter_type == "json":
        if "output_path" not in config:
            errors.append("JSON exporter requires 'output_path'")
        rotation = config.get("rotation", "daily")
        if rotation not in ["daily", "hourly", "size"]:
            errors.append("rotation must be 'daily', 'hourly', or 'size'")
        if rotation == "size" and "max_size_mb" not in config:
            warnings.append("size-based rotation without max_size_mb will use default 100MB")
    
    elif exporter_type == "postgres":
        if "connection_string" not in config and "dsn" not in config:
            errors.append("Postgres exporter requires 'connection_string' or 'dsn'")
    
    elif exporter_type == "webhook":
        if "url" not in config:
            errors.append("Webhook exporter requires 'url'")
    
    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult(valid=True, errors=[], warnings=warnings)


def validate_evaluator_config(evaluator_type: str, config: Dict[str, Any]) -> ValidationResult:
    """Validate evaluator-specific configuration."""
    errors = []
    
    if evaluator_type == "relevance":
        if config.get("use_llm_judge") and not config.get("judge_model"):
            errors.append("LLM judge evaluator requires 'judge_model'")
    
    elif evaluator_type == "toxicity":
        valid_methods = ["keyword", "detoxify", "perspective"]
        method = config.get("method", "keyword")
        if method not in valid_methods:
            errors.append(f"toxicity method must be one of: {valid_methods}")
    
    if errors:
        return ValidationResult.failure(errors)
    return ValidationResult.success()
