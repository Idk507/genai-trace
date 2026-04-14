"""
Configuration settings for GenAI-Traces.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os


@dataclass
class TracerConfig:
    """
    Configuration for the GenAI-Traces tracer.
    
    All settings can be overridden via environment variables with the
    GENAI_TRACES_ prefix (e.g., GENAI_TRACES_SERVICE_NAME).
    """
    
    # Core settings
    service_name: str = "genai-app"
    environment: str = "development"
    version: str = "0.0.0"
    
    # Sampling
    sample_rate: float = 1.0  # 0.0-1.0 (1.0 = 100%)
    enable_adaptive_sampling: bool = False
    slow_request_threshold_ms: float = 5000.0
    
    # Performance
    max_span_attributes: int = 100
    max_attribute_length: int = 4096
    enable_async_export: bool = True
    export_batch_size: int = 100
    export_interval_seconds: float = 2.0
    
    # Privacy
    enable_pii_detection: bool = False
    enable_prompt_capture: bool = True
    enable_prompt_hashing: bool = False
    pii_detection_sensitivity: str = "medium"  # low | medium | high
    redaction_strategy: str = "partial"  # full | partial | hash
    
    # Features
    enable_token_counting: bool = True
    enable_cost_tracking: bool = True
    enable_auto_evaluation: bool = False
    enable_conversation_tracking: bool = True
    enable_guardrails: bool = False
    enable_injection_detection: bool = False
    enable_anomaly_detection: bool = False
    enable_prompt_management: bool = False
    
    # Evaluation
    auto_evaluate_evaluators: List[str] = field(default_factory=list)
    evaluation_threshold: float = 0.7
    
    # Security
    injection_action: str = "log"  # block | flag | log
    use_ml_classifier: bool = False
    
    # Anomaly detection
    anomaly_window: int = 200
    anomaly_z_threshold: float = 3.0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Load settings from environment variables."""
        self._load_from_env()
    
    def _load_from_env(self):
        """Override settings from environment variables."""
        env_mappings = {
            "GENAI_TRACES_SERVICE_NAME": ("service_name", str),
            "GENAI_TRACES_ENVIRONMENT": ("environment", str),
            "GENAI_TRACES_VERSION": ("version", str),
            "GENAI_TRACES_SAMPLE_RATE": ("sample_rate", float),
            "GENAI_TRACES_ENABLE_PII_DETECTION": ("enable_pii_detection", self._parse_bool),
            "GENAI_TRACES_ENABLE_PROMPT_CAPTURE": ("enable_prompt_capture", self._parse_bool),
            "GENAI_TRACES_ENABLE_TOKEN_COUNTING": ("enable_token_counting", self._parse_bool),
            "GENAI_TRACES_ENABLE_COST_TRACKING": ("enable_cost_tracking", self._parse_bool),
            "GENAI_TRACES_ENABLE_GUARDRAILS": ("enable_guardrails", self._parse_bool),
            "GENAI_TRACES_ENABLE_INJECTION_DETECTION": ("enable_injection_detection", self._parse_bool),
            "GENAI_TRACES_INJECTION_ACTION": ("injection_action", str),
        }
        
        for env_var, (attr, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    setattr(self, attr, converter(value))
                except (ValueError, TypeError):
                    pass  # Keep default if conversion fails
    
    @staticmethod
    def _parse_bool(value: str) -> bool:
        """Parse a string to boolean."""
        return value.lower() in ("true", "1", "yes", "on")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "service_name": self.service_name,
            "environment": self.environment,
            "version": self.version,
            "sample_rate": self.sample_rate,
            "enable_adaptive_sampling": self.enable_adaptive_sampling,
            "slow_request_threshold_ms": self.slow_request_threshold_ms,
            "max_span_attributes": self.max_span_attributes,
            "max_attribute_length": self.max_attribute_length,
            "enable_async_export": self.enable_async_export,
            "export_batch_size": self.export_batch_size,
            "export_interval_seconds": self.export_interval_seconds,
            "enable_pii_detection": self.enable_pii_detection,
            "enable_prompt_capture": self.enable_prompt_capture,
            "enable_prompt_hashing": self.enable_prompt_hashing,
            "pii_detection_sensitivity": self.pii_detection_sensitivity,
            "redaction_strategy": self.redaction_strategy,
            "enable_token_counting": self.enable_token_counting,
            "enable_cost_tracking": self.enable_cost_tracking,
            "enable_auto_evaluation": self.enable_auto_evaluation,
            "enable_conversation_tracking": self.enable_conversation_tracking,
            "enable_guardrails": self.enable_guardrails,
            "enable_injection_detection": self.enable_injection_detection,
            "enable_anomaly_detection": self.enable_anomaly_detection,
            "enable_prompt_management": self.enable_prompt_management,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TracerConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
