"""
Fallback chain tracking for LLM routing.
"""

from typing import List, Optional, Any, Callable, Dict
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class FallbackResult:
    """Result of a fallback chain execution."""
    
    success: bool
    model_used: str
    attempts: int
    total_latency_ms: float
    response: Any = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "model_used": self.model_used,
            "attempts": self.attempts,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "errors": self.errors,
        }


class FallbackChain:
    """
    Manages a chain of fallback models with automatic retry.
    """
    
    def __init__(
        self,
        models: List[str],
        max_retries_per_model: int = 1,
        retry_delay_ms: float = 100,
    ):
        """
        Initialize the fallback chain.
        
        Args:
            models: Ordered list of models to try
            max_retries_per_model: Max retries per model before moving to next
            retry_delay_ms: Delay between retries
        """
        self.models = models
        self.max_retries = max_retries_per_model
        self.retry_delay_ms = retry_delay_ms
        self._errors: List[Dict[str, Any]] = []
    
    def execute(
        self,
        call_fn: Callable[[str], Any],
        should_retry: Optional[Callable[[Exception], bool]] = None,
    ) -> FallbackResult:
        """
        Execute the fallback chain.
        
        Args:
            call_fn: Function that takes model name and returns response
            should_retry: Optional function to determine if error is retryable
            
        Returns:
            FallbackResult with outcome
        """
        start_time = time.time()
        attempts = 0
        self._errors = []
        
        for model in self.models:
            for retry in range(self.max_retries + 1):
                attempts += 1
                
                try:
                    response = call_fn(model)
                    
                    return FallbackResult(
                        success=True,
                        model_used=model,
                        attempts=attempts,
                        total_latency_ms=(time.time() - start_time) * 1000,
                        response=response,
                        errors=self._errors,
                    )
                    
                except Exception as e:
                    error_info = {
                        "model": model,
                        "attempt": retry + 1,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    self._errors.append(error_info)
                    
                    if should_retry and not should_retry(e):
                        break
                    
                    if retry < self.max_retries:
                        time.sleep(self.retry_delay_ms / 1000)
        
        return FallbackResult(
            success=False,
            model_used=self.models[-1] if self.models else "none",
            attempts=attempts,
            total_latency_ms=(time.time() - start_time) * 1000,
            response=None,
            errors=self._errors,
        )
    
    async def execute_async(
        self,
        call_fn: Callable[[str], Any],
        should_retry: Optional[Callable[[Exception], bool]] = None,
    ) -> FallbackResult:
        """
        Execute the fallback chain asynchronously.
        
        Args:
            call_fn: Async function that takes model name and returns response
            should_retry: Optional function to determine if error is retryable
            
        Returns:
            FallbackResult with outcome
        """
        import asyncio
        
        start_time = time.time()
        attempts = 0
        self._errors = []
        
        for model in self.models:
            for retry in range(self.max_retries + 1):
                attempts += 1
                
                try:
                    response = await call_fn(model)
                    
                    return FallbackResult(
                        success=True,
                        model_used=model,
                        attempts=attempts,
                        total_latency_ms=(time.time() - start_time) * 1000,
                        response=response,
                        errors=self._errors,
                    )
                    
                except Exception as e:
                    error_info = {
                        "model": model,
                        "attempt": retry + 1,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    self._errors.append(error_info)
                    
                    if should_retry and not should_retry(e):
                        break
                    
                    if retry < self.max_retries:
                        await asyncio.sleep(self.retry_delay_ms / 1000)
        
        return FallbackResult(
            success=False,
            model_used=self.models[-1] if self.models else "none",
            attempts=attempts,
            total_latency_ms=(time.time() - start_time) * 1000,
            response=None,
            errors=self._errors,
        )


def is_retryable_error(error: Exception) -> bool:
    """
    Default function to determine if an error is retryable.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error should trigger a retry
    """
    error_name = type(error).__name__.lower()
    error_msg = str(error).lower()
    
    retryable_patterns = [
        "rate_limit",
        "ratelimit",
        "timeout",
        "connection",
        "server_error",
        "503",
        "502",
        "429",
        "overloaded",
        "capacity",
    ]
    
    for pattern in retryable_patterns:
        if pattern in error_name or pattern in error_msg:
            return True
    
    return False
