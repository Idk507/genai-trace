"""
Pre-call token estimation for GenAI-Traces.

Estimates token counts before making LLM calls to enable:
- Cost prediction
- Context window management
- Rate limiting decisions
"""

from typing import List, Dict, Any, Optional, Union
from functools import lru_cache


class TokenEstimator:
    """
    Estimates token counts before LLM calls.
    
    Uses tiktoken for accurate estimation with fallback to heuristics.
    """
    
    CHARS_PER_TOKEN_ESTIMATES = {
        "gpt-4": 4.0,
        "gpt-3.5": 4.0,
        "claude": 3.5,
        "llama": 4.0,
        "default": 4.0,
    }
    
    COMPLETION_RATIO_ESTIMATES = {
        "summarization": 0.3,
        "translation": 1.0,
        "qa": 0.5,
        "chat": 0.8,
        "code": 1.2,
        "default": 0.5,
    }
    
    def __init__(self, use_tiktoken: bool = True):
        """
        Initialize the token estimator.
        
        Args:
            use_tiktoken: Whether to use tiktoken for accurate counting
        """
        self._use_tiktoken = use_tiktoken
        self._tiktoken_available = False
        
        if use_tiktoken:
            try:
                import tiktoken
                self._tiktoken_available = True
            except ImportError:
                pass
    
    def estimate_prompt_tokens(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: str = "gpt-4o",
    ) -> int:
        """
        Estimate token count for a prompt.
        
        Args:
            prompt: Either a string prompt or list of message dicts
            model: Model name for accurate estimation
            
        Returns:
            Estimated token count
        """
        if isinstance(prompt, list):
            return self._estimate_messages_tokens(prompt, model)
        return self._estimate_text_tokens(prompt, model)
    
    def estimate_completion_tokens(
        self,
        prompt_tokens: int,
        task_type: str = "default",
        max_tokens: Optional[int] = None,
    ) -> int:
        """
        Estimate expected completion tokens based on prompt and task type.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            task_type: Type of task (summarization, translation, qa, chat, code)
            max_tokens: Maximum tokens if specified in request
            
        Returns:
            Estimated completion token count
        """
        ratio = self.COMPLETION_RATIO_ESTIMATES.get(
            task_type, 
            self.COMPLETION_RATIO_ESTIMATES["default"]
        )
        
        estimated = int(prompt_tokens * ratio)
        
        if max_tokens:
            estimated = min(estimated, max_tokens)
        
        return max(estimated, 10)
    
    def estimate_total_tokens(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: str = "gpt-4o",
        task_type: str = "default",
        max_tokens: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Estimate total tokens for a request.
        
        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens estimates
        """
        prompt_tokens = self.estimate_prompt_tokens(prompt, model)
        completion_tokens = self.estimate_completion_tokens(
            prompt_tokens, task_type, max_tokens
        )
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    
    def check_context_window(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: str = "gpt-4o",
        max_completion_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Check if prompt fits within model's context window.
        
        Returns:
            Dict with fits, prompt_tokens, available_for_completion, context_limit
        """
        context_limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
        }
        
        context_limit = context_limits.get(model, 8192)
        for key, limit in context_limits.items():
            if key in model.lower():
                context_limit = limit
                break
        
        prompt_tokens = self.estimate_prompt_tokens(prompt, model)
        available = context_limit - prompt_tokens
        
        return {
            "fits": available >= max_completion_tokens,
            "prompt_tokens": prompt_tokens,
            "available_for_completion": max(0, available),
            "context_limit": context_limit,
            "utilization": prompt_tokens / context_limit,
        }
    
    def _estimate_text_tokens(self, text: str, model: str) -> int:
        """Estimate tokens for plain text."""
        if self._tiktoken_available:
            try:
                return self._count_with_tiktoken(text, model)
            except Exception:
                pass
        
        chars_per_token = self._get_chars_per_token(model)
        return max(1, int(len(text) / chars_per_token))
    
    def _estimate_messages_tokens(
        self, 
        messages: List[Dict[str, str]], 
        model: str
    ) -> int:
        """Estimate tokens for chat messages."""
        if self._tiktoken_available:
            try:
                return self._count_messages_with_tiktoken(messages, model)
            except Exception:
                pass
        
        total = 0
        chars_per_token = self._get_chars_per_token(model)
        
        for message in messages:
            content = message.get("content", "")
            total += int(len(content) / chars_per_token)
            total += 4
        
        total += 3
        
        return max(1, total)
    
    @lru_cache(maxsize=100)
    def _count_with_tiktoken(self, text: str, model: str) -> int:
        """Count tokens using tiktoken."""
        import tiktoken
        
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    
    def _count_messages_with_tiktoken(
        self, 
        messages: List[Dict[str, str]], 
        model: str
    ) -> int:
        """Count tokens in messages using tiktoken."""
        import tiktoken
        
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens_per_message = 3
        tokens_per_name = 1
        
        total = 0
        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    total += len(encoding.encode(value))
                if key == "name":
                    total += tokens_per_name
        
        total += 3
        
        return total
    
    def _get_chars_per_token(self, model: str) -> float:
        """Get estimated characters per token for a model."""
        model_lower = model.lower()
        
        for key, ratio in self.CHARS_PER_TOKEN_ESTIMATES.items():
            if key in model_lower:
                return ratio
        
        return self.CHARS_PER_TOKEN_ESTIMATES["default"]


_estimator = TokenEstimator()


def estimate_tokens(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = "gpt-4o",
) -> int:
    """Convenience function to estimate prompt tokens."""
    return _estimator.estimate_prompt_tokens(prompt, model)


def check_context_window(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = "gpt-4o",
    max_completion_tokens: int = 4096,
) -> Dict[str, Any]:
    """Convenience function to check context window fit."""
    return _estimator.check_context_window(prompt, model, max_completion_tokens)
