"""
Token counting utilities using tiktoken.
"""

from typing import Dict, List, Optional
from functools import lru_cache


# Model → encoding name mapping
_MODEL_ENCODING_MAP = {
    # OpenAI GPT-4 family
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-2024-05-13": "o200k_base",
    "gpt-4o-2024-08-06": "o200k_base",
    
    # OpenAI GPT-3.5 family
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    
    # OpenAI o1 family
    "o1-preview": "o200k_base",
    "o1-mini": "o200k_base",
    
    # Anthropic Claude (approximate using cl100k_base)
    "claude-3-opus": "cl100k_base",
    "claude-3-opus-20240229": "cl100k_base",
    "claude-3-sonnet": "cl100k_base",
    "claude-3-sonnet-20240229": "cl100k_base",
    "claude-3-5-sonnet": "cl100k_base",
    "claude-3-5-sonnet-20240620": "cl100k_base",
    "claude-3-5-sonnet-20241022": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
    "claude-3-haiku-20240307": "cl100k_base",
    "claude-sonnet-4-6": "cl100k_base",
    
    # Embeddings
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}


class TokenCounter:
    """
    Token counter using tiktoken for accurate token counting.
    
    Supports OpenAI models natively and approximates counts for
    other providers using compatible encodings.
    
    Usage:
        counter = TokenCounter()
        count = counter.count("Hello, world!", model="gpt-4o")
        message_count = counter.count_messages([
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ], model="gpt-4o")
    """
    
    def __init__(self, cache_encodings: bool = True):
        """
        Initialize the token counter.
        
        Args:
            cache_encodings: Whether to cache encoding objects
        """
        self._cache = cache_encodings
        self._tiktoken_available = self._check_tiktoken()
    
    def _check_tiktoken(self) -> bool:
        """Check if tiktoken is available."""
        try:
            import tiktoken
            return True
        except ImportError:
            return False
    
    @lru_cache(maxsize=16)
    def _get_encoding(self, model: str):
        """Get the tiktoken encoding for a model."""
        if not self._tiktoken_available:
            return None
        
        import tiktoken
        
        enc_name = _MODEL_ENCODING_MAP.get(model, "cl100k_base")
        try:
            return tiktoken.get_encoding(enc_name)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    
    def count(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            model: Model name for encoding selection
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if not self._tiktoken_available:
            # Fallback: rough estimate of 4 chars per token
            return len(text) // 4
        
        enc = self._get_encoding(model)
        if enc is None:
            return len(text) // 4
        
        return len(enc.encode(text))
    
    def count_messages(
        self,
        messages: List[Dict],
        model: str = "gpt-4"
    ) -> int:
        """
        Count tokens for chat messages, including per-message overhead.
        
        Based on OpenAI's token counting cookbook.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name for encoding selection
            
        Returns:
            Total token count
        """
        if not self._tiktoken_available:
            # Fallback: rough estimate
            total_chars = sum(
                len(str(m.get("content", ""))) + len(str(m.get("role", "")))
                for m in messages
            )
            return total_chars // 4 + len(messages) * 4
        
        enc = self._get_encoding(model)
        if enc is None:
            total_chars = sum(
                len(str(m.get("content", ""))) + len(str(m.get("role", "")))
                for m in messages
            )
            return total_chars // 4 + len(messages) * 4
        
        # Per-message overhead varies by model
        if model.startswith("gpt-4o"):
            tokens_per_message = 3
            tokens_per_name = 1
        elif model.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message = 4
            tokens_per_name = -1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if value is not None:
                    num_tokens += len(enc.encode(str(value)))
                    if key == "name":
                        num_tokens += tokens_per_name
        
        # Reply priming
        num_tokens += 3
        
        return num_tokens
    
    def estimate_completion(
        self,
        prompt_tokens: int,
        max_tokens: int = 500
    ) -> int:
        """
        Heuristic pre-call estimate for completion tokens.
        
        Use for cost estimation before the call.
        Defaults to half of max_tokens as a conservative estimate.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            max_tokens: Maximum tokens allowed for completion
            
        Returns:
            Estimated completion tokens
        """
        return min(max_tokens, max(prompt_tokens // 4, 50))


# Global instance for convenience
_default_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get the default token counter instance."""
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter()
    return _default_counter


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Convenience function to count tokens."""
    return get_token_counter().count(text, model)


def count_message_tokens(messages: List[Dict], model: str = "gpt-4") -> int:
    """Convenience function to count message tokens."""
    return get_token_counter().count_messages(messages, model)
