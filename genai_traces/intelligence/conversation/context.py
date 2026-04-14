"""
Conversation context management for GenAI-Traces.

Provides APIs for setting and retrieving conversation context.
"""

from contextvars import ContextVar
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationContext:
    """
    Represents the context of a conversation.
    
    Attributes:
        conversation_id: Unique identifier for the conversation
        user_id: Optional user identifier
        session_id: Optional session identifier
        turn_number: Current turn number in the conversation
        messages: List of messages in the conversation
        metadata: Additional context metadata
    """
    conversation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    turn_number: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "turn": self.turn_number,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        })
        self.turn_number += 1
        self.updated_at = datetime.utcnow()
    
    def add_user_message(self, content: str, **kwargs) -> None:
        """Add a user message."""
        self.add_message("user", content, **kwargs)
    
    def add_assistant_message(self, content: str, **kwargs) -> None:
        """Add an assistant message."""
        self.add_message("assistant", content, **kwargs)
    
    def add_system_message(self, content: str, **kwargs) -> None:
        """Add a system message."""
        self.add_message("system", content, **kwargs)
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return msg.get("content")
        return None
    
    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message."""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                return msg.get("content")
        return None
    
    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)
    
    def get_user_message_count(self) -> int:
        """Get user message count."""
        return sum(1 for m in self.messages if m.get("role") == "user")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "messages": self.messages,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


_conversation_context: ContextVar[Optional[ConversationContext]] = ContextVar(
    "conversation_context",
    default=None,
)


def set_conversation_context(
    conversation_id: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationContext:
    """
    Set the current conversation context.
    
    Usage:
        context = set_conversation_context(
            conversation_id="conv-123",
            user_id="user-456",
            metadata={"source": "web"}
        )
        
        # Now all traces will include this context
    
    Args:
        conversation_id: Unique conversation identifier
        user_id: Optional user identifier
        session_id: Optional session identifier
        metadata: Additional metadata
        
    Returns:
        The created ConversationContext
    """
    context = ConversationContext(
        conversation_id=conversation_id,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata or {},
    )
    _conversation_context.set(context)
    return context


def get_conversation_context() -> Optional[ConversationContext]:
    """
    Get the current conversation context.
    
    Returns:
        The current ConversationContext or None if not set
    """
    return _conversation_context.get()


def clear_conversation_context() -> None:
    """Clear the current conversation context."""
    _conversation_context.set(None)


def update_conversation_context(**kwargs) -> Optional[ConversationContext]:
    """
    Update the current conversation context.
    
    Args:
        **kwargs: Fields to update
        
    Returns:
        The updated ConversationContext or None if not set
    """
    context = _conversation_context.get()
    if context:
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
        context.updated_at = datetime.utcnow()
    return context
