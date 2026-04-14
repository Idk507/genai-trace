"""
Session management for GenAI-Traces.

Provides session grouping and metadata tracking.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import uuid


@dataclass
class Session:
    """
    Represents a user session.
    
    Attributes:
        session_id: Unique session identifier
        user_id: Optional user identifier
        started_at: When the session started
        last_activity: Last activity timestamp
        conversation_ids: List of conversation IDs in this session
        metadata: Session metadata
    """
    session_id: str
    user_id: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    conversation_ids: List[str] = field(default_factory=list)
    trace_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_conversation(self, conversation_id: str) -> None:
        """Add a conversation to this session."""
        if conversation_id not in self.conversation_ids:
            self.conversation_ids.append(conversation_id)
        self.last_activity = datetime.utcnow()
    
    def add_trace(self, trace_id: str) -> None:
        """Add a trace to this session."""
        if trace_id not in self.trace_ids:
            self.trace_ids.append(trace_id)
        self.last_activity = datetime.utcnow()
    
    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if the session has expired."""
        expiry_time = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.utcnow() > expiry_time
    
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        return (self.last_activity - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "conversation_ids": self.conversation_ids,
            "trace_ids": self.trace_ids,
            "metadata": self.metadata,
            "duration_seconds": self.duration_seconds(),
        }


class SessionManager:
    """
    Manages user sessions.
    
    Usage:
        manager = SessionManager()
        
        session = manager.get_or_create_session(user_id="user-123")
        session.add_conversation("conv-456")
    """
    
    def __init__(self, session_timeout_minutes: int = 30):
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, List[str]] = defaultdict(list)
        self._timeout_minutes = session_timeout_minutes
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )
        
        self._sessions[session_id] = session
        if user_id:
            self._user_sessions[user_id].append(session_id)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def get_or_create_session(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Get an existing session or create a new one."""
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            if not session.is_expired(self._timeout_minutes):
                session.update_activity()
                return session
        
        if user_id:
            for sid in reversed(self._user_sessions.get(user_id, [])):
                session = self._sessions.get(sid)
                if session and not session.is_expired(self._timeout_minutes):
                    session.update_activity()
                    return session
        
        return self.create_session(user_id, metadata)
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for a user."""
        session_ids = self._user_sessions.get(user_id, [])
        return [
            self._sessions[sid]
            for sid in session_ids
            if sid in self._sessions
        ]
    
    def get_active_sessions(self) -> List[Session]:
        """Get all active (non-expired) sessions."""
        return [
            session
            for session in self._sessions.values()
            if not session.is_expired(self._timeout_minutes)
        ]
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        expired = [
            sid
            for sid, session in self._sessions.items()
            if session.is_expired(self._timeout_minutes)
        ]
        
        for sid in expired:
            session = self._sessions.pop(sid)
            if session.user_id and session.user_id in self._user_sessions:
                self._user_sessions[session.user_id] = [
                    s for s in self._user_sessions[session.user_id] if s != sid
                ]
        
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        active = self.get_active_sessions()
        
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": len(active),
            "unique_users": len(self._user_sessions),
            "avg_duration_seconds": (
                sum(s.duration_seconds() for s in active) / len(active)
                if active else 0
            ),
        }


_global_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager."""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    return _global_session_manager
