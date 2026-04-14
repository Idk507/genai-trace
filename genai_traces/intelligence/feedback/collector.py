"""
Feedback collection for traces.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, List


@dataclass
class FeedbackRecord:
    """Record of feedback for a trace."""
    trace_id: str
    span_id: Optional[str] = None
    score: Optional[int] = None  # 1–5
    rating: Optional[str] = None  # "thumbs_up" | "thumbs_down"
    comment: Optional[str] = None
    dimensions: Dict[str, float] = field(default_factory=dict)
    source: str = "human"  # human | automated | llm_judge
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "score": self.score,
            "rating": self.rating,
            "comment": self.comment,
            "dimensions": self.dimensions,
            "source": self.source,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# In-memory feedback store (for development/testing)
_feedback_store: List[FeedbackRecord] = []


def record_feedback(
    trace_id: str,
    score: Optional[int] = None,
    rating: Optional[str] = None,
    comment: Optional[str] = None,
    dimensions: Optional[Dict[str, float]] = None,
    source: str = "human",
    user_id: Optional[str] = None,
    span_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FeedbackRecord:
    """
    Record human or automated feedback for a trace.
    
    Args:
        trace_id: ID of the trace to attach feedback to
        score: Numeric score (1-5)
        rating: Thumbs up/down rating
        comment: Text comment
        dimensions: Multi-dimensional scores (e.g., {"accuracy": 5, "helpfulness": 4})
        source: Source of feedback (human, automated, llm_judge)
        user_id: ID of the user providing feedback
        span_id: Optional specific span ID
        metadata: Additional metadata
        
    Returns:
        FeedbackRecord instance
        
    Example:
        record_feedback(
            trace_id=get_current_trace_id(),
            score=4,
            rating="thumbs_up",
            comment="Very accurate",
            dimensions={"accuracy": 5, "helpfulness": 4}
        )
    """
    fb = FeedbackRecord(
        trace_id=trace_id,
        span_id=span_id,
        score=score,
        rating=rating,
        comment=comment,
        dimensions=dimensions or {},
        source=source,
        user_id=user_id,
        metadata=metadata or {},
    )
    
    _feedback_store.append(fb)
    
    # Export to registered exporters
    try:
        from ...core.tracer import get_tracer
        tracer = get_tracer()
        for exporter in tracer.exporters:
            if hasattr(exporter, "export_feedback"):
                exporter.export_feedback(fb)
    except Exception:
        pass
    
    return fb


def get_feedback_for_trace(trace_id: str) -> List[FeedbackRecord]:
    """Get all feedback for a trace."""
    return [fb for fb in _feedback_store if fb.trace_id == trace_id]


def get_all_feedback() -> List[FeedbackRecord]:
    """Get all recorded feedback."""
    return _feedback_store.copy()


def clear_feedback() -> None:
    """Clear all feedback (for testing)."""
    _feedback_store.clear()
