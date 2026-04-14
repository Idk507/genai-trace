"""Feedback collection utilities."""

from .collector import record_feedback, FeedbackRecord
from .schema import FeedbackRecord as FeedbackSchema, FeedbackType, FeedbackSource
from .aggregator import FeedbackAggregator, get_feedback_aggregator


class FeedbackCollector:
    """
    Collects and manages feedback for traces.
    
    Usage:
        collector = FeedbackCollector()
        collector.record(trace_id="...", score=5, comment="Great!")
    """
    
    def __init__(self):
        self._aggregator = get_feedback_aggregator()
    
    def record(
        self,
        trace_id: str,
        score: int = None,
        rating: str = None,
        comment: str = None,
        **kwargs
    ) -> FeedbackRecord:
        """Record feedback."""
        return record_feedback(
            trace_id=trace_id,
            score=score,
            rating=rating,
            comment=comment,
            **kwargs
        )
    
    def get_aggregator(self) -> FeedbackAggregator:
        """Get the feedback aggregator."""
        return self._aggregator


__all__ = [
    "record_feedback",
    "FeedbackRecord",
    "FeedbackCollector",
    "FeedbackAggregator",
    "get_feedback_aggregator",
    "FeedbackType",
    "FeedbackSource",
]
