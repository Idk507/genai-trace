"""
Feedback aggregation for GenAI-Traces.

Aggregates multi-dimensional feedback scores.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import statistics

from .schema import FeedbackRecord, FeedbackType


@dataclass
class AggregatedFeedback:
    """Aggregated feedback statistics."""
    total_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    average_rating: Optional[float] = None
    dimension_averages: Dict[str, float] = field(default_factory=dict)
    dimension_counts: Dict[str, int] = field(default_factory=dict)
    rating_distribution: Dict[int, int] = field(default_factory=dict)
    
    @property
    def positive_rate(self) -> float:
        """Calculate positive feedback rate."""
        if self.total_count == 0:
            return 0.0
        return self.positive_count / self.total_count
    
    @property
    def negative_rate(self) -> float:
        """Calculate negative feedback rate."""
        if self.total_count == 0:
            return 0.0
        return self.negative_count / self.total_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_count": self.total_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "positive_rate": self.positive_rate,
            "negative_rate": self.negative_rate,
            "average_rating": self.average_rating,
            "dimension_averages": self.dimension_averages,
            "dimension_counts": self.dimension_counts,
            "rating_distribution": self.rating_distribution,
        }


class FeedbackAggregator:
    """
    Aggregates feedback records for analysis.
    
    Usage:
        aggregator = FeedbackAggregator()
        
        for feedback in feedback_records:
            aggregator.add(feedback)
        
        stats = aggregator.get_aggregate()
        print(f"Positive rate: {stats.positive_rate:.2%}")
    """
    
    def __init__(self):
        self._records: List[FeedbackRecord] = []
        self._by_trace: Dict[str, List[FeedbackRecord]] = defaultdict(list)
        self._by_user: Dict[str, List[FeedbackRecord]] = defaultdict(list)
        self._by_session: Dict[str, List[FeedbackRecord]] = defaultdict(list)
    
    def add(self, feedback: FeedbackRecord) -> None:
        """Add a feedback record."""
        self._records.append(feedback)
        self._by_trace[feedback.trace_id].append(feedback)
        
        if feedback.user_id:
            self._by_user[feedback.user_id].append(feedback)
        
        if feedback.session_id:
            self._by_session[feedback.session_id].append(feedback)
    
    def get_aggregate(
        self,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> AggregatedFeedback:
        """Get aggregated feedback statistics."""
        records = self._filter_records(trace_id, user_id, session_id, since)
        return self._compute_aggregate(records)
    
    def get_aggregate_by_trace(self) -> Dict[str, AggregatedFeedback]:
        """Get aggregated feedback for each trace."""
        return {
            trace_id: self._compute_aggregate(records)
            for trace_id, records in self._by_trace.items()
        }
    
    def get_aggregate_by_user(self) -> Dict[str, AggregatedFeedback]:
        """Get aggregated feedback for each user."""
        return {
            user_id: self._compute_aggregate(records)
            for user_id, records in self._by_user.items()
        }
    
    def get_trends(
        self,
        window_hours: int = 24,
        num_windows: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get feedback trends over time."""
        now = datetime.utcnow()
        trends = []
        
        for i in range(num_windows):
            window_end = now - timedelta(hours=i * window_hours)
            window_start = window_end - timedelta(hours=window_hours)
            
            records = [
                r for r in self._records
                if window_start <= r.timestamp < window_end
            ]
            
            agg = self._compute_aggregate(records)
            trends.append({
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "stats": agg.to_dict(),
            })
        
        return list(reversed(trends))
    
    def get_low_rated_traces(
        self,
        threshold: float = 3.0,
        min_feedback_count: int = 1,
    ) -> List[str]:
        """Get trace IDs with low average ratings."""
        low_rated = []
        
        for trace_id, records in self._by_trace.items():
            if len(records) < min_feedback_count:
                continue
            
            agg = self._compute_aggregate(records)
            if agg.average_rating is not None and agg.average_rating < threshold:
                low_rated.append(trace_id)
        
        return low_rated
    
    def _filter_records(
        self,
        trace_id: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str],
        since: Optional[datetime],
    ) -> List[FeedbackRecord]:
        """Filter records based on criteria."""
        if trace_id:
            records = self._by_trace.get(trace_id, [])
        elif user_id:
            records = self._by_user.get(user_id, [])
        elif session_id:
            records = self._by_session.get(session_id, [])
        else:
            records = self._records
        
        if since:
            records = [r for r in records if r.timestamp >= since]
        
        return records
    
    def _compute_aggregate(self, records: List[FeedbackRecord]) -> AggregatedFeedback:
        """Compute aggregate statistics from records."""
        agg = AggregatedFeedback()
        agg.total_count = len(records)
        
        ratings = []
        dimension_values: Dict[str, List[float]] = defaultdict(list)
        
        for record in records:
            if record.is_positive():
                agg.positive_count += 1
            elif record.is_negative():
                agg.negative_count += 1
            else:
                agg.neutral_count += 1
            
            if record.feedback_type == FeedbackType.RATING:
                if isinstance(record.value, (int, float)):
                    ratings.append(record.value)
                    rating_int = int(record.value)
                    agg.rating_distribution[rating_int] = (
                        agg.rating_distribution.get(rating_int, 0) + 1
                    )
            
            for dim_name, dim_value in record.dimensions.items():
                dimension_values[dim_name].append(dim_value)
        
        if ratings:
            agg.average_rating = statistics.mean(ratings)
        
        for dim_name, values in dimension_values.items():
            agg.dimension_averages[dim_name] = statistics.mean(values)
            agg.dimension_counts[dim_name] = len(values)
        
        return agg
    
    def clear(self) -> None:
        """Clear all feedback records."""
        self._records.clear()
        self._by_trace.clear()
        self._by_user.clear()
        self._by_session.clear()


_global_aggregator: Optional[FeedbackAggregator] = None


def get_feedback_aggregator() -> FeedbackAggregator:
    """Get the global feedback aggregator."""
    global _global_aggregator
    if _global_aggregator is None:
        _global_aggregator = FeedbackAggregator()
    return _global_aggregator
