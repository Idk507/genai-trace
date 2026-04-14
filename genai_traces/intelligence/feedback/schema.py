"""
Feedback schema definitions for GenAI-Traces.

Defines the FeedbackRecord dataclass and related types.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class FeedbackType(Enum):
    """Types of feedback that can be recorded."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    CORRECTION = "correction"
    COMMENT = "comment"
    PREFERENCE = "preference"
    CUSTOM = "custom"


class FeedbackSource(Enum):
    """Source of the feedback."""
    USER = "user"
    ANNOTATOR = "annotator"
    AUTOMATED = "automated"
    SYSTEM = "system"


@dataclass
class FeedbackRecord:
    """
    Represents a feedback record for an LLM interaction.
    
    Attributes:
        trace_id: ID of the trace this feedback is for
        span_id: Optional span ID for more specific feedback
        feedback_type: Type of feedback (thumbs, rating, etc.)
        value: The feedback value (varies by type)
        source: Source of the feedback
        user_id: Optional user identifier
        timestamp: When the feedback was recorded
        metadata: Additional metadata
    """
    trace_id: str
    feedback_type: FeedbackType
    value: Any
    span_id: Optional[str] = None
    source: FeedbackSource = FeedbackSource.USER
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    comment: Optional[str] = None
    correction: Optional[str] = None
    dimensions: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["feedback_type"] = self.feedback_type.value
        data["source"] = self.source.value
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackRecord":
        """Create from dictionary."""
        data = data.copy()
        data["feedback_type"] = FeedbackType(data["feedback_type"])
        data["source"] = FeedbackSource(data["source"])
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
    
    def is_positive(self) -> bool:
        """Check if this is positive feedback."""
        if self.feedback_type == FeedbackType.THUMBS_UP:
            return True
        if self.feedback_type == FeedbackType.THUMBS_DOWN:
            return False
        if self.feedback_type == FeedbackType.RATING:
            if isinstance(self.value, (int, float)):
                return self.value >= 4
        return False
    
    def is_negative(self) -> bool:
        """Check if this is negative feedback."""
        if self.feedback_type == FeedbackType.THUMBS_DOWN:
            return True
        if self.feedback_type == FeedbackType.THUMBS_UP:
            return False
        if self.feedback_type == FeedbackType.RATING:
            if isinstance(self.value, (int, float)):
                return self.value <= 2
        return False


@dataclass
class FeedbackDimension:
    """
    Represents a dimension for multi-dimensional feedback.
    
    Attributes:
        name: Name of the dimension (e.g., "helpfulness", "accuracy")
        min_value: Minimum value for this dimension
        max_value: Maximum value for this dimension
        description: Description of what this dimension measures
    """
    name: str
    min_value: float = 1.0
    max_value: float = 5.0
    description: str = ""
    
    def validate(self, value: float) -> bool:
        """Check if a value is valid for this dimension."""
        return self.min_value <= value <= self.max_value
    
    def normalize(self, value: float) -> float:
        """Normalize value to 0-1 range."""
        return (value - self.min_value) / (self.max_value - self.min_value)


@dataclass
class FeedbackSchema:
    """
    Schema for collecting structured feedback.
    
    Attributes:
        name: Name of the feedback schema
        dimensions: List of feedback dimensions
        allow_comments: Whether to allow free-text comments
        require_correction: Whether to require correction for negative feedback
    """
    name: str
    dimensions: List[FeedbackDimension] = field(default_factory=list)
    allow_comments: bool = True
    require_correction: bool = False
    
    def add_dimension(self, dimension: FeedbackDimension) -> None:
        """Add a dimension to the schema."""
        self.dimensions.append(dimension)
    
    def validate_feedback(self, feedback: FeedbackRecord) -> List[str]:
        """Validate feedback against this schema."""
        errors = []
        
        for dim in self.dimensions:
            if dim.name in feedback.dimensions:
                value = feedback.dimensions[dim.name]
                if not dim.validate(value):
                    errors.append(
                        f"Dimension '{dim.name}' value {value} out of range "
                        f"[{dim.min_value}, {dim.max_value}]"
                    )
        
        if self.require_correction and feedback.is_negative():
            if not feedback.correction:
                errors.append("Correction required for negative feedback")
        
        return errors


def create_default_schema() -> FeedbackSchema:
    """Create a default feedback schema with common dimensions."""
    schema = FeedbackSchema(name="default")
    
    schema.add_dimension(FeedbackDimension(
        name="helpfulness",
        description="How helpful was the response?",
    ))
    
    schema.add_dimension(FeedbackDimension(
        name="accuracy",
        description="How accurate was the information?",
    ))
    
    schema.add_dimension(FeedbackDimension(
        name="relevance",
        description="How relevant was the response to the question?",
    ))
    
    schema.add_dimension(FeedbackDimension(
        name="clarity",
        description="How clear and understandable was the response?",
    ))
    
    return schema
