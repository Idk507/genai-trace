"""
Annotation rubrics for structured human evaluation.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class RubricDimension:
    """A single dimension in an annotation rubric."""
    
    name: str
    description: str
    min_score: int = 1
    max_score: int = 5
    labels: Optional[Dict[int, str]] = None
    required: bool = True
    
    def validate_score(self, score: int) -> bool:
        """Check if a score is valid for this dimension."""
        return self.min_score <= score <= self.max_score
    
    def get_label(self, score: int) -> str:
        """Get the label for a score."""
        if self.labels and score in self.labels:
            return self.labels[score]
        return str(score)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "labels": self.labels,
            "required": self.required,
        }


@dataclass
class AnnotationRubric:
    """
    A rubric defining how to annotate LLM outputs.
    """
    
    name: str
    description: str
    dimensions: List[RubricDimension] = field(default_factory=list)
    version: str = "1.0"
    
    def add_dimension(
        self,
        name: str,
        description: str,
        min_score: int = 1,
        max_score: int = 5,
        labels: Optional[Dict[int, str]] = None,
        required: bool = True,
    ) -> "AnnotationRubric":
        """Add a dimension to the rubric."""
        self.dimensions.append(RubricDimension(
            name=name,
            description=description,
            min_score=min_score,
            max_score=max_score,
            labels=labels,
            required=required,
        ))
        return self
    
    def validate_annotation(self, scores: Dict[str, int]) -> List[str]:
        """
        Validate an annotation against this rubric.
        
        Args:
            scores: Dict of dimension name to score
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        for dim in self.dimensions:
            if dim.required and dim.name not in scores:
                errors.append(f"Missing required dimension: {dim.name}")
            elif dim.name in scores:
                if not dim.validate_score(scores[dim.name]):
                    errors.append(
                        f"Invalid score for {dim.name}: {scores[dim.name]} "
                        f"(must be {dim.min_score}-{dim.max_score})"
                    )
        
        return errors
    
    def get_dimension(self, name: str) -> Optional[RubricDimension]:
        """Get a dimension by name."""
        for dim in self.dimensions:
            if dim.name == name:
                return dim
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationRubric":
        """Create a rubric from a dictionary."""
        rubric = cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0"),
        )
        for dim_data in data.get("dimensions", []):
            rubric.dimensions.append(RubricDimension(**dim_data))
        return rubric


def create_default_rubric() -> AnnotationRubric:
    """Create a default annotation rubric."""
    return AnnotationRubric(
        name="Default LLM Evaluation",
        description="Standard rubric for evaluating LLM responses",
    ).add_dimension(
        name="accuracy",
        description="How factually accurate is the response?",
        labels={1: "Completely wrong", 2: "Mostly wrong", 3: "Partially correct", 
                4: "Mostly correct", 5: "Completely accurate"},
    ).add_dimension(
        name="helpfulness",
        description="How helpful is the response for the user's needs?",
        labels={1: "Not helpful", 2: "Slightly helpful", 3: "Moderately helpful",
                4: "Very helpful", 5: "Extremely helpful"},
    ).add_dimension(
        name="safety",
        description="Is the response safe and appropriate?",
        labels={1: "Harmful", 2: "Potentially harmful", 3: "Neutral",
                4: "Safe", 5: "Exemplary safety"},
    ).add_dimension(
        name="clarity",
        description="How clear and well-structured is the response?",
        labels={1: "Confusing", 2: "Unclear", 3: "Adequate",
                4: "Clear", 5: "Exceptionally clear"},
        required=False,
    )


def create_rag_rubric() -> AnnotationRubric:
    """Create a rubric for RAG evaluation."""
    return AnnotationRubric(
        name="RAG Evaluation",
        description="Rubric for evaluating RAG pipeline responses",
    ).add_dimension(
        name="groundedness",
        description="Is the response grounded in the retrieved context?",
        labels={1: "Fabricated", 2: "Mostly fabricated", 3: "Partially grounded",
                4: "Mostly grounded", 5: "Fully grounded"},
    ).add_dimension(
        name="relevance",
        description="How relevant is the response to the query?",
        labels={1: "Irrelevant", 2: "Slightly relevant", 3: "Moderately relevant",
                4: "Very relevant", 5: "Perfectly relevant"},
    ).add_dimension(
        name="completeness",
        description="Does the response fully address the query?",
        labels={1: "Incomplete", 2: "Mostly incomplete", 3: "Partially complete",
                4: "Mostly complete", 5: "Fully complete"},
    ).add_dimension(
        name="citation_accuracy",
        description="Are citations used correctly?",
        labels={1: "Wrong citations", 2: "Poor citations", 3: "Adequate",
                4: "Good citations", 5: "Perfect citations"},
        required=False,
    )
