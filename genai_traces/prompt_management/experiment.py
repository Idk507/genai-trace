"""
Experiment tracking for GenAI-Traces.

Provides experiment tracking and results analysis for prompt experiments.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics


class ExperimentStatus(Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentMetric:
    """A metric recorded for an experiment."""
    name: str
    value: float
    variant: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results for a single variant in an experiment."""
    variant: str
    sample_count: int
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        self.sample_count = max(len(v) for v in self.metrics.values())
    
    def get_mean(self, metric: str) -> Optional[float]:
        """Get mean of a metric."""
        values = self.metrics.get(metric, [])
        return statistics.mean(values) if values else None
    
    def get_std(self, metric: str) -> Optional[float]:
        """Get standard deviation of a metric."""
        values = self.metrics.get(metric, [])
        return statistics.stdev(values) if len(values) > 1 else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant": self.variant,
            "sample_count": self.sample_count,
            "metrics": {
                name: {
                    "values": values,
                    "mean": statistics.mean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                }
                for name, values in self.metrics.items()
            },
        }


@dataclass
class Experiment:
    """
    Represents a prompt experiment.
    
    Attributes:
        experiment_id: Unique experiment identifier
        name: Human-readable name
        description: Description of the experiment
        variants: List of variant names
        status: Current experiment status
        start_date: When the experiment started
        end_date: When the experiment ended
        results: Results by variant
    """
    experiment_id: str
    name: str
    description: str = ""
    variants: List[str] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    results: Dict[str, ExperimentResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def start(self) -> None:
        """Start the experiment."""
        self.status = ExperimentStatus.RUNNING
        self.start_date = datetime.utcnow()
        
        for variant in self.variants:
            if variant not in self.results:
                self.results[variant] = ExperimentResult(variant=variant, sample_count=0)
    
    def pause(self) -> None:
        """Pause the experiment."""
        self.status = ExperimentStatus.PAUSED
    
    def resume(self) -> None:
        """Resume a paused experiment."""
        self.status = ExperimentStatus.RUNNING
    
    def complete(self) -> None:
        """Mark the experiment as completed."""
        self.status = ExperimentStatus.COMPLETED
        self.end_date = datetime.utcnow()
    
    def cancel(self) -> None:
        """Cancel the experiment."""
        self.status = ExperimentStatus.CANCELLED
        self.end_date = datetime.utcnow()
    
    def record_metric(self, variant: str, metric: str, value: float) -> None:
        """Record a metric for a variant."""
        if variant not in self.results:
            self.results[variant] = ExperimentResult(variant=variant, sample_count=0)
        self.results[variant].add_metric(metric, value)
    
    def get_winner(self, metric: str) -> Optional[str]:
        """Get the winning variant for a metric (highest mean)."""
        best_variant = None
        best_mean = float("-inf")
        
        for variant, result in self.results.items():
            mean = result.get_mean(metric)
            if mean is not None and mean > best_mean:
                best_mean = mean
                best_variant = variant
        
        return best_variant
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status.value,
            "variants": self.variants,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "results": {
                variant: result.to_dict()
                for variant, result in self.results.items()
            },
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "variants": self.variants,
            "status": self.status.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "results": {v: r.to_dict() for v, r in self.results.items()},
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class ExperimentTracker:
    """
    Tracks multiple experiments.
    
    Usage:
        tracker = ExperimentTracker()
        
        exp = tracker.create_experiment(
            name="Prompt Style Test",
            variants=["formal", "casual"],
        )
        exp.start()
        
        # Record metrics
        tracker.record("exp-123", "formal", "quality", 0.85)
        tracker.record("exp-123", "casual", "quality", 0.82)
        
        # Get results
        results = tracker.get_experiment("exp-123").get_summary()
    """
    
    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
    
    def create_experiment(
        self,
        name: str,
        variants: List[str],
        description: str = "",
        experiment_id: Optional[str] = None,
        **metadata,
    ) -> Experiment:
        """Create a new experiment."""
        import uuid
        
        exp_id = experiment_id or str(uuid.uuid4())[:8]
        
        experiment = Experiment(
            experiment_id=exp_id,
            name=name,
            description=description,
            variants=variants,
            metadata=metadata,
        )
        
        self._experiments[exp_id] = experiment
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
    ) -> List[Experiment]:
        """List experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return experiments
    
    def record(
        self,
        experiment_id: str,
        variant: str,
        metric: str,
        value: float,
    ) -> None:
        """Record a metric for an experiment variant."""
        experiment = self._experiments.get(experiment_id)
        if experiment:
            experiment.record_metric(variant, metric, value)
    
    def get_active_experiments(self) -> List[Experiment]:
        """Get all running experiments."""
        return self.list_experiments(status=ExperimentStatus.RUNNING)


_global_tracker: Optional[ExperimentTracker] = None


def get_experiment_tracker() -> ExperimentTracker:
    """Get the global experiment tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ExperimentTracker()
    return _global_tracker
