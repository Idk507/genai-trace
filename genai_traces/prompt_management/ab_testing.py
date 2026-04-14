"""
A/B testing framework for prompts and models.

Traffic-split A/B testing for prompts, models, and parameters.

Usage:
    ab = ABTestManager()
    
    # Define an experiment
    ab.create_experiment(
        experiment_id="summarize_style_v2",
        variants=[
            {"id": "control", "prompt_name": "summarize", "version": "1.0.0", "weight": 0.5},
            {"id": "concise", "prompt_name": "summarize", "version": "1.1.0", "weight": 0.5},
        ]
    )
    
    # Activate in context
    variant = ab.activate("summarize_style_v2", user_id="user_123")
    
    # Get assigned variant
    variant = ab.get_variant("summarize_style_v2", user_id="user_123")
"""

import random
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.context import _experiment_id, _variant_id


@dataclass
class ExperimentVariant:
    """A variant in an A/B experiment."""
    id: str
    prompt_name: Optional[str] = None
    version: Optional[str] = None
    model: Optional[str] = None
    weight: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """An A/B experiment configuration."""
    experiment_id: str
    variants: List[ExperimentVariant]
    status: str = "active"  # active | paused | concluded
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    results: Dict[str, Any] = field(default_factory=dict)
    
    def get_variant_for_user(self, user_id: Optional[str] = None) -> ExperimentVariant:
        """
        Get consistent variant assignment for a user.
        
        Same user always gets same variant. Falls back to random if no user_id.
        
        Args:
            user_id: User identifier for consistent assignment
            
        Returns:
            Assigned ExperimentVariant
        """
        if user_id:
            # Hash-based consistent assignment
            digest = int(hashlib.md5(
                f"{self.experiment_id}:{user_id}".encode()
            ).hexdigest(), 16)
            normalized = (digest % 10000) / 10000.0
        else:
            normalized = random.random()
        
        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if normalized < cumulative:
                return variant
        
        return self.variants[-1]


class ABTestManager:
    """
    Manager for A/B testing experiments.
    
    Supports:
    - Creating experiments with weighted variants
    - Consistent user assignment
    - Result recording and statistical analysis
    """
    
    def __init__(self, storage_path: str = "./ab_experiments.json"):
        """
        Initialize the A/B test manager.
        
        Args:
            storage_path: Path to JSON storage file
        """
        self._path = Path(storage_path)
        self._experiments: Dict[str, Experiment] = {}
        self._load()
    
    def create_experiment(
        self,
        experiment_id: str,
        variants: List[Dict],
        status: str = "active",
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            variants: List of variant configurations
            status: Initial status (active, paused, concluded)
            
        Returns:
            Created Experiment
            
        Raises:
            ValueError: If weights don't sum to 1.0
        """
        exp = Experiment(
            experiment_id=experiment_id,
            variants=[ExperimentVariant(**v) for v in variants],
            status=status,
        )
        
        # Validate weights sum to 1.0
        total = sum(v.weight for v in exp.variants)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Variant weights must sum to 1.0, got {total}")
        
        self._experiments[experiment_id] = exp
        self._save()
        return exp
    
    def activate(
        self,
        experiment_id: str,
        user_id: Optional[str] = None
    ) -> ExperimentVariant:
        """
        Activate an experiment in the current context.
        
        Sets context variables that will be automatically attached to spans.
        
        Args:
            experiment_id: Experiment to activate
            user_id: User ID for consistent assignment
            
        Returns:
            Assigned variant
        """
        exp = self._get_active(experiment_id)
        variant = exp.get_variant_for_user(user_id)
        
        _experiment_id.set(experiment_id)
        _variant_id.set(variant.id)
        
        return variant
    
    def get_variant(
        self,
        experiment_id: str,
        user_id: Optional[str] = None
    ) -> ExperimentVariant:
        """
        Get variant for a user without activating context.
        
        Args:
            experiment_id: Experiment ID
            user_id: User ID for consistent assignment
            
        Returns:
            Assigned variant
        """
        return self._get_active(experiment_id).get_variant_for_user(user_id)
    
    def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        metric: str,
        value: float,
    ) -> None:
        """
        Record a metric observation for a variant.
        
        Args:
            experiment_id: Experiment ID
            variant_id: Variant ID
            metric: Metric name
            value: Metric value
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return
        
        if variant_id not in exp.results:
            exp.results[variant_id] = {}
        if metric not in exp.results[variant_id]:
            exp.results[variant_id][metric] = []
        
        exp.results[variant_id][metric].append(value)
        self._save()
    
    def get_results_summary(self, experiment_id: str) -> Dict:
        """
        Get summary statistics for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dictionary with mean, stdev, n per variant per metric
        """
        import statistics
        
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {}
        
        summary = {}
        for variant_id, metrics in exp.results.items():
            summary[variant_id] = {}
            for metric, values in metrics.items():
                if len(values) > 1:
                    summary[variant_id][metric] = {
                        "mean": statistics.mean(values),
                        "stdev": statistics.stdev(values),
                        "n": len(values),
                    }
                elif values:
                    summary[variant_id][metric] = {"mean": values[0], "n": 1}
        
        return summary
    
    def check_significance(
        self,
        experiment_id: str,
        metric: str,
        variant_a: str,
        variant_b: str,
        alpha: float = 0.05,
    ) -> Dict:
        """
        Two-sample t-test for statistical significance.
        
        Args:
            experiment_id: Experiment ID
            metric: Metric to compare
            variant_a: First variant ID
            variant_b: Second variant ID
            alpha: Significance level
            
        Returns:
            Dictionary with t_statistic, p_value, significant, etc.
        """
        try:
            from scipy import stats
        except ImportError:
            return {"error": "scipy required for significance testing"}
        
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {"error": "Experiment not found"}
        
        a_vals = exp.results.get(variant_a, {}).get(metric, [])
        b_vals = exp.results.get(variant_b, {}).get(metric, [])
        
        if len(a_vals) < 2 or len(b_vals) < 2:
            return {"error": "Not enough data for significance test"}
        
        t_stat, p_value = stats.ttest_ind(a_vals, b_vals)
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "alpha": alpha,
            "n_a": len(a_vals),
            "n_b": len(b_vals),
        }
    
    def pause_experiment(self, experiment_id: str) -> None:
        """Pause an experiment."""
        if experiment_id in self._experiments:
            self._experiments[experiment_id].status = "paused"
            self._save()
    
    def conclude_experiment(self, experiment_id: str) -> None:
        """Conclude an experiment."""
        if experiment_id in self._experiments:
            self._experiments[experiment_id].status = "concluded"
            self._save()
    
    def list_experiments(self, status: Optional[str] = None) -> List[str]:
        """List experiment IDs, optionally filtered by status."""
        if status:
            return [eid for eid, exp in self._experiments.items() if exp.status == status]
        return list(self._experiments.keys())
    
    def _get_active(self, experiment_id: str) -> Experiment:
        """Get an active experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment '{experiment_id}' not found")
        if exp.status != "active":
            raise RuntimeError(f"Experiment '{experiment_id}' is {exp.status}, not active")
        return exp
    
    def _load(self):
        """Load from storage."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    raw = json.load(f)
                for eid, edata in raw.items():
                    variants = [ExperimentVariant(**v) for v in edata.pop("variants")]
                    self._experiments[eid] = Experiment(variants=variants, **edata)
            except (json.JSONDecodeError, IOError):
                self._experiments = {}
    
    def _save(self):
        """Save to storage."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out = {}
        for eid, exp in self._experiments.items():
            d = asdict(exp)
            out[eid] = d
        with open(self._path, "w") as f:
            json.dump(out, f, indent=2, default=str)
