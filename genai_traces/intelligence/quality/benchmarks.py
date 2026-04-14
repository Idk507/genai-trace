"""
Benchmark comparisons for GenAI-Traces.

Provides golden dataset comparisons and benchmark running.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    id: str
    prompt: str
    expected_response: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of running a benchmark case."""
    case_id: str
    actual_response: str
    scores: Dict[str, float] = field(default_factory=dict)
    passed: bool = True
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class Benchmark:
    """
    A benchmark dataset for evaluation.
    
    Attributes:
        name: Name of the benchmark
        description: Description of what this benchmark tests
        cases: List of benchmark cases
    """
    name: str
    description: str = ""
    cases: List[BenchmarkCase] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_case(
        self,
        id: str,
        prompt: str,
        expected_response: Optional[str] = None,
        context: Optional[str] = None,
        **metadata,
    ) -> None:
        """Add a test case to the benchmark."""
        case = BenchmarkCase(
            id=id,
            prompt=prompt,
            expected_response=expected_response,
            context=context,
            metadata=metadata,
        )
        self.cases.append(case)
    
    def get_case(self, case_id: str) -> Optional[BenchmarkCase]:
        """Get a case by ID."""
        for case in self.cases:
            if case.id == case_id:
                return case
        return None
    
    def filter_by_tag(self, tag: str) -> List[BenchmarkCase]:
        """Get cases with a specific tag."""
        return [c for c in self.cases if tag in c.tags]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "cases": [
                {
                    "id": c.id,
                    "prompt": c.prompt,
                    "expected_response": c.expected_response,
                    "context": c.context,
                    "metadata": c.metadata,
                    "tags": c.tags,
                }
                for c in self.cases
            ],
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Benchmark":
        """Create from dictionary."""
        benchmark = cls(
            name=data["name"],
            description=data.get("description", ""),
        )
        
        for case_data in data.get("cases", []):
            case = BenchmarkCase(
                id=case_data["id"],
                prompt=case_data["prompt"],
                expected_response=case_data.get("expected_response"),
                context=case_data.get("context"),
                metadata=case_data.get("metadata", {}),
                tags=case_data.get("tags", []),
            )
            benchmark.cases.append(case)
        
        return benchmark
    
    def save(self, path: str) -> None:
        """Save benchmark to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Benchmark":
        """Load benchmark from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class BenchmarkReport:
    """Report from running a benchmark."""
    benchmark_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    average_scores: Dict[str, float]
    results: List[BenchmarkResult]
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "pass_rate": self.pass_rate,
            "average_scores": self.average_scores,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class BenchmarkRunner:
    """
    Runs benchmarks against an LLM.
    
    Usage:
        runner = BenchmarkRunner(llm_fn=my_llm_call)
        runner.add_scorer(quality_scorer)
        
        report = runner.run(benchmark)
        print(f"Pass rate: {report.pass_rate:.2%}")
    """
    
    def __init__(
        self,
        llm_fn: Callable[[str, Optional[str]], str],
        pass_threshold: float = 0.7,
    ):
        """
        Initialize the runner.
        
        Args:
            llm_fn: Function that takes (prompt, context) and returns response
            pass_threshold: Minimum score to pass (default 0.7)
        """
        self._llm_fn = llm_fn
        self._pass_threshold = pass_threshold
        self._scorers: List[Any] = []
    
    def add_scorer(self, scorer: Any) -> None:
        """Add a quality scorer."""
        self._scorers.append(scorer)
    
    def run(
        self,
        benchmark: Benchmark,
        tags: Optional[List[str]] = None,
    ) -> BenchmarkReport:
        """
        Run a benchmark.
        
        Args:
            benchmark: The benchmark to run
            tags: Optional tags to filter cases
            
        Returns:
            BenchmarkReport with results
        """
        import time
        
        cases = benchmark.cases
        if tags:
            cases = [c for c in cases if any(t in c.tags for t in tags)]
        
        results = []
        all_scores: Dict[str, List[float]] = {}
        
        start_time = time.perf_counter()
        
        for case in cases:
            result = self._run_case(case)
            results.append(result)
            
            for dim, score in result.scores.items():
                if dim not in all_scores:
                    all_scores[dim] = []
                all_scores[dim].append(score)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        avg_scores = {
            dim: sum(scores) / len(scores)
            for dim, scores in all_scores.items()
            if scores
        }
        
        return BenchmarkReport(
            benchmark_name=benchmark.name,
            total_cases=len(results),
            passed_cases=passed,
            failed_cases=failed,
            average_scores=avg_scores,
            results=results,
            duration_ms=duration_ms,
        )
    
    def _run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run a single benchmark case."""
        import time
        
        start_time = time.perf_counter()
        
        try:
            response = self._llm_fn(case.prompt, case.context)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            scores = {}
            for scorer in self._scorers:
                score_result = scorer.score(
                    prompt=case.prompt,
                    response=response,
                    context=case.context,
                    reference=case.expected_response,
                )
                
                if hasattr(score_result, "overall_score"):
                    scores["overall"] = score_result.overall_score
                    scores.update(score_result.dimension_scores)
                elif isinstance(score_result, dict):
                    scores.update(score_result)
            
            overall = scores.get("overall", sum(scores.values()) / len(scores) if scores else 0)
            passed = overall >= self._pass_threshold
            
            return BenchmarkResult(
                case_id=case.id,
                actual_response=response,
                scores=scores,
                passed=passed,
                duration_ms=duration_ms,
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                case_id=case.id,
                actual_response="",
                scores={},
                passed=False,
                duration_ms=duration_ms,
                error=str(e),
            )


def create_golden_dataset(
    name: str,
    cases: List[Dict[str, Any]],
    description: str = "",
) -> Benchmark:
    """
    Create a golden dataset benchmark from a list of cases.
    
    Args:
        name: Name of the benchmark
        cases: List of dicts with 'prompt', 'expected_response', etc.
        description: Optional description
        
    Returns:
        Benchmark instance
    """
    benchmark = Benchmark(name=name, description=description)
    
    for i, case in enumerate(cases):
        benchmark.add_case(
            id=case.get("id", f"case_{i}"),
            prompt=case["prompt"],
            expected_response=case.get("expected_response"),
            context=case.get("context"),
            **case.get("metadata", {}),
        )
    
    return benchmark
