"""
Quality filtering for fine-tuning datasets.

Filters traces based on quality criteria before export.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class FilterCriteria:
    """Criteria for filtering traces."""
    min_prompt_length: int = 10
    max_prompt_length: int = 10000
    min_completion_length: int = 10
    max_completion_length: int = 10000
    min_quality_score: float = 0.0
    require_positive_feedback: bool = False
    exclude_errors: bool = True
    exclude_pii: bool = True
    custom_filters: List[Callable[[Dict], bool]] = field(default_factory=list)


@dataclass
class FilterResult:
    """Result of filtering operation."""
    passed: List[Dict[str, Any]]
    filtered: List[Dict[str, Any]]
    filter_reasons: Dict[str, int]
    
    @property
    def pass_rate(self) -> float:
        total = len(self.passed) + len(self.filtered)
        return len(self.passed) / total if total > 0 else 0.0


class QualityFilter:
    """
    Filters traces for fine-tuning quality.
    
    Usage:
        filter = QualityFilter(FilterCriteria(
            min_prompt_length=20,
            min_quality_score=0.7,
        ))
        
        result = filter.filter(traces)
        print(f"Pass rate: {result.pass_rate:.2%}")
    """
    
    def __init__(self, criteria: Optional[FilterCriteria] = None):
        self._criteria = criteria or FilterCriteria()
    
    def filter(self, traces: List[Dict[str, Any]]) -> FilterResult:
        """
        Filter traces based on criteria.
        
        Args:
            traces: List of trace dictionaries
            
        Returns:
            FilterResult with passed and filtered traces
        """
        passed = []
        filtered = []
        reasons: Dict[str, int] = {}
        
        for trace in traces:
            is_valid, reason = self._check_trace(trace)
            
            if is_valid:
                passed.append(trace)
            else:
                filtered.append(trace)
                reasons[reason] = reasons.get(reason, 0) + 1
        
        return FilterResult(
            passed=passed,
            filtered=filtered,
            filter_reasons=reasons,
        )
    
    def _check_trace(self, trace: Dict[str, Any]) -> tuple:
        """Check if a trace passes all criteria."""
        attrs = trace.get("attributes", {})
        
        if self._criteria.exclude_errors:
            status = trace.get("status", "")
            if status == "ERROR" or "error" in trace:
                return False, "error_status"
        
        prompt = attrs.get("llm.prompt", "")
        if len(prompt) < self._criteria.min_prompt_length:
            return False, "prompt_too_short"
        if len(prompt) > self._criteria.max_prompt_length:
            return False, "prompt_too_long"
        
        completion = attrs.get("llm.completion", "")
        if len(completion) < self._criteria.min_completion_length:
            return False, "completion_too_short"
        if len(completion) > self._criteria.max_completion_length:
            return False, "completion_too_long"
        
        quality_score = attrs.get("quality.score", 1.0)
        if quality_score < self._criteria.min_quality_score:
            return False, "low_quality_score"
        
        if self._criteria.require_positive_feedback:
            feedback = attrs.get("feedback.positive", None)
            if feedback is not True:
                return False, "no_positive_feedback"
        
        if self._criteria.exclude_pii:
            has_pii = attrs.get("privacy.has_pii", False)
            if has_pii:
                return False, "contains_pii"
        
        for custom_filter in self._criteria.custom_filters:
            if not custom_filter(trace):
                return False, "custom_filter"
        
        return True, ""
    
    def add_custom_filter(self, filter_fn: Callable[[Dict], bool]) -> None:
        """Add a custom filter function."""
        self._criteria.custom_filters.append(filter_fn)


class DiversityFilter:
    """
    Filters for diversity in the dataset.
    
    Ensures the dataset has diverse examples.
    """
    
    def __init__(
        self,
        max_similar: int = 5,
        similarity_threshold: float = 0.9,
    ):
        self._max_similar = max_similar
        self._similarity_threshold = similarity_threshold
    
    def filter(self, traces: List[Dict[str, Any]]) -> FilterResult:
        """Filter for diversity."""
        passed = []
        filtered = []
        seen_prompts: List[str] = []
        
        for trace in traces:
            attrs = trace.get("attributes", {})
            prompt = attrs.get("llm.prompt", "")
            
            similar_count = sum(
                1 for seen in seen_prompts
                if self._is_similar(prompt, seen)
            )
            
            if similar_count < self._max_similar:
                passed.append(trace)
                seen_prompts.append(prompt)
            else:
                filtered.append(trace)
        
        return FilterResult(
            passed=passed,
            filtered=filtered,
            filter_reasons={"too_similar": len(filtered)},
        )
    
    def _is_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union >= self._similarity_threshold


class FilterPipeline:
    """
    Pipeline of filters for fine-tuning data.
    
    Usage:
        pipeline = FilterPipeline()
        pipeline.add(QualityFilter(criteria))
        pipeline.add(DiversityFilter())
        
        result = pipeline.run(traces)
    """
    
    def __init__(self):
        self._filters: List[Any] = []
    
    def add(self, filter_obj: Any) -> "FilterPipeline":
        """Add a filter to the pipeline."""
        self._filters.append(filter_obj)
        return self
    
    def run(self, traces: List[Dict[str, Any]]) -> FilterResult:
        """Run all filters in sequence."""
        current = traces
        all_filtered = []
        all_reasons: Dict[str, int] = {}
        
        for filter_obj in self._filters:
            result = filter_obj.filter(current)
            current = result.passed
            all_filtered.extend(result.filtered)
            
            for reason, count in result.filter_reasons.items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count
        
        return FilterResult(
            passed=current,
            filtered=all_filtered,
            filter_reasons=all_reasons,
        )
