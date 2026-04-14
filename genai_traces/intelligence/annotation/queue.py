"""
Priority-based annotation queue for human review.

Low-scoring spans get surfaced for human review.
Annotations feed back into evaluation datasets and future fine-tuning.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...core.span import Span


@dataclass
class AnnotationItem:
    """An item in the annotation queue."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""
    span_id: str = ""
    prompt: str = ""
    completion: str = ""
    priority: str = "normal"
    status: str = "pending"
    annotation: Optional[Dict] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model: str = ""
    eval_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnnotationQueue:
    """
    Priority-based queue for human annotation of LLM outputs.
    """
    
    PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
    
    def __init__(self, storage_path: str = "./annotation_queue.json"):
        """
        Initialize the annotation queue.
        
        Args:
            storage_path: Path to persist queue data
        """
        self._path = Path(storage_path)
        self._items: Dict[str, AnnotationItem] = {}
        self._load()
    
    def enqueue(
        self,
        span: Span,
        priority: str = "normal",
        metadata: Optional[Dict] = None,
    ) -> AnnotationItem:
        """
        Add a span to the annotation queue.
        
        Args:
            span: The span to queue for annotation
            priority: Priority level (urgent, high, normal, low)
            metadata: Additional metadata
            
        Returns:
            The created AnnotationItem
        """
        item = AnnotationItem(
            trace_id=span.trace_id,
            span_id=span.span_id,
            prompt=span.get_attribute("llm.prompt") or "",
            completion=span.get_attribute("llm.completion") or "",
            priority=priority,
            model=span.get_attribute("llm.model.name") or "",
            eval_score=span.get_attribute("eval.quality") or 0.0,
            metadata=metadata or {},
        )
        
        self._items[item.id] = item
        self._save()
        return item
    
    def enqueue_raw(
        self,
        prompt: str,
        completion: str,
        priority: str = "normal",
        trace_id: str = "",
        model: str = "",
        eval_score: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> AnnotationItem:
        """
        Add raw prompt/completion to the queue.
        
        Args:
            prompt: The prompt text
            completion: The completion text
            priority: Priority level
            trace_id: Optional trace ID
            model: Model name
            eval_score: Evaluation score
            metadata: Additional metadata
            
        Returns:
            The created AnnotationItem
        """
        item = AnnotationItem(
            trace_id=trace_id,
            prompt=prompt,
            completion=completion,
            priority=priority,
            model=model,
            eval_score=eval_score,
            metadata=metadata or {},
        )
        
        self._items[item.id] = item
        self._save()
        return item
    
    def next(self, reviewer: Optional[str] = None) -> Optional[AnnotationItem]:
        """
        Get the next highest-priority pending item.
        
        Args:
            reviewer: Optional reviewer identifier
            
        Returns:
            Next item to annotate, or None if queue is empty
        """
        pending = [i for i in self._items.values() if i.status == "pending"]
        if not pending:
            return None
        
        pending.sort(key=lambda x: (
            self.PRIORITY_ORDER.get(x.priority, 99),
            x.created_at
        ))
        
        item = pending[0]
        item.status = "in_review"
        if reviewer:
            item.metadata["reviewer"] = reviewer
            item.metadata["review_started_at"] = datetime.utcnow().isoformat()
        
        self._save()
        return item
    
    def annotate(
        self,
        item_id: str,
        scores: Dict[str, float],
        comment: str = "",
        reviewer: str = "",
        corrected_completion: Optional[str] = None,
    ) -> bool:
        """
        Submit an annotation for an item.
        
        Args:
            item_id: ID of the item to annotate
            scores: Dict of dimension scores (e.g., {"accuracy": 4, "helpfulness": 5})
            comment: Optional reviewer comment
            reviewer: Reviewer identifier
            corrected_completion: Optional corrected response
            
        Returns:
            True if successful
        """
        item = self._items.get(item_id)
        if not item:
            return False
        
        item.annotation = {
            "scores": scores,
            "comment": comment,
            "reviewer": reviewer,
            "timestamp": datetime.utcnow().isoformat(),
            "corrected_completion": corrected_completion,
        }
        item.status = "done"
        
        self._save()
        return True
    
    def skip(self, item_id: str, reason: str = "") -> bool:
        """
        Skip an item without annotating.
        
        Args:
            item_id: ID of the item to skip
            reason: Reason for skipping
            
        Returns:
            True if successful
        """
        item = self._items.get(item_id)
        if not item:
            return False
        
        item.status = "skipped"
        item.metadata["skip_reason"] = reason
        item.metadata["skipped_at"] = datetime.utcnow().isoformat()
        
        self._save()
        return True
    
    def get_item(self, item_id: str) -> Optional[AnnotationItem]:
        """Get an item by ID."""
        return self._items.get(item_id)
    
    def list_items(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 100,
    ) -> List[AnnotationItem]:
        """
        List items with optional filters.
        
        Args:
            status: Filter by status
            priority: Filter by priority
            limit: Maximum items to return
            
        Returns:
            List of matching items
        """
        items = list(self._items.values())
        
        if status:
            items = [i for i in items if i.status == status]
        if priority:
            items = [i for i in items if i.priority == priority]
        
        items.sort(key=lambda x: (
            self.PRIORITY_ORDER.get(x.priority, 99),
            x.created_at
        ))
        
        return items[:limit]
    
    def stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        statuses = [i.status for i in self._items.values()]
        return {
            "pending": statuses.count("pending"),
            "in_review": statuses.count("in_review"),
            "done": statuses.count("done"),
            "skipped": statuses.count("skipped"),
            "total": len(statuses),
        }
    
    def export_dataset(
        self,
        output_path: str,
        include_corrections: bool = True,
    ) -> int:
        """
        Export annotated items as a fine-tuning dataset.
        
        Args:
            output_path: Path to write JSONL file
            include_corrections: Whether to use corrected completions
            
        Returns:
            Number of records exported
        """
        done = [i for i in self._items.values() if i.status == "done" and i.annotation]
        
        with open(output_path, "w") as f:
            for item in done:
                completion = item.completion
                if include_corrections and item.annotation.get("corrected_completion"):
                    completion = item.annotation["corrected_completion"]
                
                record = {
                    "messages": [
                        {"role": "user", "content": item.prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    "metadata": {
                        "scores": item.annotation["scores"],
                        "reviewer": item.annotation["reviewer"],
                        "trace_id": item.trace_id,
                        "eval_score": item.eval_score,
                        "model": item.model,
                    }
                }
                f.write(json.dumps(record) + "\n")
        
        return len(done)
    
    def _load(self) -> None:
        """Load queue from storage."""
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._items = {k: AnnotationItem(**v) for k, v in raw.items()}
            except Exception:
                self._items = {}
    
    def _save(self) -> None:
        """Save queue to storage."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(
            {k: asdict(v) for k, v in self._items.items()},
            indent=2,
            default=str
        ))
