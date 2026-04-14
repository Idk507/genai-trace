"""
Export high-quality production traces as labeled datasets for fine-tuning.

Supports:
- OpenAI JSONL format  ({"messages": [{"role":..., "content":...}]})
- HuggingFace format   ({"prompt": ..., "completion": ...})
- Alpaca format        ({"instruction": ..., "input": ..., "output": ...})
- ShareGPT format      ({"conversations": [{"from": ..., "value": ...}]})

Usage:
    exporter = FineTuneExporter(
        min_quality_score=0.8,
        min_feedback_score=4,
        max_records=10_000,
    )
    dataset = exporter.export_from_spans(spans, output_path="dataset.jsonl")
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Union, Iterator
import gzip


class DatasetFormat(Enum):
    """Supported fine-tuning dataset formats."""
    
    OPENAI = "openai"
    HUGGINGFACE = "hf"
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"


@dataclass
class FineTuneRecord:
    """Represents a single fine-tuning example."""
    
    prompt: str
    completion: str
    quality: float
    source_trace_id: str
    source_span_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    system_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    
    feedback_score: Optional[int] = None
    feedback_comment: Optional[str] = None
    
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    
    timestamp: Optional[str] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI fine-tuning JSONL format."""
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        if self.messages:
            messages.extend(self.messages)
        else:
            messages.append({"role": "user", "content": self.prompt})
            messages.append({"role": "assistant", "content": self.completion})
        
        return {"messages": messages}
    
    def to_hf_format(self) -> Dict[str, Any]:
        """Convert to HuggingFace format."""
        result = {
            "prompt": self.prompt,
            "completion": self.completion,
        }
        if self.system_prompt:
            result["system"] = self.system_prompt
        return result
    
    def to_alpaca_format(self) -> Dict[str, Any]:
        """Convert to Alpaca format."""
        return {
            "instruction": self.prompt,
            "input": "",
            "output": self.completion,
        }
    
    def to_sharegpt_format(self) -> Dict[str, Any]:
        """Convert to ShareGPT format."""
        conversations = []
        
        if self.system_prompt:
            conversations.append({"from": "system", "value": self.system_prompt})
        
        if self.messages:
            for msg in self.messages:
                role = msg.get("role", "user")
                from_value = "human" if role == "user" else "gpt" if role == "assistant" else role
                conversations.append({"from": from_value, "value": msg.get("content", "")})
        else:
            conversations.append({"from": "human", "value": self.prompt})
            conversations.append({"from": "gpt", "value": self.completion})
        
        return {"conversations": conversations}
    
    def to_dict(self, format: DatasetFormat = DatasetFormat.OPENAI) -> Dict[str, Any]:
        """Convert to specified format."""
        if format == DatasetFormat.OPENAI:
            return self.to_openai_format()
        elif format == DatasetFormat.HUGGINGFACE:
            return self.to_hf_format()
        elif format == DatasetFormat.ALPACA:
            return self.to_alpaca_format()
        elif format == DatasetFormat.SHAREGPT:
            return self.to_sharegpt_format()
        else:
            return self.to_hf_format()


class FineTuneExporter:
    """
    Export high-quality production traces as fine-tuning datasets.
    
    Features:
    - Quality filtering based on eval scores and feedback
    - Deduplication by prompt hash
    - Multiple output formats (OpenAI, HuggingFace, Alpaca, ShareGPT)
    - Custom filtering functions
    - Gzip compression support
    - Streaming export for large datasets
    """
    
    def __init__(
        self,
        min_quality_score: float = 0.7,
        min_feedback_score: int = 4,
        max_records: int = 50_000,
        dedup: bool = True,
        format: Union[str, DatasetFormat] = DatasetFormat.OPENAI,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        include_metadata: bool = False,
        compress: bool = False,
    ):
        """
        Initialize the fine-tuning exporter.
        
        Args:
            min_quality_score: Minimum quality score (0-1) for inclusion
            min_feedback_score: Minimum feedback score (1-5) for inclusion
            max_records: Maximum number of records to export
            dedup: Whether to deduplicate by prompt hash
            format: Output format (openai, hf, alpaca, sharegpt)
            filter_fn: Custom filter function that takes a span dict
            include_metadata: Whether to include metadata in output
            compress: Whether to gzip compress the output
        """
        self.min_quality = min_quality_score
        self.min_feedback = min_feedback_score
        self.max_records = max_records
        self.dedup = dedup
        self.format = DatasetFormat(format) if isinstance(format, str) else format
        self.filter_fn = filter_fn
        self.include_metadata = include_metadata
        self.compress = compress
        
        self._stats = {
            "total_spans": 0,
            "filtered_no_prompt": 0,
            "filtered_no_completion": 0,
            "filtered_low_quality": 0,
            "filtered_low_feedback": 0,
            "filtered_custom": 0,
            "filtered_duplicate": 0,
            "exported": 0,
        }
    
    def export_from_spans(
        self,
        spans: List[Dict[str, Any]],
        output_path: str,
    ) -> int:
        """
        Filter and convert in-memory span dicts to a fine-tuning dataset.
        
        Args:
            spans: List of span dictionaries
            output_path: Path to write the output file
        
        Returns:
            Number of records written
        """
        self._reset_stats()
        records = list(self._process_spans(spans))
        return self._write_records(records, output_path)
    
    def export_from_jsonl(
        self,
        input_path: str,
        output_path: str,
    ) -> int:
        """
        Filter and convert spans from a JSONL file.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to write the output file
        
        Returns:
            Number of records written
        """
        self._reset_stats()
        
        def read_spans():
            open_fn = gzip.open if input_path.endswith('.gz') else open
            with open_fn(input_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        
        records = list(self._process_spans(read_spans()))
        return self._write_records(records, output_path)
    
    def _reset_stats(self):
        """Reset export statistics."""
        for key in self._stats:
            self._stats[key] = 0
    
    def _process_spans(
        self,
        spans: Iterator[Dict[str, Any]],
    ) -> Iterator[FineTuneRecord]:
        """Process spans and yield valid fine-tuning records."""
        seen_hashes = set()
        
        for span in spans:
            self._stats["total_spans"] += 1
            
            if self._stats["exported"] >= self.max_records:
                break
            
            record = self._extract_record(span)
            if record is None:
                continue
            
            if self.dedup:
                prompt_hash = hashlib.md5(record.prompt.encode()).hexdigest()
                if prompt_hash in seen_hashes:
                    self._stats["filtered_duplicate"] += 1
                    continue
                seen_hashes.add(prompt_hash)
            
            self._stats["exported"] += 1
            yield record
    
    def _extract_record(self, span: Dict[str, Any]) -> Optional[FineTuneRecord]:
        """Extract a fine-tuning record from a span dict."""
        attrs = span.get("attributes", {})
        
        prompt = attrs.get("llm.prompt", "")
        completion = attrs.get("llm.completion", "")
        
        if not prompt:
            messages = attrs.get("llm.messages", [])
            if messages:
                user_messages = [m for m in messages if m.get("role") == "user"]
                if user_messages:
                    prompt = user_messages[-1].get("content", "")
        
        if not prompt:
            self._stats["filtered_no_prompt"] += 1
            return None
        
        if not completion:
            self._stats["filtered_no_completion"] += 1
            return None
        
        quality = attrs.get("eval.quality", 0.0)
        if quality < self.min_quality:
            relevance = attrs.get("eval.relevance", 0.0)
            groundedness = attrs.get("eval.groundedness", 0.0)
            quality = max(quality, relevance, groundedness)
        
        if quality < self.min_quality:
            self._stats["filtered_low_quality"] += 1
            return None
        
        feedback = attrs.get("feedback.score")
        if feedback is not None and int(feedback) < self.min_feedback:
            self._stats["filtered_low_feedback"] += 1
            return None
        
        if self.filter_fn and not self.filter_fn(span):
            self._stats["filtered_custom"] += 1
            return None
        
        messages = attrs.get("llm.messages", [])
        system_prompt = attrs.get("llm.system_prompt")
        
        if not system_prompt and messages:
            system_msgs = [m for m in messages if m.get("role") == "system"]
            if system_msgs:
                system_prompt = system_msgs[0].get("content")
        
        return FineTuneRecord(
            prompt=prompt,
            completion=completion,
            quality=quality,
            source_trace_id=span.get("trace_id", ""),
            source_span_id=span.get("span_id", ""),
            system_prompt=system_prompt,
            messages=messages if messages else None,
            feedback_score=int(feedback) if feedback else None,
            feedback_comment=attrs.get("feedback.comment"),
            model=attrs.get("llm.model.name"),
            prompt_tokens=attrs.get("llm.prompt.tokens"),
            completion_tokens=attrs.get("llm.completion.tokens"),
            timestamp=span.get("start_time"),
            metadata={
                "model": attrs.get("llm.model.name"),
                "tokens": attrs.get("llm.total_tokens"),
                "cost_usd": attrs.get("cost.total_usd"),
                "duration_ms": attrs.get("llm.duration_ms"),
            } if self.include_metadata else {},
        )
    
    def _write_records(
        self,
        records: List[FineTuneRecord],
        output_path: str,
    ) -> int:
        """Write records to output file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if self.compress or output_path.endswith('.gz'):
            open_fn = lambda p: gzip.open(p, 'wt', encoding='utf-8')
        else:
            open_fn = lambda p: open(p, 'w', encoding='utf-8')
        
        with open_fn(output) as f:
            for record in records:
                data = record.to_dict(self.format)
                
                if self.include_metadata and record.metadata:
                    data["_metadata"] = record.metadata
                    data["_source"] = {
                        "trace_id": record.source_trace_id,
                        "span_id": record.source_span_id,
                    }
                
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        return len(records)
    
    def _format_record(self, record: FineTuneRecord) -> Dict[str, Any]:
        """Format a record according to the configured format."""
        return record.to_dict(self.format)
    
    def get_stats(self) -> Dict[str, int]:
        """Get export statistics."""
        return self._stats.copy()
    
    def validate_dataset(
        self,
        path: str,
    ) -> Dict[str, Any]:
        """
        Validate an exported dataset file.
        
        Args:
            path: Path to the dataset file
        
        Returns:
            Validation results including counts and any issues
        """
        issues = []
        record_count = 0
        empty_prompts = 0
        empty_completions = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        open_fn = gzip.open if path.endswith('.gz') else open
        
        try:
            with open_fn(path, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        record_count += 1
                        
                        if self.format == DatasetFormat.OPENAI:
                            messages = data.get("messages", [])
                            for msg in messages:
                                content = msg.get("content", "")
                                if msg.get("role") == "user":
                                    total_prompt_tokens += len(content.split())
                                    if not content:
                                        empty_prompts += 1
                                elif msg.get("role") == "assistant":
                                    total_completion_tokens += len(content.split())
                                    if not content:
                                        empty_completions += 1
                        else:
                            prompt = data.get("prompt") or data.get("instruction", "")
                            completion = data.get("completion") or data.get("output", "")
                            
                            if not prompt:
                                empty_prompts += 1
                            else:
                                total_prompt_tokens += len(prompt.split())
                            
                            if not completion:
                                empty_completions += 1
                            else:
                                total_completion_tokens += len(completion.split())
                    
                    except json.JSONDecodeError as e:
                        issues.append(f"Line {i}: Invalid JSON - {e}")
        
        except Exception as e:
            issues.append(f"File error: {e}")
        
        return {
            "valid": len(issues) == 0,
            "record_count": record_count,
            "empty_prompts": empty_prompts,
            "empty_completions": empty_completions,
            "avg_prompt_words": total_prompt_tokens / record_count if record_count > 0 else 0,
            "avg_completion_words": total_completion_tokens / record_count if record_count > 0 else 0,
            "issues": issues,
        }
    
    @staticmethod
    def split_dataset(
        input_path: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Dict[str, str]:
        """
        Split a dataset into train/val/test sets.
        
        Args:
            input_path: Path to input dataset
            output_dir: Directory to write split files
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility
        
        Returns:
            Dict with paths to train, val, test files
        """
        import random
        
        records = []
        open_fn = gzip.open if input_path.endswith('.gz') else open
        
        with open_fn(input_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(line)
        
        if shuffle:
            random.seed(seed)
            random.shuffle(records)
        
        total = len(records)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_records = records[:train_end]
        val_records = records[train_end:val_end]
        test_records = records[val_end:]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        suffix = ".jsonl"
        paths = {}
        
        for name, data in [("train", train_records), ("val", val_records), ("test", test_records)]:
            if data:
                file_path = output_path / f"{name}{suffix}"
                with open(file_path, 'w', encoding='utf-8') as f:
                    for record in data:
                        f.write(record + "\n")
                paths[name] = str(file_path)
        
        return paths
    
    @staticmethod
    def merge_datasets(
        input_paths: List[str],
        output_path: str,
        dedup: bool = True,
    ) -> int:
        """
        Merge multiple dataset files into one.
        
        Args:
            input_paths: List of input file paths
            output_paths: Output file path
            dedup: Whether to deduplicate
        
        Returns:
            Number of records in merged file
        """
        seen_hashes = set()
        count = 0
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for input_path in input_paths:
                open_fn = gzip.open if input_path.endswith('.gz') else open
                
                with open_fn(input_path, 'rt', encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        if dedup:
                            line_hash = hashlib.md5(line.encode()).hexdigest()
                            if line_hash in seen_hashes:
                                continue
                            seen_hashes.add(line_hash)
                        
                        out_f.write(line + "\n")
                        count += 1
        
        return count
