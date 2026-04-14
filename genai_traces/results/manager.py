"""
Results manager for GenAI-Traces.

Handles saving test results to JSON and CSV formats.
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import uuid

from .models import TestResult, ModuleResult, FunctionalityResult, ResultStatus


@dataclass
class ResultsConfig:
    """Configuration for results manager."""
    output_dir: str = "./outputs"
    json_enabled: bool = True
    csv_enabled: bool = True
    csv_separate_by_functionality: bool = True
    csv_separate_by_module: bool = True
    include_summary: bool = True
    timestamp_format: str = "%Y-%m-%d_%H%M%S"
    
    @classmethod
    def from_env(cls) -> "ResultsConfig":
        """Create config from environment variables."""
        return cls(
            output_dir=os.getenv("GENAI_RESULTS_OUTPUT_DIR", "./outputs"),
            json_enabled=os.getenv("GENAI_RESULTS_JSON_ENABLED", "true").lower() == "true",
            csv_enabled=os.getenv("GENAI_RESULTS_CSV_ENABLED", "true").lower() == "true",
            csv_separate_by_functionality=os.getenv("GENAI_RESULTS_CSV_BY_FUNCTIONALITY", "true").lower() == "true",
            csv_separate_by_module=os.getenv("GENAI_RESULTS_CSV_BY_MODULE", "true").lower() == "true",
            include_summary=os.getenv("GENAI_RESULTS_INCLUDE_SUMMARY", "true").lower() == "true",
        )


class ResultsManager:
    """
    Manages test results with JSON and CSV export capabilities.
    
    Features:
    - Save results to JSON (complete data)
    - Save results to CSV (organized by functionality/module)
    - Generate summary reports
    - Query results by various criteria
    
    Usage:
        manager = ResultsManager()
        
        # Record results
        manager.record("test_pii", "pass", module="privacy", functionality="pii_detection",
                      details={"pii_count": 3, "types": ["email", "phone"]})
        
        # Save all results
        manager.save_all()
    """
    
    # Predefined functionality categories
    FUNCTIONALITIES = {
        "core": "Core tracing functionality (tracer, span, decorators)",
        "telemetry": "Telemetry and metrics (tokens, cost, latency)",
        "privacy": "Privacy features (PII detection, redaction, encryption)",
        "security": "Security features (injection detection, guardrails)",
        "intelligence": "Intelligence features (feedback, evaluation)",
        "prompt_management": "Prompt management (registry, versioning, A/B testing)",
        "exporters": "Data exporters (JSON, CSV, database)",
        "api_integration": "API integrations (Azure OpenAI, Gemini, etc.)",
    }
    
    def __init__(self, config: Optional[ResultsConfig] = None):
        """
        Initialize the results manager.
        
        Args:
            config: ResultsConfig instance or None to use defaults/env vars
        """
        self.config = config or ResultsConfig.from_env()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.json_dir = self.output_dir / "json"
        self.csv_dir = self.output_dir / "csv"
        self.json_dir.mkdir(exist_ok=True)
        self.csv_dir.mkdir(exist_ok=True)
        
        # Results storage
        self._results: List[TestResult] = []
        self._by_module: Dict[str, ModuleResult] = {}
        self._by_functionality: Dict[str, FunctionalityResult] = {}
        
        # Session info
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = datetime.utcnow()
        
        # Initialize functionality categories
        for func, desc in self.FUNCTIONALITIES.items():
            self._by_functionality[func] = FunctionalityResult(
                functionality=func,
                description=desc
            )
    
    def record(
        self,
        name: str,
        status: str,
        module: str,
        functionality: str,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        message: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> TestResult:
        """
        Record a test result.
        
        Args:
            name: Test name
            status: Status string (pass, fail, skip, error)
            module: Module name (e.g., "privacy.pii")
            functionality: Functionality category (e.g., "privacy")
            details: Additional details dict
            error: Error message if failed
            message: Optional status message
            duration_ms: Test duration in milliseconds
            
        Returns:
            TestResult instance
        """
        result = TestResult(
            test_id=f"{self.session_id}_{len(self._results)}",
            name=name,
            status=ResultStatus(status),
            module=module,
            functionality=functionality,
            details=details or {},
            error=error,
            message=message,
            duration_ms=duration_ms,
        )
        
        self._results.append(result)
        
        # Update module aggregation
        if module not in self._by_module:
            self._by_module[module] = ModuleResult(module=module)
        self._by_module[module].add_result(result)
        
        # Update functionality aggregation
        if functionality not in self._by_functionality:
            self._by_functionality[functionality] = FunctionalityResult(
                functionality=functionality
            )
        self._by_functionality[functionality].add_result(result)
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary of results."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.status == ResultStatus.PASS)
        failed = sum(1 for r in self._results if r.status == ResultStatus.FAIL)
        skipped = sum(1 for r in self._results if r.status == ResultStatus.SKIP)
        errors = sum(1 for r in self._results if r.status == ResultStatus.ERROR)
        
        return {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0,
            "modules_tested": len(self._by_module),
            "functionalities_tested": sum(1 for f in self._by_functionality.values() if f.total_tests > 0),
        }
    
    def save_json(self, filename: Optional[str] = None) -> Path:
        """
        Save all results to JSON.
        
        Args:
            filename: Optional filename (default: results_{timestamp}.json)
            
        Returns:
            Path to saved file
        """
        if not self.config.json_enabled:
            return None
        
        timestamp = datetime.utcnow().strftime(self.config.timestamp_format)
        filename = filename or f"results_{timestamp}.json"
        filepath = self.json_dir / filename
        
        data = {
            "summary": self.get_summary(),
            "modules": {m: r.to_dict() for m, r in self._by_module.items()},
            "functionalities": {f: r.to_dict() for f, r in self._by_functionality.items() if r.total_tests > 0},
            "results": [r.to_dict() for r in self._results],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def save_csv_by_functionality(self) -> Dict[str, Path]:
        """
        Save results to CSV files organized by functionality.
        
        Returns:
            Dict mapping functionality to file path
        """
        if not self.config.csv_enabled or not self.config.csv_separate_by_functionality:
            return {}
        
        timestamp = datetime.utcnow().strftime(self.config.timestamp_format)
        func_dir = self.csv_dir / "by_functionality"
        func_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        for func_name, func_result in self._by_functionality.items():
            if func_result.total_tests == 0:
                continue
            
            filepath = func_dir / f"{func_name}_{timestamp}.csv"
            
            # Collect all possible columns
            all_columns = set()
            for test in func_result.tests:
                row = test.to_csv_row()
                all_columns.update(row.keys())
            
            # Sort columns for consistent output
            base_cols = ["test_id", "name", "status", "module", "functionality", 
                        "timestamp", "duration_ms", "message", "error"]
            detail_cols = sorted([c for c in all_columns if c.startswith("detail.")])
            columns = base_cols + detail_cols
            
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for test in func_result.tests:
                    writer.writerow(test.to_csv_row())
            
            saved_files[func_name] = filepath
        
        return saved_files
    
    def save_csv_by_module(self) -> Dict[str, Path]:
        """
        Save results to CSV files organized by module.
        
        Returns:
            Dict mapping module to file path
        """
        if not self.config.csv_enabled or not self.config.csv_separate_by_module:
            return {}
        
        timestamp = datetime.utcnow().strftime(self.config.timestamp_format)
        module_dir = self.csv_dir / "by_module"
        module_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        for module_name, module_result in self._by_module.items():
            if module_result.total_tests == 0:
                continue
            
            # Sanitize module name for filename
            safe_name = module_name.replace(".", "_").replace("/", "_")
            filepath = module_dir / f"{safe_name}_{timestamp}.csv"
            
            # Collect all possible columns
            all_columns = set()
            for test in module_result.tests:
                row = test.to_csv_row()
                all_columns.update(row.keys())
            
            base_cols = ["test_id", "name", "status", "module", "functionality",
                        "timestamp", "duration_ms", "message", "error"]
            detail_cols = sorted([c for c in all_columns if c.startswith("detail.")])
            columns = base_cols + detail_cols
            
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for test in module_result.tests:
                    writer.writerow(test.to_csv_row())
            
            saved_files[module_name] = filepath
        
        return saved_files
    
    def save_csv_summary(self) -> Path:
        """
        Save summary CSV with one row per functionality.
        
        Returns:
            Path to saved file
        """
        if not self.config.csv_enabled or not self.config.include_summary:
            return None
        
        timestamp = datetime.utcnow().strftime(self.config.timestamp_format)
        filepath = self.csv_dir / f"summary_{timestamp}.csv"
        
        columns = ["functionality", "description", "total_tests", "passed", "failed", "pass_rate"]
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for func_result in self._by_functionality.values():
                if func_result.total_tests > 0:
                    writer.writerow(func_result.to_csv_row())
        
        return filepath
    
    def save_csv_all_results(self) -> Path:
        """
        Save all results to a single CSV file.
        
        Returns:
            Path to saved file
        """
        if not self.config.csv_enabled:
            return None
        
        timestamp = datetime.utcnow().strftime(self.config.timestamp_format)
        filepath = self.csv_dir / f"all_results_{timestamp}.csv"
        
        # Collect all possible columns
        all_columns = set()
        for test in self._results:
            row = test.to_csv_row()
            all_columns.update(row.keys())
        
        base_cols = ["test_id", "name", "status", "module", "functionality",
                    "timestamp", "duration_ms", "message", "error"]
        detail_cols = sorted([c for c in all_columns if c.startswith("detail.")])
        columns = base_cols + detail_cols
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for test in self._results:
                writer.writerow(test.to_csv_row())
        
        return filepath
    
    def save_all(self) -> Dict[str, Any]:
        """
        Save all results in all configured formats.
        
        Returns:
            Dict with paths to all saved files
        """
        saved = {
            "json": None,
            "csv_all": None,
            "csv_summary": None,
            "csv_by_functionality": {},
            "csv_by_module": {},
        }
        
        if self.config.json_enabled:
            saved["json"] = str(self.save_json())
        
        if self.config.csv_enabled:
            saved["csv_all"] = str(self.save_csv_all_results())
            saved["csv_summary"] = str(self.save_csv_summary())
            saved["csv_by_functionality"] = {k: str(v) for k, v in self.save_csv_by_functionality().items()}
            saved["csv_by_module"] = {k: str(v) for k, v in self.save_csv_by_module().items()}
        
        return saved
    
    def get_results(
        self,
        module: Optional[str] = None,
        functionality: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[TestResult]:
        """
        Query results with optional filters.
        
        Args:
            module: Filter by module name
            functionality: Filter by functionality
            status: Filter by status (pass, fail, skip, error)
            
        Returns:
            List of matching TestResult objects
        """
        results = self._results
        
        if module:
            results = [r for r in results if r.module == module]
        if functionality:
            results = [r for r in results if r.functionality == functionality]
        if status:
            results = [r for r in results if r.status.value == status]
        
        return results
    
    def clear(self):
        """Clear all results."""
        self._results.clear()
        self._by_module.clear()
        self._by_functionality.clear()
        
        # Re-initialize functionality categories
        for func, desc in self.FUNCTIONALITIES.items():
            self._by_functionality[func] = FunctionalityResult(
                functionality=func,
                description=desc
            )
