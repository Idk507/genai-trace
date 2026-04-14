"""
Data models for test results.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class ResultStatus(Enum):
    """Status of a test result."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class TestResult:
    """Individual test result."""
    test_id: str
    name: str
    status: ResultStatus
    module: str
    functionality: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status.value,
            "module": self.module,
            "functionality": self.functionality,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "message": self.message,
            "details": self.details,
            "error": self.error,
        }
    
    def to_csv_row(self) -> Dict[str, Any]:
        """Convert to flat CSV row."""
        row = {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status.value,
            "module": self.module,
            "functionality": self.functionality,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms or "",
            "message": self.message or "",
            "error": self.error or "",
        }
        # Flatten details
        for key, value in self.details.items():
            if isinstance(value, (dict, list)):
                row[f"detail.{key}"] = str(value)
            else:
                row[f"detail.{key}"] = value
        return row
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestResult":
        """Create from dictionary."""
        return cls(
            test_id=data["test_id"],
            name=data["name"],
            status=ResultStatus(data["status"]),
            module=data["module"],
            functionality=data["functionality"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            duration_ms=data.get("duration_ms"),
            message=data.get("message"),
            details=data.get("details", {}),
            error=data.get("error"),
        )


@dataclass
class ModuleResult:
    """Aggregated results for a module."""
    module: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    tests: List[TestResult] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests
    
    def add_result(self, result: TestResult):
        """Add a test result."""
        self.tests.append(result)
        self.total_tests += 1
        if result.status == ResultStatus.PASS:
            self.passed += 1
        elif result.status == ResultStatus.FAIL:
            self.failed += 1
        elif result.status == ResultStatus.SKIP:
            self.skipped += 1
        elif result.status == ResultStatus.ERROR:
            self.errors += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "module": self.module,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "pass_rate": self.pass_rate,
            "tests": [t.to_dict() for t in self.tests],
        }


@dataclass
class FunctionalityResult:
    """Results grouped by functionality."""
    functionality: str
    description: str = ""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    tests: List[TestResult] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests
    
    def add_result(self, result: TestResult):
        """Add a test result."""
        self.tests.append(result)
        self.total_tests += 1
        if result.status == ResultStatus.PASS:
            self.passed += 1
        elif result.status == ResultStatus.FAIL:
            self.failed += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "functionality": self.functionality,
            "description": self.description,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "tests": [t.to_dict() for t in self.tests],
        }
    
    def to_csv_row(self) -> Dict[str, Any]:
        """Convert to summary CSV row."""
        return {
            "functionality": self.functionality,
            "description": self.description,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": f"{self.pass_rate:.2%}",
        }
