"""Results management for GenAI-Traces."""

from .manager import ResultsManager, ResultsConfig
from .models import TestResult, ModuleResult, FunctionalityResult

__all__ = [
    "ResultsManager",
    "ResultsConfig",
    "TestResult",
    "ModuleResult",
    "FunctionalityResult",
]
