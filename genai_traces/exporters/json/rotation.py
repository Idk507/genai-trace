"""
File rotation for JSON exporters in GenAI-Traces.

Provides daily, hourly, and size-based rotation.
"""

import os
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum


class RotationStrategy(Enum):
    """File rotation strategies."""
    DAILY = "daily"
    HOURLY = "hourly"
    SIZE = "size"
    NONE = "none"


@dataclass
class RotationConfig:
    """Configuration for file rotation."""
    strategy: RotationStrategy = RotationStrategy.DAILY
    max_size_mb: float = 100.0
    max_files: int = 30
    compress_rotated: bool = True
    
    def to_dict(self):
        return {
            "strategy": self.strategy.value,
            "max_size_mb": self.max_size_mb,
            "max_files": self.max_files,
            "compress_rotated": self.compress_rotated,
        }


class FileRotator:
    """
    Handles file rotation for trace exports.
    
    Usage:
        rotator = FileRotator(
            base_path="./traces/traces.jsonl",
            config=RotationConfig(strategy=RotationStrategy.DAILY),
        )
        
        # Get current file path (may rotate)
        current_file = rotator.get_current_file()
        
        # Check if rotation needed
        if rotator.should_rotate():
            rotator.rotate()
    """
    
    def __init__(
        self,
        base_path: str,
        config: Optional[RotationConfig] = None,
    ):
        self._base_path = Path(base_path)
        self._config = config or RotationConfig()
        self._current_file: Optional[Path] = None
        self._current_period: Optional[str] = None
        self._bytes_written = 0
    
    def get_current_file(self) -> Path:
        """Get the current file path, rotating if necessary."""
        if self.should_rotate():
            self.rotate()
        
        if self._current_file is None:
            self._current_file = self._generate_filename()
            self._current_period = self._get_current_period()
            self._bytes_written = 0
            
            if self._current_file.exists():
                self._bytes_written = self._current_file.stat().st_size
        
        return self._current_file
    
    def should_rotate(self) -> bool:
        """Check if rotation is needed."""
        if self._config.strategy == RotationStrategy.NONE:
            return False
        
        if self._current_file is None:
            return False
        
        if self._config.strategy == RotationStrategy.SIZE:
            max_bytes = self._config.max_size_mb * 1024 * 1024
            return self._bytes_written >= max_bytes
        
        current_period = self._get_current_period()
        return current_period != self._current_period
    
    def rotate(self) -> Optional[Path]:
        """Perform rotation and return the rotated file path."""
        if self._current_file is None or not self._current_file.exists():
            self._current_file = None
            self._current_period = None
            self._bytes_written = 0
            return None
        
        rotated_path = self._current_file
        
        if self._config.compress_rotated:
            rotated_path = self._compress_file(self._current_file)
        
        self._current_file = self._generate_filename()
        self._current_period = self._get_current_period()
        self._bytes_written = 0
        
        self._cleanup_old_files()
        
        return rotated_path
    
    def record_write(self, bytes_written: int) -> None:
        """Record bytes written to current file."""
        self._bytes_written += bytes_written
    
    def _generate_filename(self) -> Path:
        """Generate filename based on rotation strategy."""
        base = self._base_path
        stem = base.stem
        suffix = base.suffix
        parent = base.parent
        
        parent.mkdir(parents=True, exist_ok=True)
        
        if self._config.strategy == RotationStrategy.NONE:
            return base
        
        timestamp = datetime.now()
        
        if self._config.strategy == RotationStrategy.DAILY:
            date_str = timestamp.strftime("%Y-%m-%d")
            return parent / f"{stem}-{date_str}{suffix}"
        
        elif self._config.strategy == RotationStrategy.HOURLY:
            datetime_str = timestamp.strftime("%Y-%m-%d-%H")
            return parent / f"{stem}-{datetime_str}{suffix}"
        
        elif self._config.strategy == RotationStrategy.SIZE:
            datetime_str = timestamp.strftime("%Y%m%d-%H%M%S")
            return parent / f"{stem}-{datetime_str}{suffix}"
        
        return base
    
    def _get_current_period(self) -> str:
        """Get current time period for rotation check."""
        now = datetime.now()
        
        if self._config.strategy == RotationStrategy.DAILY:
            return now.strftime("%Y-%m-%d")
        elif self._config.strategy == RotationStrategy.HOURLY:
            return now.strftime("%Y-%m-%d-%H")
        else:
            return ""
    
    def _compress_file(self, file_path: Path) -> Path:
        """Compress a file with gzip."""
        gz_path = file_path.with_suffix(file_path.suffix + ".gz")
        
        with open(file_path, "rb") as f_in:
            with gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        file_path.unlink()
        
        return gz_path
    
    def _cleanup_old_files(self) -> None:
        """Remove old rotated files beyond max_files limit."""
        parent = self._base_path.parent
        stem = self._base_path.stem
        
        pattern = f"{stem}-*"
        files = sorted(parent.glob(pattern), key=lambda f: f.stat().st_mtime)
        
        while len(files) > self._config.max_files:
            oldest = files.pop(0)
            oldest.unlink()
    
    def list_rotated_files(self) -> List[Path]:
        """List all rotated files."""
        parent = self._base_path.parent
        stem = self._base_path.stem
        
        if not parent.exists():
            return []
        
        pattern = f"{stem}-*"
        return sorted(parent.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)


def create_rotator(
    base_path: str,
    strategy: str = "daily",
    max_size_mb: float = 100.0,
    max_files: int = 30,
    compress: bool = True,
) -> FileRotator:
    """
    Create a file rotator with the given configuration.
    
    Args:
        base_path: Base path for trace files
        strategy: Rotation strategy ("daily", "hourly", "size", "none")
        max_size_mb: Max file size for size-based rotation
        max_files: Max number of rotated files to keep
        compress: Whether to compress rotated files
        
    Returns:
        Configured FileRotator
    """
    config = RotationConfig(
        strategy=RotationStrategy(strategy),
        max_size_mb=max_size_mb,
        max_files=max_files,
        compress_rotated=compress,
    )
    
    return FileRotator(base_path, config)
