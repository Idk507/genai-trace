"""
Structured logging for GenAI-Traces.

Uses structlog for structured, contextual logging.
"""

import logging
import sys
from typing import Any, Optional
from datetime import datetime


class StructuredLogger:
    """
    Structured logger with context support.
    """
    
    def __init__(
        self,
        name: str = "genai_traces",
        level: int = logging.INFO,
        json_output: bool = False,
    ):
        self.name = name
        self.level = level
        self.json_output = json_output
        self._context: dict = {}
        
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(level)
            
            if json_output:
                formatter = JsonFormatter()
            else:
                formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                )
            
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
    
    def bind(self, **kwargs) -> "StructuredLogger":
        """
        Create a new logger with bound context.
        
        Args:
            **kwargs: Context to bind
            
        Returns:
            New logger with context
        """
        new_logger = StructuredLogger(
            name=self.name,
            level=self.level,
            json_output=self.json_output,
        )
        new_logger._context = {**self._context, **kwargs}
        new_logger._logger = self._logger
        return new_logger
    
    def _log(self, level: int, msg: str, **kwargs):
        """Internal log method."""
        extra = {**self._context, **kwargs}
        
        if self.json_output:
            import json
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": logging.getLevelName(level),
                "message": msg,
                **extra
            }
            self._logger.log(level, json.dumps(log_data))
        else:
            if extra:
                extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
                msg = f"{msg} | {extra_str}"
            self._logger.log(level, msg)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, msg, **kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log exception with traceback."""
        import traceback
        kwargs["traceback"] = traceback.format_exc()
        self._log(logging.ERROR, msg, **kwargs)


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            import traceback
            log_data["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            )
        
        return json.dumps(log_data)


_default_logger: Optional[StructuredLogger] = None


def get_logger(
    name: str = "genai_traces",
    level: int = logging.INFO,
    json_output: bool = False,
) -> StructuredLogger:
    """
    Get or create a logger.
    
    Args:
        name: Logger name
        level: Log level
        json_output: Whether to output JSON
        
    Returns:
        StructuredLogger instance
    """
    global _default_logger
    
    if _default_logger is None or _default_logger.name != name:
        _default_logger = StructuredLogger(
            name=name,
            level=level,
            json_output=json_output,
        )
    
    return _default_logger


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
):
    """
    Configure logging for GenAI-Traces.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR)
        json_output: Whether to output JSON
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    global _default_logger
    _default_logger = StructuredLogger(
        name="genai_traces",
        level=log_level,
        json_output=json_output,
    )
