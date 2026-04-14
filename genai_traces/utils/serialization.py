"""
Serialization utilities for spans and other objects.
"""

import json
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict
from enum import Enum


def json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for objects not serializable by default json code.
    
    Handles:
    - datetime objects
    - date objects
    - Decimal objects
    - Enum values
    - Objects with to_dict() method
    - Objects with __dict__ attribute
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


def span_to_jsonable(span: Any) -> Dict[str, Any]:
    """
    Convert a Span object to a JSON-serializable dictionary.
    
    Args:
        span: Span object to convert
        
    Returns:
        Dictionary that can be serialized to JSON
    """
    if hasattr(span, "to_dict"):
        data = span.to_dict()
    else:
        data = dict(span) if hasattr(span, "__iter__") else {"data": str(span)}
    
    # Ensure all values are JSON-serializable
    return _make_jsonable(data)


def _make_jsonable(obj: Any) -> Any:
    """Recursively make an object JSON-serializable."""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {str(k): _make_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_jsonable(item) for item in obj]
    elif hasattr(obj, "to_dict"):
        return _make_jsonable(obj.to_dict())
    elif hasattr(obj, "__dict__"):
        return _make_jsonable(obj.__dict__)
    else:
        return str(obj)


def dumps(obj: Any, **kwargs) -> str:
    """
    Serialize object to JSON string with custom serializer.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments passed to json.dumps
        
    Returns:
        JSON string
    """
    return json.dumps(obj, default=json_serializer, **kwargs)


def loads(s: str, **kwargs) -> Any:
    """
    Deserialize JSON string to object.
    
    Args:
        s: JSON string to deserialize
        **kwargs: Additional arguments passed to json.loads
        
    Returns:
        Deserialized object
    """
    return json.loads(s, **kwargs)


def serialize_span(span: Any) -> str:
    """
    Serialize a span to JSON string.
    
    Args:
        span: Span object to serialize
        
    Returns:
        JSON string representation
    """
    return dumps(span_to_jsonable(span))


def deserialize_span(json_str: str) -> Dict[str, Any]:
    """
    Deserialize a span from JSON string.
    
    Args:
        json_str: JSON string to deserialize
        
    Returns:
        Dictionary representation of span
    """
    return loads(json_str)
