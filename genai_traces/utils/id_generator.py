"""
ID generation utilities for trace and span IDs.
"""

import uuid
import time
import random


def generate_trace_id() -> str:
    """
    Generate a unique trace ID.
    
    Uses UUID4 for uniqueness, hex-encoded for compatibility with
    OpenTelemetry and other tracing systems.
    
    Returns:
        32-character hex string
    """
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """
    Generate a unique span ID.
    
    Uses a combination of timestamp and random bytes for uniqueness
    while being shorter than trace IDs.
    
    Returns:
        16-character hex string
    """
    # Use first 8 chars of UUID4 + timestamp-based suffix for uniqueness
    timestamp_part = hex(int(time.time() * 1000) % 0xFFFFFFFF)[2:].zfill(8)
    random_part = hex(random.getrandbits(32))[2:].zfill(8)
    return (timestamp_part + random_part)[:16]


def generate_short_id(length: int = 8) -> str:
    """
    Generate a short random ID.
    
    Args:
        length: Length of the ID (default 8)
        
    Returns:
        Random hex string of specified length
    """
    return uuid.uuid4().hex[:length]
