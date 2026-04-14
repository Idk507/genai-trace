"""
Multi-modal trace support for GenAI-Traces.

Captures metadata for image and audio inputs without storing raw content.
"""

from .image_tracer import capture_image_metadata, ImageMetadata
from .audio_tracer import capture_audio_metadata, AudioMetadata
from .content_hash import hash_content, ContentHash

__all__ = [
    "capture_image_metadata",
    "ImageMetadata",
    "capture_audio_metadata",
    "AudioMetadata",
    "hash_content",
    "ContentHash",
]
