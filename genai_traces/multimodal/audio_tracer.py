"""
Audio metadata capture for multi-modal LLM inputs.

Privacy-first: only hashes and metadata are stored, never raw audio.
"""

import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AudioMetadata:
    """Metadata for an audio input."""
    
    content_hash: str
    size_bytes: int
    media_type: str
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format,
        }


def capture_audio_metadata(
    audio_bytes: bytes,
    media_type: str = "audio/wav",
    duration_seconds: Optional[float] = None,
) -> AudioMetadata:
    """
    Extract privacy-safe metadata from audio bytes.
    
    Args:
        audio_bytes: Raw audio bytes
        media_type: MIME type of the audio
        duration_seconds: Known duration if available
        
    Returns:
        AudioMetadata with hash and audio properties
    """
    content_hash = hashlib.sha256(audio_bytes).hexdigest()[:16]
    
    sample_rate = None
    channels = None
    audio_format = None
    detected_duration = duration_seconds
    
    if media_type in ("audio/wav", "audio/x-wav") or audio_bytes[:4] == b"RIFF":
        try:
            import wave
            from io import BytesIO
            
            with wave.open(BytesIO(audio_bytes), 'rb') as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                frames = wav.getnframes()
                detected_duration = frames / sample_rate if sample_rate else None
                audio_format = "wav"
        except Exception:
            pass
    
    if detected_duration is None:
        try:
            import soundfile as sf
            from io import BytesIO
            
            data, sr = sf.read(BytesIO(audio_bytes))
            sample_rate = sr
            channels = data.shape[1] if len(data.shape) > 1 else 1
            detected_duration = len(data) / sr
        except ImportError:
            pass
        except Exception:
            pass
    
    return AudioMetadata(
        content_hash=content_hash,
        size_bytes=len(audio_bytes),
        media_type=media_type,
        duration_seconds=detected_duration,
        sample_rate=sample_rate,
        channels=channels,
        format=audio_format,
    )


def estimate_audio_tokens(
    duration_seconds: float,
    model: str = "whisper-1",
) -> int:
    """
    Estimate token count for audio transcription.
    
    Args:
        duration_seconds: Audio duration
        model: Model being used
        
    Returns:
        Estimated token count
    """
    words_per_second = 2.5
    tokens_per_word = 1.3
    
    estimated_words = duration_seconds * words_per_second
    estimated_tokens = int(estimated_words * tokens_per_word)
    
    return max(1, estimated_tokens)


def set_audio_attributes(span: Any, audio_data: Any, prefix: str = "modal") -> None:
    """
    Set audio-related attributes on a span.
    
    Args:
        span: The span to update
        audio_data: Audio bytes or metadata dict
        prefix: Attribute prefix
    """
    span.set_attribute(f"{prefix}.input_type", "audio")
    
    if isinstance(audio_data, bytes):
        meta = capture_audio_metadata(audio_data)
        span.set_attribute(f"{prefix}.content_hash", meta.content_hash)
        span.set_attribute(f"{prefix}.size_bytes", meta.size_bytes)
        if meta.duration_seconds:
            span.set_attribute(f"{prefix}.audio_seconds", meta.duration_seconds)
        if meta.sample_rate:
            span.set_attribute(f"{prefix}.sample_rate", meta.sample_rate)
    elif isinstance(audio_data, dict):
        if "hash" in audio_data:
            span.set_attribute(f"{prefix}.content_hash", audio_data["hash"])
        if "duration_seconds" in audio_data:
            span.set_attribute(f"{prefix}.audio_seconds", audio_data["duration_seconds"])
