"""
Image metadata capture for multi-modal LLM inputs.

Privacy-first: only hashes and metadata are stored, never raw images.
"""

import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ImageMetadata:
    """Metadata for an image input."""
    
    content_hash: str
    size_bytes: int
    media_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
            "format": self.format,
        }


def capture_image_metadata(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
) -> ImageMetadata:
    """
    Extract privacy-safe metadata from image bytes.
    
    Args:
        image_bytes: Raw image bytes
        media_type: MIME type of the image
        
    Returns:
        ImageMetadata with hash and dimensions
    """
    content_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    
    width = height = None
    img_format = None
    
    try:
        from io import BytesIO
        from PIL import Image
        
        img = Image.open(BytesIO(image_bytes))
        width, height = img.size
        img_format = img.format
    except ImportError:
        pass
    except Exception:
        pass
    
    return ImageMetadata(
        content_hash=content_hash,
        size_bytes=len(image_bytes),
        media_type=media_type,
        width=width,
        height=height,
        format=img_format,
    )


def capture_image_url_metadata(
    image_url: str,
    fetch: bool = False,
) -> Dict[str, Any]:
    """
    Capture metadata for an image URL.
    
    Args:
        image_url: URL of the image
        fetch: Whether to fetch the image to get dimensions
        
    Returns:
        Dict with URL hash and optionally dimensions
    """
    url_hash = hashlib.sha256(image_url.encode()).hexdigest()[:16]
    
    metadata = {
        "url_hash": url_hash,
        "is_data_url": image_url.startswith("data:"),
    }
    
    if image_url.startswith("data:"):
        try:
            header, data = image_url.split(",", 1)
            media_type = header.split(";")[0].replace("data:", "")
            metadata["media_type"] = media_type
            
            import base64
            image_bytes = base64.b64decode(data)
            img_meta = capture_image_metadata(image_bytes, media_type)
            metadata.update(img_meta.to_dict())
        except Exception:
            pass
    
    elif fetch:
        try:
            import urllib.request
            with urllib.request.urlopen(image_url, timeout=5) as response:
                image_bytes = response.read()
                content_type = response.headers.get("Content-Type", "image/jpeg")
                img_meta = capture_image_metadata(image_bytes, content_type)
                metadata.update(img_meta.to_dict())
        except Exception:
            pass
    
    return metadata


def set_image_attributes(span: Any, images: list, prefix: str = "modal") -> None:
    """
    Set image-related attributes on a span.
    
    Args:
        span: The span to update
        images: List of image bytes or URLs
        prefix: Attribute prefix
    """
    span.set_attribute(f"{prefix}.input_type", "image")
    span.set_attribute(f"{prefix}.image_count", len(images))
    
    hashes = []
    total_size = 0
    
    for img in images:
        if isinstance(img, bytes):
            meta = capture_image_metadata(img)
            hashes.append(meta.content_hash)
            total_size += meta.size_bytes
        elif isinstance(img, str):
            meta = capture_image_url_metadata(img)
            hashes.append(meta.get("url_hash") or meta.get("hash", ""))
    
    if hashes:
        span.set_attribute(f"{prefix}.content_hashes", hashes)
    if total_size > 0:
        span.set_attribute(f"{prefix}.total_size_bytes", total_size)
