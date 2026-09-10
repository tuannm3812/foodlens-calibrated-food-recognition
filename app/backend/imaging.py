"""Image decode and crop encoding for uploaded and detected regions.

Must not import inference.
"""

import base64
from io import BytesIO
from typing import Any

from .detector_policy import detector_region_role


def build_crop_data_url(crop: Any) -> str:
    """Encode a crop preview as a browser-ready JPEG data URL."""
    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def open_rgb_image(image_bytes: bytes) -> Any:
    """Open uploaded image bytes without requiring the classifier runtime."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Image decoding requires Pillow.") from exc

    return Image.open(BytesIO(image_bytes)).convert("RGB")


def build_full_image_region(image: Any) -> dict[str, Any]:
    """Build a fallback region when the detector produces no usable crops."""
    source_width, source_height = image.size
    return {
        "detection_index": 0,
        "detector_label": "whole_image",
        "proposal_role": detector_region_role("whole_image"),
        "detector_confidence": 1.0,
        "crop_area_ratio": 1.0,
        "x1": 0,
        "y1": 0,
        "x2": source_width,
        "y2": source_height,
        "source_width": source_width,
        "source_height": source_height,
    }
