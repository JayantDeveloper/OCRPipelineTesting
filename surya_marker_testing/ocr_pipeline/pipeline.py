"""Main pipeline — orchestrates Surya OCR and Marker parsing.

Model loading strategy
-----------------------
Marker internally uses Surya for all OCR/layout work.  Both tools share the
same underlying models.  To prevent loading those models twice, this module
calls create_model_dict() exactly once and passes the resulting dict to both
run_surya() and parse_with_marker().

  image
    └─► create_model_dict()   ← single load
          ├─► run_surya()      raw text lines + layout blocks
          └─► parse_with_marker() structured markdown + sections
"""

from __future__ import annotations

import os

from .surya_ocr import run_surya
from .marker_parser import parse_with_marker


def _configure_torch_runtime() -> None:
    """Apply runtime workarounds before model creation.

    Some remote CUDA setups on shared servers fail during the first cuDNN-backed
    convolution even though plain CUDA is otherwise usable. Setting
    `OCRFULL_DISABLE_CUDNN=1` forces PyTorch to use non-cuDNN kernels for the
    Surya/Marker process.
    """
    if os.getenv("OCRFULL_DISABLE_CUDNN") != "1":
        return

    import torch

    torch.backends.cudnn.enabled = False


def process_image(image_path: str, langs: list[str] | None = None) -> dict:
    """Run the full Surya + Marker OCR pipeline on a single image or PDF.

    Args:
        image_path: Path to a PNG, JPG, or PDF file.
        langs: Language codes for Surya OCR (default: ["en"]).

    Returns:
        {
            "source": str,
            "surya": {
                "text_lines": [...],
                "layout_blocks": [...]
            },
            "marker": {
                "markdown": str,
                "sections": [...],
                "metadata": {...}
            }
        }
    """
    _configure_torch_runtime()

    # Load all Surya/Marker models once.
    from marker.models import create_model_dict
    models = create_model_dict()

    surya_output = run_surya(image_path, models=models, langs=langs)
    marker_output = parse_with_marker(image_path, models=models)

    return {
        "source": image_path,
        "surya": surya_output,
        "marker": marker_output,
    }
