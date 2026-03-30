"""Image loading, normalization, and format-conversion helpers."""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
from PIL import Image


def load_pil(path: str) -> Image.Image:
    """Load any image (PNG, JPG, …) as an RGB PIL Image."""
    with Image.open(path) as img:
        return img.convert("RGB")


def load_np(path: str) -> np.ndarray:
    """Load image as a BGR numpy array (OpenCV format)."""
    arr = cv2.imread(path)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return arr


def pil_to_np(img: Image.Image) -> np.ndarray:
    """Convert RGB PIL Image to BGR numpy array."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to RGB PIL Image."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def image_to_temp_pdf(image_path: str) -> str:
    """Save an image as a temporary single-page PDF and return the file path.

    The caller is responsible for deleting the file when done.
    Marker is PDF-first; this bridge lets it handle plain images.
    """
    img = load_pil(image_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    img.save(tmp.name, format="PDF", resolution=150)
    return tmp.name


def load_pdf_images(pdf_path: str, scale: float = 2.0) -> list[Image.Image]:
    """Render a PDF into detached RGB PIL images.

    The explicit copies are important. `pypdfium2` PIL images may otherwise
    retain buffers tied to PDFium objects that are finalized during interpreter
    shutdown, which can surface as noisy "library is destroyed" stderr lines.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        renderer = pdf.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=list(range(len(pdf))),
            scale=scale,
            draw_annots=False,
        )
        return [page_image.convert("RGB").copy() for page_image in renderer]
    finally:
        pdf.close()


def is_pdf(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".pdf"
