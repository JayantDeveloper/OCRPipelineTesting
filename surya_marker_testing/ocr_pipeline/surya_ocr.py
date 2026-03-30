"""Surya OCR wrapper — text recognition + layout detection.

Uses the predictor-class API introduced in surya-ocr >= 0.7 (the same API
that marker-pdf >= 0.3 uses internally).  Models are passed in from the
caller so they are never loaded more than once per process.

If called without a pre-loaded model dict it falls back to creating one via
marker's create_model_dict(), which is the single source of truth for model
instantiation in this pipeline.

Surya predictor output reference
---------------------------------
DetectionPredictor([img])
  -> list[TextDetectionResult]
       .bboxes: list[PolygonBox]
         .bbox:    [x1, y1, x2, y2]
         .polygon: [[x,y], ...]

RecognitionPredictor([img], [langs], det_results)
  -> list[OCRResult]
       .text_lines: list[TextLine]
         .text:       str
         .confidence: float
         .bbox:       [x1, y1, x2, y2]
         .polygon:    [[x,y], ...]

LayoutPredictor([img])
  -> list[LayoutResult]
       .bboxes: list[LayoutBox]
         .label:    str  ("Text", "Title", "Table", "Figure", …)
         .bbox:     [x1, y1, x2, y2]
         .position: int  (reading order)
"""

from __future__ import annotations

from PIL import Image
from surya.common.surya.schema import TaskNames

from .image_utils import is_pdf, load_pdf_images


def _get_model(models: dict, *names: str):
    for name in names:
        if name in models:
            return models[name]
    available = ", ".join(sorted(models))
    raise KeyError(f"None of {names!r} found in models dict. Available keys: {available}")


def _rounded_bbox(obj) -> list[int]:
    bbox = getattr(obj, "bbox", None)
    if bbox is None and hasattr(obj, "polygon"):
        xs = [pt[0] for pt in obj.polygon]
        ys = [pt[1] for pt in obj.polygon]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    if bbox is None:
        return [0, 0, 0, 0]
    return [round(v) for v in bbox]


def run_surya(
    image_path: str,
    models: dict | None = None,
    langs: list[str] | None = None,
) -> dict:
    """Run Surya OCR + layout analysis using shared predictor instances.

    Args:
        image_path: Path to the image file.
        models: Pre-loaded model dict from marker's create_model_dict().
                If None, one is created (avoid in production — pass it in).
        langs: Language codes (default: ["en"]).

    Returns:
        {
            "text_lines": [
                {
                    "text":       str,
                    "confidence": float,
                    "bbox":       [x1, y1, x2, y2],
                    "polygon":    [[x, y], ...]
                },
                ...
            ],
            "layout_blocks": [
                {
                    "label":    str,
                    "bbox":     [x1, y1, x2, y2],
                    "position": int
                },
                ...
            ]
        }
    """
    if langs is None:
        langs = ["en"]

    if models is None:
        from marker.models import create_model_dict
        models = create_model_dict()

    if is_pdf(image_path):
        images = load_pdf_images(image_path, scale=2.0)
    else:
        with Image.open(image_path) as image:
            images = [image.convert("RGB")]

    # --- Text detection + recognition ---
    det_predictor = _get_model(models, "detection", "detection_model")
    rec_predictor = _get_model(models, "recognition", "recognition_model")
    layout_predictor = _get_model(models, "layout", "layout_model")

    text_lines = []
    layout_blocks = []

    for image in images:
        rec_results = rec_predictor(
            [image],
            task_names=[TaskNames.ocr_with_boxes],
            det_predictor=det_predictor,
        )
        page = rec_results[0]

        text_lines += [
            {
                "text": line.text,
                "confidence": round(float(line.confidence), 4),
                "bbox": _rounded_bbox(line),
                "polygon": [[round(pt[0]), round(pt[1])] for pt in line.polygon],
            }
            for line in page.text_lines
        ]

        layout_results = layout_predictor([image])
        layout_page = layout_results[0]

        layout_blocks += sorted(
            [
                {
                    "label": box.label,
                    "bbox": _rounded_bbox(box),
                    "position": box.position,
                }
                for box in layout_page.bboxes
            ],
            key=lambda b: b["position"],
        )

    return {
        "text_lines": text_lines,
        "layout_blocks": layout_blocks,
    }
