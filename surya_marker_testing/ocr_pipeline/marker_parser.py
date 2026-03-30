"""Marker wrapper — structured markdown + JSON output.

Marker is PDF-first.  For image inputs this module temporarily saves the image
as a single-page PDF, runs Marker, then cleans up.

IMPORTANT: Marker internally loads Surya models via create_model_dict().
To avoid loading those models twice, callers should pass in the already-loaded
model dict.  pipeline.py owns the single create_model_dict() call and passes
the same dict to both run_surya() and parse_with_marker().

Marker API note (marker-pdf >= 0.3)
------------------------------------
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

rendered = converter(path)         # path can be pdf or image (newer builds)
rendered.markdown                  # full markdown string
rendered.metadata                  # dict: pages, languages, toc, …
rendered.children                  # list of Block objects (sections, tables, …)
"""

from __future__ import annotations

import os
import re

from .image_utils import image_to_temp_pdf, is_pdf


# ---------------------------------------------------------------------------
# Markdown → structured sections
# ---------------------------------------------------------------------------

def _parse_markdown(markdown: str) -> list[dict]:
    """Split markdown into sections by heading and extract tables.

    Returns a list of block dicts:
        {
            "type":    "section" | "table" | "text",
            "heading": str | None,
            "content": str,           # raw markdown content
            "rows":    list[list[str]] | None   # only for type == "table"
        }
    """
    blocks: list[dict] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush():
        text = "\n".join(buffer).strip()
        if not text:
            return
        if _is_markdown_table(text):
            blocks.append({
                "type": "table",
                "heading": current_heading,
                "content": text,
                "rows": _parse_markdown_table(text),
            })
        else:
            blocks.append({
                "type": "section" if current_heading else "text",
                "heading": current_heading,
                "content": text,
                "rows": None,
            })
        buffer.clear()

    for line in markdown.splitlines():
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            flush()
            current_heading = heading_match.group(2).strip()
        else:
            buffer.append(line)

    flush()
    return blocks


def _is_markdown_table(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    return len(lines) >= 2 and "|" in lines[0] and re.match(r"[\s|:-]+", lines[1])


def _parse_markdown_table(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if re.match(r"^\s*\|?[\s:-]+\|", line):
            continue  # separator row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_with_marker(image_path: str, models: dict | None = None) -> dict:
    """Run Marker on an image (or PDF) and return structured output.

    Args:
        image_path: Path to a PNG, JPG, or PDF file.
        models: Pre-loaded model dict from marker's create_model_dict().
                If None, one is created (avoid in production — causes a
                second full model load if run_surya was already called).

    Returns:
        {
            "markdown": str,          # full Marker markdown output
            "sections": [
                {
                    "type":    "section" | "table" | "text",
                    "heading": str | None,
                    "content": str,
                    "rows":    list[list[str]] | None
                },
                ...
            ],
            "metadata": dict          # Marker metadata (pages, toc, etc.)
        }
    """
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser
    from marker.output import text_from_rendered

    if models is None:
        from marker.models import create_model_dict
        models = create_model_dict()

    tmp_pdf: str | None = None
    if not is_pdf(image_path):
        tmp_pdf = image_to_temp_pdf(image_path)
        source_path = tmp_pdf
    else:
        source_path = image_path

    try:
        config_parser = ConfigParser({"output_format": "markdown"})
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        rendered = converter(source_path)
        markdown, _, _ = text_from_rendered(rendered)
        metadata = rendered.metadata if hasattr(rendered, "metadata") else {}

    finally:
        if tmp_pdf and os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)

    sections = _parse_markdown(markdown)

    return {
        "markdown": markdown,
        "sections": sections,
        "metadata": metadata,
    }
