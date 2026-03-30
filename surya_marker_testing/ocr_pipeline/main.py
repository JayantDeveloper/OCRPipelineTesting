"""CLI entry point.

Usage:
    python -m ocr_pipeline.main path/to/image.png
    python -m ocr_pipeline.main path/to/image.png --output result.json
    python -m ocr_pipeline.main path/to/doc.pdf --output result.json --langs en fr
"""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import process_image


def main():
    parser = argparse.ArgumentParser(
        description="Surya OCR + Marker structured document pipeline"
    )
    parser.add_argument("image", help="Path to the input image or PDF")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Write JSON result to this file (default: print to stdout)",
    )
    parser.add_argument(
        "--langs", "-l",
        nargs="+",
        default=["en"],
        metavar="LANG",
        help="Language codes for Surya OCR (default: en)",
    )
    args = parser.parse_args()

    print(f"[pipeline] Processing: {args.image}", file=sys.stderr)
    result = process_image(args.image, langs=args.langs)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"[pipeline] Saved → {args.output}", file=sys.stderr)
        _print_summary(result)
    else:
        print(output_json)


def _print_summary(result: dict):
    surya = result.get("surya", {})
    marker = result.get("marker", {})
    text_lines = surya.get("text_lines", [])
    layout_blocks = surya.get("layout_blocks", [])
    sections = marker.get("sections", [])
    tables = [s for s in sections if s.get("type") == "table"]

    print(f"  Surya text lines:    {len(text_lines)}")
    print(f"  Surya layout blocks: {len(layout_blocks)}")
    print(f"  Marker sections:     {len(sections)}")
    print(f"  Marker tables:       {len(tables)}")
    print(f"  Marker markdown len: {len(marker.get('markdown', ''))} chars")


if __name__ == "__main__":
    main()
