"""Create side-by-side comparable text outputs for P1 and P2 benchmark JSONs.

Usage:
    python compare_outputs.py f1040example_filled_fake
    python compare_outputs.py --p1-json path/to/p1.json --p2-json path/to/p2.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path


ROOT = Path(__file__).parent
P1_BENCH_DIR = ROOT / "paddleocr_testing" / "output" / "bench"
P2_BENCH_DIR = ROOT / "surya_marker_testing" / "output" / "bench"
COMPARE_DIR = ROOT / "comparison_output"


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip() + "\n"


def _extract_p1_text(data: dict) -> str:
    parts = ["# Raw Text", data.get("raw_text", "").strip()]
    fields = data.get("structured_fields", [])
    if fields:
        parts.append("\n# Structured Fields")
        for item in fields:
            page = item.get("page_index")
            prefix = f"[page {page}] " if page is not None else ""
            parts.append(f"{prefix}{item.get('field', '')}: {item.get('value', '')}".strip(": "))
    tables = data.get("tables", [])
    if tables:
        parts.append("\n# Tables")
        for idx, table in enumerate(tables, start=1):
            parts.append(f"\n## Table {idx}")
            for cell in table.get("cells", []):
                field = cell.get("field", "").strip()
                value = cell.get("value", "").strip()
                parts.append(f"{field}: {value}".strip(": "))
    return _normalize_text("\n".join(part for part in parts if part))


def _extract_p2_text(data: dict) -> str:
    marker = data.get("marker", {})
    markdown = marker.get("markdown", "").strip()
    sections = marker.get("sections", [])
    parts = ["# Marker Markdown", markdown]
    table_sections = [s for s in sections if s.get("type") == "table"]
    if table_sections:
        parts.append("\n# Parsed Tables")
        for idx, section in enumerate(table_sections, start=1):
            parts.append(f"\n## Table {idx}")
            for row in section.get("rows", []):
                parts.append(" | ".join(str(cell).strip() for cell in row))
    return _normalize_text("\n".join(part for part in parts if part))


def _print_side_by_side(left: str, right: str, preview_lines: int) -> None:
    width = shutil.get_terminal_size((160, 40)).columns
    col_width = max(30, (width - 3) // 2)
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    total = max(len(left_lines), len(right_lines), preview_lines)

    print("\nP1".ljust(col_width) + " | " + "P2".ljust(col_width))
    print("-" * col_width + "-+-" + "-" * col_width)
    for idx in range(min(total, preview_lines)):
        l = left_lines[idx] if idx < len(left_lines) else ""
        r = right_lines[idx] if idx < len(right_lines) else ""
        l_wrapped = textwrap.wrap(l, width=col_width) or [""]
        r_wrapped = textwrap.wrap(r, width=col_width) or [""]
        rows = max(len(l_wrapped), len(r_wrapped))
        for sub in range(rows):
            l_part = l_wrapped[sub] if sub < len(l_wrapped) else ""
            r_part = r_wrapped[sub] if sub < len(r_wrapped) else ""
            print(l_part.ljust(col_width) + " | " + r_part.ljust(col_width))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark outputs side by side")
    parser.add_argument("image_stem", nargs="?", help="Image stem used by benchmark outputs")
    parser.add_argument("--p1-json", type=Path, help="Path to Pipeline 1 JSON")
    parser.add_argument("--p2-json", type=Path, help="Path to Pipeline 2 JSON")
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=80,
        help="Number of side-by-side terminal lines to preview (default: 80)",
    )
    args = parser.parse_args()

    if args.p1_json and args.p2_json:
        p1_json = args.p1_json
        p2_json = args.p2_json
        stem = p1_json.stem.replace("_p1", "")
    elif args.image_stem:
        stem = args.image_stem
        p1_json = P1_BENCH_DIR / f"{stem}_p1.json"
        p2_json = P2_BENCH_DIR / f"{stem}_p2.json"
    else:
        parser.error("Provide either image_stem or both --p1-json and --p2-json.")

    if not p1_json.exists():
        raise SystemExit(f"P1 JSON not found: {p1_json}")
    if not p2_json.exists():
        raise SystemExit(f"P2 JSON not found: {p2_json}")

    p1_text = _extract_p1_text(_read_json(p1_json))
    p2_text = _extract_p2_text(_read_json(p2_json))

    COMPARE_DIR.mkdir(exist_ok=True)
    p1_out = COMPARE_DIR / f"{stem}_p1.txt"
    p2_out = COMPARE_DIR / f"{stem}_p2.txt"
    p1_out.write_text(p1_text, encoding="utf-8")
    p2_out.write_text(p2_text, encoding="utf-8")

    print(f"Saved P1 text: {p1_out}")
    print(f"Saved P2 text: {p2_out}")
    _print_side_by_side(p1_text, p2_text, args.preview_lines)


if __name__ == "__main__":
    main()
