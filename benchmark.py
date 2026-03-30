"""Benchmark runner — compares multiple OCR pipelines across test images.

Usage:
    python3.11 benchmark.py
    python3.11 benchmark.py --images test_images/f1040example_filled_fake.pdf
    python3.11 benchmark.py --skip-p1   # skip PaddleOCR + Tesseract
    python3.11 benchmark.py --skip-p2   # skip Surya/Marker
    python3.11 benchmark.py --run-p3    # include Paddle-only P3
    python3.11 benchmark.py --run-p4    # include PP-StructureV3 P4

Outputs:
    benchmark_results.csv   — full per-file metrics
    benchmark_summary.txt   — printed summary table (also saved)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from statistics import mean

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
TEST_IMAGES_DIR = ROOT / "test_images"

P1_PYTHON = ROOT / "paddleocr_testing" / "paddle-env" / "bin" / "python"
P1_MODULE = "ocr_pipeline.main"
P1_CWD = ROOT / "paddleocr_testing"
P1_OUTPUT_DIR = P1_CWD / "output" / "bench"

P3_PYTHON = P1_PYTHON
P3_MODULE = "ocr_pipeline.main_paddle_only"
P3_CWD = P1_CWD
P3_OUTPUT_DIR = P1_OUTPUT_DIR

P4_PYTHON = P1_PYTHON
P4_MODULE = "ocr_pipeline.main_p4"
P4_CWD = P1_CWD
P4_OUTPUT_DIR = P1_OUTPUT_DIR

P6_PYTHON = P1_PYTHON
P6_MODULE = "ocr_pipeline.main_p6"
P6_CWD = P1_CWD
P6_OUTPUT_DIR = P1_OUTPUT_DIR

P2_PYTHON = ROOT / "surya_marker_testing" / "surya-env" / "bin" / "python"
P2_MODULE = "ocr_pipeline.main"
P2_CWD = ROOT / "surya_marker_testing"
P2_OUTPUT_DIR = P2_CWD / "output" / "bench"

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".pdf"}
CSV_FIELDNAMES = [
    "image",
    "pipeline",
    "elapsed",
    "error",
    "text_chars",
    "text_words",
    "tables_found",
    "table_cells",
    "text_lines",
    "avg_confidence",
    "layout_blocks",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_test_images(paths: list[str] | None = None) -> list[Path]:
    if paths:
        return [Path(p).resolve() for p in paths]
    return sorted(
        p for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )


def run_pipeline(
    python: Path,
    module: str,
    cwd: Path,
    image: Path,
    output_file: Path,
    timeout_seconds: int = 300,
) -> tuple[float, str | None]:
    """Run a pipeline in its own venv, return (elapsed_seconds, error_msg)."""
    if not python.exists():
        return 0.0, f"venv not found: {python}"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(python), "-m", module, str(image), "--output", str(output_file)]

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return elapsed, summarize_process_output(result.stdout, result.stderr)
        return elapsed, None
    except subprocess.TimeoutExpired:
        return float(timeout_seconds), "timeout"
    except Exception as e:
        return 0.0, str(e)


def summarize_process_output(stdout: str, stderr: str) -> str:
    """Return the most useful error line from a subprocess traceback/log stream.

    Some libraries write non-error status lines to stderr, and some tracebacks
    spill across stdout/stderr. We scan both streams and filter known noise.

    pypdfium2 sometimes emits a shutdown warning as the last stderr line:
    "-> Cannot close object, library is destroyed..."
    That line is usually secondary noise, not the real failure.
    """
    combined = "\n".join(part for part in (stderr, stdout) if part)
    if not combined:
        return "non-zero exit"

    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    if not lines:
        return "non-zero exit"

    noise_prefixes = (
        "-> Cannot close object, library is destroyed.",
        "Model files already exist. Using cached files.",
        "Processing:",
    )
    informative = [line for line in lines if not line.startswith(noise_prefixes)]
    if not informative:
        informative = lines

    for line in reversed(informative):
        if ":" in line:
            return line

    return informative[-1]


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def metrics_p1(json_path: Path) -> dict:
    """Extract metrics from Paddle-based output JSON."""
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return {}

    raw_text = data.get("raw_text", "")
    tables = data.get("tables", [])
    total_cells = sum(len(t.get("cells", [])) for t in tables)

    return {
        "text_chars": len(raw_text),
        "text_words": len(raw_text.split()),
        "tables_found": len(tables),
        "table_cells": total_cells,
        "text_lines": None,
        "avg_confidence": None,
        "layout_blocks": None,
    }


def metrics_p2(json_path: Path) -> dict:
    """Extract metrics from Pipeline 2 output JSON."""
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return {}

    surya = data.get("surya", {})
    marker = data.get("marker", {})

    markdown = marker.get("markdown", "")
    sections = marker.get("sections", [])
    table_sections = [s for s in sections if s.get("type") == "table"]
    total_rows = sum(len(s.get("rows") or []) for s in table_sections)

    text_lines = surya.get("text_lines", [])
    confidences = [l["confidence"] for l in text_lines if "confidence" in l]
    layout_blocks = surya.get("layout_blocks", [])

    return {
        "text_chars": len(markdown),
        "text_words": len(markdown.split()),
        "tables_found": len(table_sections),
        "table_cells": total_rows,
        "text_lines": len(text_lines),
        "avg_confidence": round(mean(confidences), 4) if confidences else None,
        "layout_blocks": len(layout_blocks),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(val, unit="") -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.2f}{unit}"
    return f"{val}{unit}"


def print_table(rows: list[dict], headers: list[str]):
    col_widths = [max(len(h), max((len(str(r.get(h, "—"))) for r in rows), default=0))
                  for h in headers]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(row.get(h, "—")).ljust(w)
                                  for h, w in zip(headers, col_widths)) + " |"
        print(line)
    print(sep)


def build_report_rows(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        name = Path(r["image"]).name[:40]
        rows.append({
            "Image": name,
            "Pipeline": r["pipeline"],
            "Time (s)": _fmt(r.get("elapsed"), "s"),
            "Words": _fmt(r.get("text_words")),
            "Tables": _fmt(r.get("tables_found")),
            "Cells/Rows": _fmt(r.get("table_cells")),
            "Text Lines": _fmt(r.get("text_lines")),
            "Avg Conf": _fmt(r.get("avg_confidence")),
            "Error": r.get("error") or "",
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR pipelines")
    parser.add_argument("--images", nargs="+", help="Specific image paths to test")
    parser.add_argument("--skip-p1", action="store_true", help="Skip Pipeline 1 (PaddleOCR)")
    parser.add_argument("--skip-p2", action="store_true", help="Skip Pipeline 2 (Surya+Marker)")
    parser.add_argument("--run-p3", action="store_true", help="Include Pipeline 3 (Paddle-only)")
    parser.add_argument("--run-p4", action="store_true", help="Include Pipeline 4 (PP-StructureV3)")
    parser.add_argument("--run-p6", action="store_true", help="Include Pipeline 6 (PaddleOCR-VL)")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Per-pipeline timeout in seconds (default: 300).",
    )
    args = parser.parse_args()

    images = find_test_images(args.images)
    if not images:
        print("No test images found.", file=sys.stderr)
        sys.exit(1)

    pipeline_names = []
    if not args.skip_p1:
        pipeline_names.append("P1")
    if not args.skip_p2:
        pipeline_names.append("P2")
    if args.run_p3:
        pipeline_names.append("P3")
    if args.run_p4:
        pipeline_names.append("P4")
    if args.run_p6:
        pipeline_names.append("P6")

    print(f"\nFound {len(images)} test image(s).")
    print(f"Running {' + '.join(pipeline_names)}...\n")

    all_results = []

    for img in images:
        print(f"▶  {img.name}")

        if not args.skip_p1:
            out = P1_OUTPUT_DIR / f"{img.stem}_p1.json"
            elapsed, err = run_pipeline(
                P1_PYTHON,
                P1_MODULE,
                P1_CWD,
                img,
                out,
                timeout_seconds=args.timeout_seconds,
            )
            m = metrics_p1(out) if not err else {}
            status = f"✓ {elapsed:.1f}s" if not err else f"✗ {err}"
            print(f"   P1 (PaddleOCR): {status}")
            all_results.append({"image": str(img), "pipeline": "P1-PaddleOCR",
                                 "elapsed": elapsed, "error": err, **m})

        if not args.skip_p2:
            out = P2_OUTPUT_DIR / f"{img.stem}_p2.json"
            elapsed, err = run_pipeline(
                P2_PYTHON,
                P2_MODULE,
                P2_CWD,
                img,
                out,
                timeout_seconds=args.timeout_seconds,
            )
            m = metrics_p2(out) if not err else {}
            status = f"✓ {elapsed:.1f}s" if not err else f"✗ {err}"
            print(f"   P2 (Surya+Marker): {status}")
            all_results.append({"image": str(img), "pipeline": "P2-Surya+Marker",
                                 "elapsed": elapsed, "error": err, **m})

        if args.run_p3:
            out = P3_OUTPUT_DIR / f"{img.stem}_p3.json"
            elapsed, err = run_pipeline(
                P3_PYTHON,
                P3_MODULE,
                P3_CWD,
                img,
                out,
                timeout_seconds=args.timeout_seconds,
            )
            m = metrics_p1(out) if not err else {}
            status = f"✓ {elapsed:.1f}s" if not err else f"✗ {err}"
            print(f"   P3 (Paddle-only): {status}")
            all_results.append({"image": str(img), "pipeline": "P3-PaddleOnly",
                                 "elapsed": elapsed, "error": err, **m})

        if args.run_p4:
            out = P4_OUTPUT_DIR / f"{img.stem}_p4.json"
            elapsed, err = run_pipeline(
                P4_PYTHON,
                P4_MODULE,
                P4_CWD,
                img,
                out,
                timeout_seconds=args.timeout_seconds,
            )
            m = metrics_p1(out) if not err else {}
            status = f"✓ {elapsed:.1f}s" if not err else f"✗ {err}"
            print(f"   P4 (PP-StructureV3): {status}")
            all_results.append({"image": str(img), "pipeline": "P4-PPStructureV3",
                                 "elapsed": elapsed, "error": err, **m})

        if args.run_p6:
            out = P6_OUTPUT_DIR / f"{img.stem}_p6.json"
            elapsed, err = run_pipeline(
                P6_PYTHON,
                P6_MODULE,
                P6_CWD,
                img,
                out,
                timeout_seconds=args.timeout_seconds,
            )
            m = metrics_p1(out) if not err else {}
            status = f"✓ {elapsed:.1f}s" if not err else f"✗ {err}"
            print(f"   P6 (PaddleOCR-VL): {status}")
            all_results.append({"image": str(img), "pipeline": "P6-PaddleOCRVL",
                                 "elapsed": elapsed, "error": err, **m})

    # --- Summary table ---
    report_rows = build_report_rows(all_results)
    headers = ["Image", "Pipeline", "Time (s)", "Words", "Tables",
               "Cells/Rows", "Text Lines", "Avg Conf", "Error"]

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print_table(report_rows, headers)

    # --- Save CSV ---
    csv_path = ROOT / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nFull results saved to: {csv_path}")

    # --- Save summary text ---
    summary_path = ROOT / "benchmark_summary.txt"
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_table(report_rows, headers)
    sys.stdout = old_stdout
    summary_path.write_text(buf.getvalue())
    print(f"Summary table saved to: {summary_path}\n")


if __name__ == "__main__":
    main()
