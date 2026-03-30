# OCRFULL

OCR benchmarking repo for comparing multiple document OCR pipelines on the same test set.

## Repo Layout

- `benchmark.py`: runs the benchmark across enabled pipelines and writes summary outputs
- `compare_outputs.py`: helper for side-by-side output inspection
- `paddleocr_testing/`: Paddle-based pipelines
  - `P1`: Paddle table extraction + Tesseract text OCR
  - `P3`: Paddle text OCR + Paddle table extraction
  - `P4`: `PP-StructureV3` structured document parsing
  - `P6`: `PaddleOCR-VL` document understanding
- `surya_marker_testing/`: `P2` Surya OCR + Marker parsing
- `test_images/`: sample PDFs and screenshots used for benchmarking

## Main Pipelines

- `P1`: balanced baseline for raw text + table extraction
- `P2`: richer structured extraction with Surya + Marker
- `P3`: Paddle-only OCR variant
- `P4`: Paddle `PP-StructureV3` structured parser
- `P6`: PaddleOCR-VL experimental document-understanding pipeline

## Main Entry Point

Run the benchmark from the repo root:

```bash
python benchmark.py
```

Useful variants:

```bash
python benchmark.py --run-p3
python benchmark.py --run-p4
python benchmark.py --run-p6
python benchmark.py --images test_images/f1040example_filled_fake.pdf
```
