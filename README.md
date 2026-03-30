# OCRFULL

Two OCR pipelines are benchmarked side by side:

- `paddleocr_testing`: PaddleOCR table extraction + Tesseract text OCR
- `surya_marker_testing`: Surya OCR + Marker structured parsing

Additional experimental Paddle-only variants are also available:

- `P3`: Paddle text OCR + Paddle table extraction
- `P4`: full `PP-StructureV3` structured parsing
- `P6`: `PaddleOCR-VL` document understanding

The project includes macOS virtual environments locally, but those must not be reused on a Linux host. `setup_remote.sh` rebuilds the environments on the target machine.

## Copy To Remote

From your local machine:

```bash
cd /Users/jaymaheshwari/Projects/AppDevBAHWork/OCRFULL
rsync -av --exclude-from=.rsync-exclude ./ your_user@your-remote-host:~/OCRFULL/
```

`.rsync-exclude` skips the local macOS venvs, cached outputs, and Python bytecode so the remote only receives the project files it actually needs.

## Remote Setup

After SSHing into your Linux host:

```bash
cd ~/OCRFULL
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_INDEX=2 bash setup_remote.sh
```

Notes:

- Replace `2` with the GPU index you actually want to use.
- If Python 3.11 is already installed somewhere unusual, pass it explicitly:

```bash
PYTHON_BIN=/path/to/python3.11 GPU_INDEX=2 bash setup_remote.sh
```

- If system packages are already installed, you can skip the `apt-get` step:

```bash
SKIP_APT=1 GPU_INDEX=2 bash setup_remote.sh
```

The setup script:

- installs Linux system dependencies
- creates fresh Linux virtual environments for both pipelines
- installs PaddlePaddle 3.3.x / PaddleOCR 3.4.0 / PaddleX 3.4.2 for Pipeline 1
- installs GPU builds when CUDA is available, otherwise CPU builds
- stores model caches under `~/data` by default so large downloads persist

## Run The Benchmark

```bash
cd ~/OCRFULL
python3.11 benchmark.py
```

If `setup_remote.sh` had to create or discover Python 3.11 in a nonstandard location, use the exact interpreter path it prints at the end of setup instead of `python3.11`.

Useful variants:

```bash
python3.11 benchmark.py --skip-p1
python3.11 benchmark.py --skip-p2
python3.11 benchmark.py --run-p3
python3.11 benchmark.py --run-p4
python3.11 benchmark.py --run-p6
python3.11 benchmark.py --images test_images/f1040example_filled_fake.pdf
```

If the target machine has a broken cuDNN setup but CUDA itself still works, you can disable cuDNN just for the Surya/Marker run:

```bash
OCRFULL_DISABLE_CUDNN=1 CUDA_VISIBLE_DEVICES=2 python3.11 benchmark.py --skip-p1 --images test_images/f1040example_filled_fake.pdf
```

If you previously built Pipeline 1 with older Paddle 2.6 wheels, rebuild that environment before attempting GPU-vs-GPU comparisons:

```bash
rm -rf paddleocr_testing/paddle-env
GPU_INDEX=0 bash setup_remote.sh
OCRFULL_P1_DEVICE=gpu:0 OCRFULL_DISABLE_CUDNN=1 CUDA_VISIBLE_DEVICES=0 python3.11 benchmark.py --images test_images/f1040example_filled_fake.pdf
```

Outputs:

- `benchmark_results.csv`
- `benchmark_summary.txt`
- per-pipeline JSON outputs under each pipeline's `output/bench/`

## Important Remote Detail

The Surya pipeline reads PDFs through `pypdfium2`, so that dependency is required even if Marker is already present in another environment. The remote setup now installs or checks that explicitly.
