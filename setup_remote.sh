#!/usr/bin/env bash
# setup_remote.sh - Run this on the remote Linux machine after copying OCRFULL.
#
# Examples:
#   bash setup_remote.sh
#   GPU_INDEX=2 bash setup_remote.sh
#   PYTHON_BIN=/usr/bin/python3.11 SKIP_APT=1 bash setup_remote.sh
#
# Optional env vars:
#   GPU_INDEX       Value to persist as CUDA_VISIBLE_DEVICES in ~/.bashrc.
#   PYTHON_BIN      Explicit Python 3.11 interpreter to use for both venvs.
#   MODEL_CACHE_DIR Root directory for Paddle/HuggingFace caches. Default: ~/data
#   SKIP_APT        Set to 1 to skip apt-get installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_INDEX="${GPU_INDEX:-${CUDA_VISIBLE_DEVICES:-}}"
PYTHON_BIN="${PYTHON_BIN:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/data}"
SKIP_APT="${SKIP_APT:-0}"
BASHRC="$HOME/.bashrc"

echo "============================================================"
echo "  OCRFULL Remote Setup"
echo "  Working directory: $SCRIPT_DIR"
echo "  GPU_INDEX: ${GPU_INDEX:-<unset>}"
echo "  MODEL_CACHE_DIR: $MODEL_CACHE_DIR"
echo "============================================================"

rewrite_managed_block() {
    local file="$1"
    local start_marker="$2"
    local end_marker="$3"
    local block_content="$4"

    touch "$file"
    awk -v start="$start_marker" -v end="$end_marker" '
        $0 == start { skip = 1; next }
        $0 == end { skip = 0; next }
        !skip { print }
    ' "$file" > "${file}.tmp"
    mv "${file}.tmp" "$file"

    {
        printf "\n%s\n" "$start_marker"
        printf "%s\n" "$block_content"
        printf "%s\n" "$end_marker"
    } >> "$file"
}

find_python311() {
    if [ -n "$PYTHON_BIN" ]; then
        if [ ! -x "$PYTHON_BIN" ]; then
            echo "  ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
            exit 1
        fi
        "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'
        echo "$PYTHON_BIN"
        return
    fi

    if command -v python3.11 >/dev/null 2>&1; then
        command -v python3.11
        return
    fi

    if command -v python3 >/dev/null 2>&1 && \
       python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' >/dev/null 2>&1; then
        command -v python3
        return
    fi

    if [ -x /opt/conda/bin/python3.11 ]; then
        echo /opt/conda/bin/python3.11
        return
    fi

    if command -v conda >/dev/null 2>&1; then
        local conda_env="ocrfull-py311"
        local conda_base=""
        local conda_python=""

        conda_base="$(conda info --base 2>/dev/null | tail -n 1 | tr -d '\r')"
        if ! conda run -n "$conda_env" python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' >/dev/null 2>&1; then
            echo "  Python 3.11 not found; creating conda env '$conda_env'..." >&2
            conda create -y -n "$conda_env" python=3.11 >&2
        fi

        if [ -n "$conda_base" ]; then
            conda_python="$conda_base/envs/$conda_env/bin/python"
            if [ -x "$conda_python" ]; then
                echo "$conda_python"
                return
            fi
        fi

        conda_python="$(conda run -n "$conda_env" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -n 1 | tr -d '\r')"
        if [ -n "$conda_python" ] && [ -x "$conda_python" ]; then
            echo "$conda_python"
            return
        fi
    fi

    echo "  ERROR: Python 3.11 was not found. Install it or rerun with PYTHON_BIN=/path/to/python3.11." >&2
    exit 1
}

echo ""
echo "[1/6] Installing system dependencies..."
if [ "$SKIP_APT" = "1" ]; then
    echo "  SKIP_APT=1 -> skipping apt-get packages"
else
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "  ERROR: apt-get not found. Install the Linux packages manually or rerun with SKIP_APT=1." >&2
        exit 1
    fi

    if [ "$(id -u)" -eq 0 ]; then
        APT_PREFIX=()
    elif command -v sudo >/dev/null 2>&1; then
        APT_PREFIX=(sudo)
    else
        echo "  ERROR: sudo is required for apt-get. Install packages manually or rerun with SKIP_APT=1." >&2
        exit 1
    fi

    "${APT_PREFIX[@]}" apt-get update -q
    "${APT_PREFIX[@]}" apt-get install -y \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        poppler-utils
fi

if command -v tesseract >/dev/null 2>&1; then
    echo "  Tesseract version: $(tesseract --version 2>&1 | head -1)"
else
    echo "  WARNING: tesseract not found on PATH"
fi

echo ""
echo "[2/6] Detecting CUDA..."
CUDA_VERSION=""
CUDA_MAJOR=""
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    CUDA_VERSION="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)"
fi

if [ -z "$CUDA_VERSION" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION="$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || true)"
fi

if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR="${CUDA_VERSION%%.*}"
    echo "  CUDA version: $CUDA_VERSION"
else
    echo "  WARNING: CUDA tools not detected. Falling back to CPU package installs where possible."
fi

echo ""
echo "[3/6] Finding Python 3.11..."
PYTHON311="$(find_python311)"
echo "  Using: $PYTHON311 ($("$PYTHON311" --version))"

echo ""
echo "[4/6] Configuring persistent model cache..."
mkdir -p "$MODEL_CACHE_DIR/.paddleocr"
mkdir -p "$MODEL_CACHE_DIR/.cache/huggingface/hub"

rewrite_managed_block "$BASHRC" \
    "# OCRFULL - persistent model cache (start)" \
    "# OCRFULL - persistent model cache (end)" \
    "export PADDLEOCR_HOME=\"$MODEL_CACHE_DIR/.paddleocr\"
export HF_HOME=\"$MODEL_CACHE_DIR/.cache/huggingface\"
export HUGGINGFACE_HUB_CACHE=\"$MODEL_CACHE_DIR/.cache/huggingface/hub\""

export PADDLEOCR_HOME="$MODEL_CACHE_DIR/.paddleocr"
export HF_HOME="$MODEL_CACHE_DIR/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE_DIR/.cache/huggingface/hub"
echo "  Cache exports refreshed in ~/.bashrc"

if [ -n "$GPU_INDEX" ]; then
    rewrite_managed_block "$BASHRC" \
        "# OCRFULL - GPU targeting (start)" \
        "# OCRFULL - GPU targeting (end)" \
        "export CUDA_VISIBLE_DEVICES=$GPU_INDEX"
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    echo "  Persisted CUDA_VISIBLE_DEVICES=$GPU_INDEX in ~/.bashrc"
else
    echo "  GPU targeting unchanged. Pass GPU_INDEX=<index> if you want to pin a device."
fi

echo ""
echo "[5/6] Setting up Pipeline 1 (PaddleOCR + Tesseract)..."
P1_DIR="$SCRIPT_DIR/paddleocr_testing"
P1_VENV="$P1_DIR/paddle-env"
P1_PYTHON="$P1_VENV/bin/python"

rm -rf "$P1_VENV"
"$PYTHON311" -m venv "$P1_VENV"
"$P1_PYTHON" -m pip install --upgrade pip wheel -q

if [ -n "$CUDA_MAJOR" ]; then
    echo "  Installing PaddlePaddle GPU 3.3.x..."
    if [ "$CUDA_MAJOR" -ge 12 ] && [ "${CUDA_MINOR:-0}" -ge 6 ]; then
        "$P1_PYTHON" -m pip install "paddlepaddle-gpu==3.3.0" \
            -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ -q
    else
        "$P1_PYTHON" -m pip install "paddlepaddle-gpu==3.3.0" \
            -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ -q
    fi
else
    echo "  Installing CPU PaddlePaddle 3.3.x..."
    "$P1_PYTHON" -m pip install "paddlepaddle==3.3.0" \
        -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ -q
fi

"$P1_PYTHON" -m pip install -r "$P1_DIR/requirements.txt" -q
echo "  Pipeline 1 venv ready: $P1_VENV"

echo ""
echo "[6/6] Setting up Pipeline 2 (Surya + Marker)..."
P2_DIR="$SCRIPT_DIR/surya_marker_testing"
P2_VENV="$P2_DIR/surya-env"
P2_PYTHON="$P2_VENV/bin/python"
MARKER_CONDA="${MARKER_CONDA:-/opt/conda/envs/marker}"

rm -rf "$P2_VENV"

if command -v conda >/dev/null 2>&1 && \
   [ -d "$MARKER_CONDA" ] && \
   conda run -p "$MARKER_CONDA" python -c "import surya, marker, pypdfium2" >/dev/null 2>&1; then
    echo "  Reusing existing 'marker' conda env at $MARKER_CONDA"
    ln -s "$MARKER_CONDA" "$P2_VENV"
else
    echo "  Building a fresh Surya/Marker venv..."
    "$PYTHON311" -m venv "$P2_VENV"
    "$P2_PYTHON" -m pip install --upgrade pip wheel -q

    "$P2_PYTHON" -m pip install -r "$P2_DIR/requirements.txt" -q

    echo "  Pinning PyTorch to a known-good version pair..."
    if [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 12 ]; then
        "$P2_PYTHON" -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 \
            --index-url https://download.pytorch.org/whl/cu121 -q
    elif [ -n "$CUDA_MAJOR" ]; then
        "$P2_PYTHON" -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 \
            --index-url https://download.pytorch.org/whl/cu118 -q
    else
        "$P2_PYTHON" -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 \
            --index-url https://download.pytorch.org/whl/cpu -q
    fi
fi

echo "  Pipeline 2 ready: $P2_VENV"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Run the benchmark:"
echo "    cd $SCRIPT_DIR"
echo "    $PYTHON311 benchmark.py --skip-p1"
echo "    $PYTHON311 benchmark.py --skip-p2"
echo "    $PYTHON311 benchmark.py"
echo ""
echo "  Model cache:"
echo "    PaddleOCR  : $PADDLEOCR_HOME"
echo "    HuggingFace: $HF_HOME"
echo ""
echo "  NOTE: First run downloads models (~2-5 GB)."
echo "============================================================"
