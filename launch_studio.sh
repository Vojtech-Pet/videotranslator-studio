#!/bin/bash
# VideoTranslator Studio launcher — auto-detects conda + nvidia CUDA libs.
# Override paths via env vars:
#   VTS_PYTHON       — explicit python interpreter path
#   VTS_PYTHON_ENV   — conda env name (default: musetalk_env)
#   VTS_CONDA_ROOT   — explicit conda installation root
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) Locate python interpreter — priorita:
#    a) explicit VTS_PYTHON env var
#    b) PORTABLE env v ./env/ (conda-pack tarball, prenosný)
#    c) systemový conda v štandardných cestách
#    d) fallback na system python3
PYTHON=""
if [[ -n "${VTS_PYTHON:-}" && -x "$VTS_PYTHON" ]]; then
    PYTHON="$VTS_PYTHON"
elif [[ -x "$SCRIPT_DIR/env/bin/python" ]]; then
    # Portable bundled env (conda-pack)
    # Prvé spustenie: spusti conda-unpack ktorý opraví relocation paths
    if [[ -x "$SCRIPT_DIR/env/bin/conda-unpack" && ! -f "$SCRIPT_DIR/env/.unpacked" ]]; then
        echo "[launch] Prvý štart portable env-u — opravujem cesty (môže to trvať pár sekúnd)..."
        "$SCRIPT_DIR/env/bin/conda-unpack" && touch "$SCRIPT_DIR/env/.unpacked"
    fi
    PYTHON="$SCRIPT_DIR/env/bin/python"
    echo "[launch] Using portable env: $SCRIPT_DIR/env"
else
    # Skús viacero env mien: VTS_PYTHON_ENV → videotranslator (install.sh) → musetalk_env (legacy)
    PY_ENVS=("${VTS_PYTHON_ENV:-}" "videotranslator" "musetalk_env")
    CONDA_ROOTS=(
        "${VTS_CONDA_ROOT:-}"
        "$HOME/miniforge3"
        "/mnt/tts_data/miniforge3"
        "/opt/miniforge3"
        "/opt/conda"
        "/usr/local/miniforge3"
        "$HOME/anaconda3"
        "$HOME/miniconda3"
    )
    for env_name in "${PY_ENVS[@]}"; do
        [[ -z "$env_name" ]] && continue
        for root in "${CONDA_ROOTS[@]}"; do
            [[ -z "$root" ]] && continue
            if [[ -x "$root/envs/$env_name/bin/python" ]]; then
                PYTHON="$root/envs/$env_name/bin/python"
                PY_ENV="$env_name"
                break 2
            fi
        done
    done
    if [[ -z "$PYTHON" ]] && command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
        echo "[launch] WARN: conda env '$PY_ENV' not found — using system $PYTHON" >&2
    fi
fi

if [[ -z "$PYTHON" ]]; then
    echo "[launch] ERROR: python3 not found." >&2
    echo "  Set VTS_PYTHON or VTS_CONDA_ROOT, or install miniforge3 + create env '$PY_ENV'." >&2
    exit 1
fi

# 2) Add nvidia CUDA libs (needed for llama-cpp GPU). Optional — skipped if missing.
PY_PREFIX="$(dirname "$(dirname "$PYTHON")")"
_NV=""
for pyver in python3.10 python3.11 python3.12 python3.9 python3.13; do
    if [[ -d "$PY_PREFIX/lib/$pyver/site-packages/nvidia" ]]; then
        _NV="$PY_PREFIX/lib/$pyver/site-packages/nvidia"
        break
    fi
done
if [[ -n "$_NV" ]]; then
    NV_PATHS=""
    for sub in cuda_runtime cublas cusparse nvjitlink cuda_nvrtc; do
        [[ -d "$_NV/$sub/lib" ]] && NV_PATHS+="$_NV/$sub/lib:"
    done
    [[ -n "$NV_PATHS" ]] && export LD_LIBRARY_PATH="${NV_PATHS}${LD_LIBRARY_PATH:-}"
fi

# 3) Launch
cd "$SCRIPT_DIR"
exec "$PYTHON" "$SCRIPT_DIR/VideoTranslator_studio.py" "$@"
