#!/bin/bash
# build_portable.sh — Build portable release artifacts for GitHub Releases.
#
# Usage:
#   ./build_portable.sh                       # build oba tarbally (env + source)
#   ./build_portable.sh --env-only            # iba env tarball
#   ./build_portable.sh --source-only         # iba source tarball
#   ./build_portable.sh --env-name foobar     # custom env name (default: videotranslator)
#
# Output:
#   release/videotranslator-source-vX.Y.Z-linux.tar.gz   (~3 MB)
#   release/videotranslator-env-linux-x64.tar.gz         (~2 GB)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$SCRIPT_DIR/release"
ENV_NAME="${VTS_ENV_NAME:-videotranslator}"
VERSION="${VTS_VERSION:-0.1.0}"
BUILD_ENV=true
BUILD_SOURCE=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-only)    BUILD_SOURCE=false; shift ;;
        --source-only) BUILD_ENV=false; shift ;;
        --env-name)    ENV_NAME="$2"; shift 2 ;;
        --version)     VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RELEASE_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[build]${NC} $*"; }
warn() { echo -e "${YELLOW}[build]${NC} $*"; }

# ── 1. Env tarball (conda-pack) ───────────────────────────────────────────────
if $BUILD_ENV; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda nie je v PATH. Source conda activate first." >&2
        exit 1
    fi

    # Source conda profile to enable activate
    CONDA_SH="$(dirname "$(dirname "$(command -v conda)")")/etc/profile.d/conda.sh"
    [[ -f "$CONDA_SH" ]] && source "$CONDA_SH"

    # Verify env exists
    if ! conda env list | grep -q "^$ENV_NAME "; then
        echo "ERROR: conda env '$ENV_NAME' nenájdený. Vytvor cez install.sh alebo prepni cez --env-name." >&2
        exit 1
    fi

    # Install conda-pack do env-u (1× setup)
    if ! conda run -n "$ENV_NAME" python -c "import conda_pack" 2>/dev/null; then
        info "Inštalujem conda-pack do '$ENV_NAME'..."
        conda install -n "$ENV_NAME" -y -c conda-forge conda-pack
    fi

    OUT_ENV="$RELEASE_DIR/videotranslator-env-linux-x64.tar.gz"
    info "Pakujem env '$ENV_NAME' → $OUT_ENV (5-10 min)..."
    conda pack -n "$ENV_NAME" -o "$OUT_ENV" --force
    SIZE=$(du -h "$OUT_ENV" | cut -f1)
    info "✓ Env tarball: $OUT_ENV ($SIZE)"
fi

# ── 2. Source tarball ─────────────────────────────────────────────────────────
if $BUILD_SOURCE; then
    OUT_SRC="$RELEASE_DIR/videotranslator-source-v${VERSION}.tar.gz"
    info "Pakujem source → $OUT_SRC..."
    cd "$SCRIPT_DIR"
    tar czf "$OUT_SRC" \
        --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
        --exclude='.env' --exclude='musetalk_gui_config.json' --exclude='.codex' --exclude='.claude' \
        --exclude='memory' --exclude='data' --exclude='output' --exclude='temp' --exclude='work' \
        --exclude='input' --exclude='models' --exclude='fish_speech' --exclude='voices/*.wav' \
        --exclude='unsloth_compiled_cache' --exclude='*.bak*' --exclude='.git' \
        --exclude='release' --exclude='env' \
        --exclude='*_segments.json' --exclude='*_tts_*.json' --exclude='*_timing_*.json' \
        --exclude='batch_checkpoint.json' --exclude='video_lists.json' --exclude='cps_calibration.json' \
        --transform "s,^,videotranslator-studio-v${VERSION}/," \
        VideoTranslator_studio.py main.py launch_studio.sh launch_studio.ps1 launch_studio.bat \
        install.sh install.ps1 \
        scripts/ src/ languages/ prompts/ mini_level11/ \
        run_*.py requirements.txt README.md LICENSE \
        .env.example .gitignore VideoTranslator_studio.desktop \
        2>/dev/null || true
    SIZE=$(du -h "$OUT_SRC" | cut -f1)
    info "✓ Source tarball: $OUT_SRC ($SIZE)"
fi

# ── 3. Hotovo ─────────────────────────────────────────────────────────────────
info ""
info "Hotovo. Súbory v $RELEASE_DIR/:"
ls -lh "$RELEASE_DIR/" 2>/dev/null
info ""
info "Pre GitHub Release upload:"
info "  gh release create v${VERSION} \\"
info "    \"$RELEASE_DIR/\"*.tar.gz \\"
info "    --title \"VideoTranslator Studio v${VERSION}\" \\"
info "    --notes-file CHANGELOG.md"
