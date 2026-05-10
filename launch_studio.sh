#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/mnt/tts_data/miniforge3/envs/musetalk_env/bin/python"
# llama_cpp v musetalk_env potrebuje CUDA knižnice z nvidia balíčkov
_NVIDIA="/mnt/tts_data/miniforge3/envs/musetalk_env/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="\
${_NVIDIA}/cuda_runtime/lib:\
${_NVIDIA}/cublas/lib:\
${_NVIDIA}/cusparse/lib:\
${_NVIDIA}/nvjitlink/lib:\
${_NVIDIA}/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SCRIPT_DIR"
exec "$PYTHON" "$SCRIPT_DIR/VideoTranslator_studio.py" "$@"
