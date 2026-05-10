#!/bin/bash
# End-to-end test: VTS pipeline s konzistentnou EuroLLM-9B trojicou
# vs pôvodným Gemma-4-26B + G3-12B run.
#
# Pipeline:
#   Whisper STT  →  V2 translate (LM Studio API)
#                →  Compressor V1 (LM Studio API, ak overflow)
#                →  Grammar V1 (lokálny llama.cpp embedded)
#                →  OmniVoice TTS  →  Render
#
# Output: temp/2 Years of C++ Programming_eurollm/  (oddelené od pôvodného Gemma run)

set -e

VIDEO="/mnt/tts_data/VideoTranslator_studio_v2/input/2 Years of C++ Programming.mp4"
PYBIN="/mnt/tts_data/miniforge3/envs/chatterbox_env/bin/python"  # VTS default — má soundfile, torch, whisperx, llama_cpp

# ---- LM Studio model IDs ----
TRANSLATE_MODEL="vojtech/eurollm-9b-sk-translate-v3-gguf/eurollm9b-sk-translate-v3-q5km.gguf"
COMPRESSOR_MODEL="vojtech/eurollm-9b-sk-compressor-v1-gguf/eurollm9b-sk-compressor-v1-q5km.gguf"
# ---- Local llama.cpp GGUF for refine ----
GRAMMAR_GGUF="/mnt/tts_data/knihy/gguf_grammar_eurollm9b_v1/eurollm9b-sk-grammar-v1-q5km.gguf"

# Pre-checks
echo "=== Pre-flight check ==="
if ! curl -s --max-time 3 http://localhost:1234/v1/models > /dev/null; then
    echo "ERROR: LM Studio nie je spustené na :1234. Spusti LM Studio app najprv."
    exit 1
fi
echo "  LM Studio :1234 OK"

if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video neexistuje: $VIDEO"
    exit 1
fi
echo "  Video OK: $VIDEO"

if [ ! -f "$GRAMMAR_GGUF" ]; then
    echo "WARN: Grammar GGUF ešte neexistuje: $GRAMMAR_GGUF"
    echo "      Refine pass bude vypnutý."
    REFINE_ARGS=""
else
    REFINE_ARGS="--refine_translation --refine_model $GRAMMAR_GGUF --refine_gpu_layers -1 --llama_ctx 4096"
    echo "  Grammar GGUF OK: $GRAMMAR_GGUF"
fi

# Pre nepoškodenie pôvodného Gemma 26B output — output dir s suffixom
OUTPUT_SUFFIX="_eurollm_e2e"
echo
echo "=== Spustam end-to-end test ==="
echo "  Translate: $TRANSLATE_MODEL"
echo "  Compress:  $COMPRESSOR_MODEL"
echo "  Refine:    $GRAMMAR_GGUF (alebo skip ak chyba)"
echo "  Output suffix: $OUTPUT_SUFFIX"
echo

cd /mnt/tts_data/VideoTranslator_studio_v2

VTS_LMSTUDIO_COMPRESSOR_MODEL="$COMPRESSOR_MODEL" \
VTS_LMSTUDIO_GRAMMAR_MODEL="$GRAMMAR_GGUF" \
"$PYBIN" main.py \
    --input "$VIDEO" \
    --tgt_lang slk \
    --use_lmstudio_translate \
    --lmstudio_translate_model "$TRANSLATE_MODEL" \
    --lmstudio_translate_url "http://localhost:1234/v1" \
    --lmstudio_translate_timeout 180 \
    --no_adapt_llm \
    --speaker_gender feminine \
    --no-phonetic_respelling \
    $REFINE_ARGS \
    --use_omnivoice \
    --timed \
    --vad \
    2>&1 | tee /mnt/tts_data/knihy/logs/e2e_eurollm_v2_run.log

echo
echo "=== Hotovo ==="
echo "  Log: /mnt/tts_data/knihy/logs/e2e_eurollm_run.log"
echo "  Output: /mnt/tts_data/VideoTranslator_studio_v2/temp/2 Years of C++ Programming/"
echo
echo "Compare s povodnym Gemma 26B run:"
echo "  diff <(jq -r '.segments[].text' temp/2*C++*sk_segments.json | head -20) ..."
