#!/bin/bash
# Reset VideoTranslator Studio do "first run" stavu.
# Použi keď chceš odovzdať čistú instaláciu — všetky settings sa resetujú na defaults.
#
# NEZmaže: source kód, modely, voices/, languages/.
# ZMaže: GUI config, paths config, checkpoint state, runtime cache.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Reset VideoTranslator Studio na first-run state ==="
echo "Script dir: $SCRIPT_DIR"
echo

echo "→ GUI config (musetalk_gui_config.json)"
rm -f "$SCRIPT_DIR/musetalk_gui_config.json"

echo "→ paths.py config (~/.config/videotranslator/paths.json)"
rm -f "$HOME/.config/videotranslator/paths.json"

echo "→ Checkpoint state files"
rm -f "$SCRIPT_DIR/batch_checkpoint.json"
rm -f "$SCRIPT_DIR/video_lists.json"
rm -f "$SCRIPT_DIR/saved_checkpoints.json"
rm -f "$SCRIPT_DIR/cps_calibration.json"
rm -f "$SCRIPT_DIR"/profiles/*.json 2>/dev/null

echo "→ Runtime cache (temp/, work/)"
rm -rf "$SCRIPT_DIR/temp"
rm -rf "$SCRIPT_DIR/work"

echo "→ TTS hash meta cache"
find "$SCRIPT_DIR" -maxdepth 4 -name "tts_*.meta.json" -delete 2>/dev/null

echo "→ Voice ref-text cache (audio cache zachovaná)"
rm -rf "$SCRIPT_DIR/voices/.ref_text_cache"

echo "→ Python __pycache__"
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "→ Output / data adresáre (ak existujú)"
rm -rf "$SCRIPT_DIR/output"
rm -rf "$SCRIPT_DIR/data"

echo
echo "✓ Reset hotový. Pri ďalšom spustení GUI bude:"
echo "  - First-run model bootstrap dialog (ak modely chýbajú)"
echo "  - Bootstrap paths dialog (ak miniforge/models cesty neexistujú)"
echo "  - Default TTS engine = OmniVoice"
echo "  - Default translation engine = lmstudio"
echo "  - Default refine = Gemma 3 12B v1 (Ollama)"
echo
echo "Prázdne settings, žiadny saved state."
