"""Config: argument parser and constants for VideoTranslator."""
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()  # VideoTranslator/
_S2_PRO_MODEL_CANDIDATES = [
    SCRIPT_DIR / "fish_speech" / "repo" / "checkpoints" / "s2-pro",
    SCRIPT_DIR.parent / "VideoTranslator" / "fish_speech" / "repo" / "checkpoints" / "s2-pro",
]
_DEFAULT_S2_PRO_MODEL = next((p for p in _S2_PRO_MODEL_CANDIDATES if p.exists()), _S2_PRO_MODEL_CANDIDATES[0])

# ---------------------------------------------------------------------------
# Audio constants — single source of truth
# ---------------------------------------------------------------------------
WHISPER_SR: int = 16000   # faster-whisper input sample rate
TTS_SR: int = 24000       # Chatterbox output sample rate

# Subprocess timeouts (seconds)
FFPROBE_TIMEOUT: int = 30
FFMPEG_TIMEOUT: int = 300


def target_lang_name(code: str) -> str:
    mapping = {
        "ces": "Czech",
        "cs": "Czech",
        "slk": "Slovak",
        "sk": "Slovak",
        "en": "English",
        "eng": "English",
        "de": "German",
        "deu": "German",
        "fr": "French",
        "fra": "French",
        "es": "Spanish",
        "spa": "Spanish",
        "ru": "Russian",
        "rus": "Russian",
        "it": "Italian",
        "ita": "Italian",
        "pl": "Polish",
    }
    return mapping.get(code, code)


TTS_EQ_PROFILES: dict = {
    "Hegen": [
        # XTTS v2 → HeyGen-quality feel: warm, clean, natural presence
        "highpass=f=100",
        "equalizer=f=250:t=q:w=1.2:g=-1.5",    # reduce boxiness
        "equalizer=f=450:t=q:w=1.0:g=+1.2",    # add warmth
        "equalizer=f=900:t=q:w=0.8:g=-0.8",    # reduce muddiness
        "equalizer=f=2200:t=q:w=1.0:g=+1.0",   # add presence/clarity
        "equalizer=f=4000:t=q:w=1.2:g=-1.5",   # tame harshness
        "equalizer=f=8000:t=q:w=1.0:g=+0.8",   # add air
        "lowpass=f=12000",
    ],
    "Warm Voice": [
        "highpass=f=45",
        "equalizer=f=90:t=q:w=1:g=3.2",
        "equalizer=f=220:t=q:w=1:g=1.8",
        "equalizer=f=2800:t=q:w=1:g=-1.2",
        "lowpass=f=10500",
    ],
    "Speech Clarity": [
        "highpass=f=60",
        "equalizer=f=170:t=q:w=1:g=-1.0",
        "equalizer=f=3200:t=q:w=1:g=2.2",
        "equalizer=f=4500:t=q:w=1:g=1.6",
        "lowpass=f=12000",
    ],
    "Podcast Deep": [
        "highpass=f=40",
        "equalizer=f=85:t=q:w=0.9:g=3.6",
        "equalizer=f=160:t=q:w=1:g=2.1",
        "equalizer=f=2800:t=q:w=1:g=-1.4",
        "equalizer=f=5000:t=q:w=1.2:g=-1.8",
        "lowpass=f=10000",
    ],
    "Radio Mid Focus": [
        "highpass=f=80",
        "equalizer=f=180:t=q:w=1:g=-2.0",
        "equalizer=f=1200:t=q:w=1:g=1.8",
        "equalizer=f=2800:t=q:w=1:g=2.6",
        "equalizer=f=5200:t=q:w=1:g=-1.4",
        "lowpass=f=9000",
    ],
    # ── Chatterbox TTS profiles ─────────────────────────────────────────────
    # HiFTGenerator vocoder (24kHz): no content >12kHz, boxiness 300-600Hz,
    # nasality 800-1.3kHz, harshness 3-5kHz, weak high-end (vocoder ceiling).
    # Source: GitHub issues devnen/Chatterbox-TTS-Server, Hacker News #44251411,
    #         arxiv HiFi-GAN vocoder research, Podigy podcast EQ guidelines.
    "Chatterbox Natural": [
        # Light general-purpose: remove boxiness, tame harshness, smooth highs
        "highpass=f=80",
        "equalizer=f=350:t=q:w=1.2:g=-2.5",    # cut boxiness (300-400 Hz)
        "equalizer=f=700:t=q:w=1.0:g=-1.5",    # cut nasality / mud
        "equalizer=f=2500:t=q:w=1.0:g=+1.5",   # presence / intelligibility
        "equalizer=f=5000:t=q:w=0.8:g=-2.0",   # tame sibilance / buzz
        "lowpass=f=11000",                       # znížený strop namiesto shelfu (menej filtrov)
    ],
    "Chatterbox Voice Clone": [
        # Voice cloning mode – typically more muffled/boxy, needs stronger treatment
        "highpass=f=80",
        "equalizer=f=280:t=q:w=1.2:g=-3.5",    # strong boxiness cut
        "equalizer=f=600:t=q:w=1.0:g=-2.5",    # mud cut
        "equalizer=f=950:t=q:w=1.0:g=-2.0",    # nasality cut
        "equalizer=f=2200:t=q:w=1.0:g=+1.8",   # presence
        "equalizer=f=4500:t=q:w=0.8:g=-2.5",   # harshness / vocoder buzz
        "equalizer=f=7000:t=q:w=1.0:g=-2.0",   # de-ess
        "lowpass=f=10500",                       # znížený strop namiesto shelfu
    ],
    "Chatterbox Audiobook": [
        # ACX-compatible: clean, warm, intelligible – no harsh peaks
        "highpass=f=80",
        "equalizer=f=350:t=q:w=1.5:g=-3.0",    # boxiness
        "equalizer=f=700:t=q:w=1.0:g=-2.0",    # mud
        "equalizer=f=2500:t=q:w=1.0:g=+1.0",   # clarity
        "equalizer=f=5000:t=q:w=0.5:g=-2.0",   # harshness
        "lowpass=f=11000",                       # znížený strop namiesto shelfu
    ],
    "Chatterbox Podcast": [
        # Punchy, forward-sounding for narration / dubbing
        "highpass=f=80",
        "equalizer=f=200:t=q:w=1.0:g=+1.5",    # body / warmth
        "equalizer=f=400:t=q:w=1.2:g=-2.0",    # boxiness
        "equalizer=f=800:t=q:w=1.0:g=-1.8",    # mud / honk
        "equalizer=f=3000:t=q:w=1.0:g=+2.0",   # presence / punch
        "equalizer=f=5500:t=q:w=0.8:g=-2.5",   # harshness
        "lowpass=f=11000",                       # znížený strop namiesto shelfu
    ],
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input video file")
    ap.add_argument("--tgt_lang", default="ces")
    ap.add_argument("--whisper_model", default="large-v3-turbo", help="Name of the Whisper model to use")
    ap.add_argument("--whisper_root", default=str(SCRIPT_DIR / "models" / "faster-whisper"), help="Path where faster-whisper model is stored (CTranslate2 format)")
    ap.add_argument("--whisper_prompt", default="", help="Initial prompt for Whisper to improve recognition (e.g. topic/terminology hints)")
    ap.add_argument("--whisper_no_speech_max", type=float, default=0.6,
                    help="Whisper hallucination guard: max allowed no_speech_prob before dropping a segment.")
    ap.add_argument("--whisper_avg_logprob_min", type=float, default=-1.0,
                    help="Whisper hallucination guard: minimum avg_logprob before dropping a segment.")
    ap.add_argument("--whisper_compression_max", type=float, default=2.4,
                    help="Whisper hallucination guard: maximum compression ratio before dropping a segment.")
    ap.add_argument("--demucs", action="store_true", help="Use Demucs vocal isolation before Whisper transcription (removes music/noise)")
    ap.add_argument("--script_mode", action="store_true", help="Script mode: Whisper→clean EN→translate full text→TTS by sentences. Gives natural pacing instead of per-segment translation.")
    ap.add_argument("--script_use_madlad", action="store_true", help="In script_mode: use MADLAD for translation instead of Gemma. Pipeline: Gemma cleans EN → MADLAD translates. Faster, no hallucinations.")
    ap.add_argument("--script_madlad_model", default="google/madlad400-7b-mt", help="MADLAD model for script_mode translation (default: 7b-mt, use 3b-mt for speed).")
    ap.add_argument("--translate_prompt", default="", help="Custom system prompt for translation (overrides default). Use {lang} for target language name, {words} for word count, {duration} for segment duration.")
    ap.add_argument("--llama_model", default="/mnt/tts_data/VideoTranslator_studio/models/gemma-4-26B/gemma-4-26B-A4B-it-Q4_K_M.gguf")
    ap.add_argument("--no_adapt_llm", action="store_true", help="Zakáž LLM pre text adaptation (len stretch) — odporúča sa pre google/translategemma enginy")
    ap.add_argument("--llama_ctx", type=int, default=4096)
    ap.add_argument("--llama_gpu_layers", type=int, default=-1)
    ap.add_argument("--llama_temp", type=float, default=0.05)
    ap.add_argument("--context_aware", action="store_true", help="Two-pass translation: first create glossary, then translate with context")
    ap.add_argument("--content_type", default="general", choices=["general", "educational", "podcast", "review", "technical", "news", "entertainment", "sql", "programming"],
                    help="Content type for translation style: general, educational (lectures), podcast (conversation), review, technical (scientific), news, entertainment, sql (SQL/database), programming (C++/game dev)")
    ap.add_argument(
        "--mixed_technical_terms",
        action="store_true",
        help="For technical Slovak output, keep important Linux/CLI/product terms in English inside Slovak sentences (e.g. Proxmox, SSH, apt update, unattended-upgrades).",
    )
    ap.add_argument("--preset", default="", choices=["", "production_dub", "teaching_sql"],
                    help="Pipeline preset. 'production_dub': clean dubbing (phonetic OFF, SQL-TTS OFF, light normalizer). "
                         "'teaching_sql': SQL/IT teaching (all normalizers ON, SQL phonetics ON).")
    ap.add_argument("--phonetic_respelling", action=argparse.BooleanOptionalAction, default=None,
                    help="Enable/disable phonetic respelling map. Default: ON for general content, OFF for production_dub preset. "
                         "Use --no_phonetic_respelling to force off.")
    ap.add_argument("--ab_export", action="store_true", default=False,
                    help="A/B export: pre každý segment uloží raw / pause_only / pause+f0 WAV do chunks/ab_export/. "
                         "Zapni pri ladení post-processingu.")
    ap.add_argument("--postproc_metrics", action=argparse.BooleanOptionalAction, default=True,
                    help="Vypočítaj dur/RMS/F0 metriky pred a po post-processingu. Vyžaduje --ab_export pre plné porovnanie.")
    ap.add_argument("--postproc_dur_limit", type=float, default=0.50,
                    help="Hard limit na duration drift po post-processingu (sekundy). "
                         "Ak dur_delta > limit → auto-fallback na pause_only alebo raw. Default: 0.50s")
    ap.add_argument("--f0_transfer", action=argparse.BooleanOptionalAction, default=False,
                    help="F0 contour transfer: prenáša tvar intonácie z originálu na TTS výstup cez pyworld. "
                         "Experimentálne — WORLD resyntéza môže degradovať kvalitu. "
                         "Vyžaduje --emotion_clone. Zapnúť: --f0_transfer")
    ap.add_argument("--f0_blend_scale", type=float, default=1.0,
                    help="Škálovací faktor pre F0 blend weight (1.0 = default váhy, 1.44 ≈ staré agresívne váhy). "
                         "Použiť pri A/B testovaní. Platné rozsahy: 0.0–2.0.")
    ap.add_argument("--wav_pause_injection", action=argparse.BooleanOptionalAction, default=True,
                    help="WAV-level pause injection: vkladá presné tichá (ms) do TTS audia podľa pause mapy originálu. "
                         "Vyžaduje --emotion_clone. Use --no_wav_pause_injection to disable.")
    ap.add_argument("--sql_tts_normalize", action="store_true", default=False,
                    help="Enable SQL-to-speech normalization (normalize_sql_for_tts). OFF by default. Auto-enabled for --content_type sql or --preset teaching_sql.")
    ap.add_argument("--normalizer_mode", default="", choices=["", "light", "full"],
                    help="Text normalizer mode: 'light' = numbers/dates/percentages only; 'full' = all (currency, units, ordinals too). Default: 'light' for production_dub, 'full' for teaching_sql.")
    ap.add_argument("--use_existing_segments", action="store_true",
                    help="Skip STT + translation + adaptation. Load pre-existing {stem}_{tgt}_segments.json and go directly to TTS + assembly.")
    ap.add_argument("--use_madlad", action="store_true", help="Use MADLAD-400 MT for translation (faster than Llama). If --context_aware is also set, Llama creates glossary and MADLAD translates.")
    ap.add_argument("--madlad_model", default="google/madlad400-7b-mt", help="MADLAD model: 3b-mt (fast), 7b-mt, or 10b-mt (better quality)")
    ap.add_argument("--use_multislav5lang", action="store_true", help="Use Allegro MultiSlav-5lang MT for translation (supports cs/en/pl/sk/sl, very fast local Marian model)")
    ap.add_argument("--multislav_model", default="allegro/multislav-5lang", help="Hugging Face model id or local path for MultiSlav-5lang")
    ap.add_argument("--use_nllb", action="store_true", help="Use NLLB-200 for translation (Meta, local, no internet, auto-detects source language from Whisper)")
    ap.add_argument("--use_ollama_translate", action="store_true", help="Use Ollama HTTP API for primary translation (e.g. sk-gemma4-26b). Fastest when model is already loaded in Ollama. Combine with --ollama_translate_model.")
    ap.add_argument("--ollama_translate_model", default="sk-gemma4-26b:latest", help="Ollama model name for primary translation (default: sk-gemma4-26b:latest).")
    ap.add_argument("--ollama_translate_url", default="http://localhost:11434/api/generate", help="Ollama generate endpoint for primary translation.")
    ap.add_argument("--ollama_translate_timeout", type=int, default=180, help="Timeout in seconds per segment for Ollama translation (default: 180).")
    ap.add_argument("--grammar_fix_only", action="store_true", help="Skip translation entirely — transcribe source audio as-is and run only Slovak grammar fix (Ollama sk-gemma4-e4b-v6). Use when source is already Slovak.")
    ap.add_argument("--use_google_translate", action="store_true", help="Use free Google Translate (mobile endpoint, no API key). Better quality than MADLAD for most languages.")
    ap.add_argument("--hybrid_translate", action="store_true", help="Hybrid: Google Translate primary (better quality) + MADLAD fallback for failed segments and SQL/code-heavy content.")
    ap.add_argument("--hybrid_qa", action="store_true", help="After hybrid translation, run LLM grammar/fluency QA pass (uses --llama_model). Requires --hybrid_translate.")
    ap.add_argument("--use_gemma4_hf", action="store_true", help="Use Gemma 4 E4B (HuggingFace transformers) for full-document segment-by-segment translation. SQL-aware prompt; better SQL term accuracy than GGUF. Combine with --refine_translation for best results.")
    ap.add_argument("--gemma4_hf_model", default="/mnt/tts_data/VideoTranslator_studio/models/gemma-4-E4B-it", help="Path to Gemma 4 E4B model directory (HuggingFace format). Default: /mnt/tts_data/VideoTranslator_studio/models/gemma-4-E4B-it")
    ap.add_argument("--max_chars", type=int, default=250)
    ap.add_argument("--tts_max_chars", type=int, default=80, help="Max chars per chunk for non-timed TTS mode")
    ap.add_argument("--out_dir", default="", help="Output directory for all non-video output files (JSON, WAV, SRT). Default: same directory as input.")
    ap.add_argument("--text_out", default="")
    ap.add_argument("--text_in", default="")
    ap.add_argument("--srt_out", default="", help="Output SRT subtitle file with timing (start/end for each segment)")
    ap.add_argument("--srt_align", action="store_true", help="Generate SRT aligned to the final TTS audio using WhisperX forced alignment (word-level precision). Saves {stem}_{lang}_tts.srt")
    ap.add_argument("--json_out", default="", help="Output JSON file with detailed segment timing information")
    ap.add_argument("--translation_only", action="store_true", help="Stop after translation/audit/JSON export and skip TTS.")
    ap.add_argument("--timed", action="store_true", help="Align TTS to segment timings")
    ap.add_argument("--reuse_tts", action="store_true", help="Reuse existing TTS chunks only when the cached semantic hash matches the current text, voice and engine settings. Useful for safe reassembly after fixing chunks.")
    ap.add_argument("--force_atempo", action="store_true", help="Force time-stretch to fit windows")
    ap.add_argument("--stretch_min", type=float, default=0.85)
    ap.add_argument("--stretch_max", type=float, default=1.30)
    ap.add_argument("--fill_slot", action="store_true", default=False,
                    help="Stretch TTS in both directions to exactly fill the original slot duration. "
                         "Slows down short TTS chunks, speeds up long ones. Capped at stretch_min/max.")
    ap.add_argument("--base_tempo", type=float, default=1.0, help="Base speed multiplier for all TTS segments (1.0 = no forced speedup)")
    ap.add_argument("--assembly_min_fill_tempo", type=float, default=0.90,
                    help="Minimálne tempo pri spomaľovaní príliš krátkych TTS segmentov. "
                         "0.70 = max 30%% spomalenie. 0.0 = vypnuté (len ticho). "
                         "Aktivuje sa keď ticho po TTS > 1.2s. Default: 0.70")
    ap.add_argument("--max_pause_gap_s", type=float, default=5.0, help="Maximum preserved pause between segments in timeline mode (seconds). Only used when --compress_timeline_gaps is enabled.")
    ap.add_argument("--compress_timeline_gaps", action="store_true", help="Allow timeline-mode pause compression before assembly. Disabled by default to preserve A/V sync.")
    ap.add_argument("--vad", action="store_true", help="Use Silero VAD for Whisper")
    ap.add_argument("--vad_threshold", type=float, default=0.5)
    ap.add_argument("--vad_min_speech_ms", type=int, default=3500, help="Min speech duration (ms) - higher = fewer segments")
    ap.add_argument("--vad_min_silence_ms", type=int, default=4500, help="Min silence to split (ms) - higher = fewer segments")
    ap.add_argument("--vad_speech_pad_ms", type=int, default=150)
    ap.add_argument("--vad_max_speech_s", type=float, default=90.0, help="Max segment duration (s)")
    ap.add_argument("--keep_music", action="store_true", help="Extract and keep background music from original (requires Demucs or MDX-Net)")
    ap.add_argument("--music_volume", type=float, default=0.25, help="Background music volume as ratio to speech RMS (0.15=subtle, 0.25=gentle, 0.50=prominent, max 1.0)")
    ap.add_argument("--vocal_separator", default="demucs",
                    choices=["demucs", "htdemucs_ft", "mdx_net"],
                    help="Vocal separation backend: demucs (default), htdemucs_ft (better quality), mdx_net (best, requires audio-separator)")
    ap.add_argument("--mdx_model", default="model_bs_roformer_ep_368_sdr_12.9628.ckpt",
                    help="MDX-Net model name (used when --vocal_separator=mdx_net)")
    ap.add_argument("--backaudio_volume", type=float, default=0.0, help="Mix original video audio as background at this volume (0.0=off, 0.2=subtle, 0.5=prominent). pyvideotrans-style: fills silence gaps with ambience.")
    ap.add_argument("--speech_gain", type=float, default=1.0, help="Manual speech gain multiplier (default: 1.0)")
    ap.add_argument("--pitch_shift", type=float, default=0.0, help="Pitch shift in semitones applied to final audio (negative = lower, e.g. -3)")
    auto_gain_group = ap.add_mutually_exclusive_group()
    auto_gain_group.add_argument(
        "--auto-speech-gain",
        dest="auto_speech_gain",
        action="store_true",
        help="Automatically scale speech to a usable level (default)",
    )
    auto_gain_group.add_argument(
        "--no-auto-speech-gain",
        dest="auto_speech_gain",
        action="store_false",
        help="Disable automatic speech normalization",
    )
    ap.set_defaults(auto_speech_gain=True)
    ap.add_argument("--sync_timing", action="store_true", help="Sync timing with original - each segment plays at exact original time")
    ap.add_argument("--timeline_mode", action="store_true", help="Use pyvideotrans concat-style assembly (gap-collapse + librosa stretch)")
    ap.add_argument("--legacy_assembly", action="store_true", help="Use old FFmpeg adelay/amix assembly instead of pyvideotrans concat style")
    ap.add_argument("--split_segments", action="store_true", help="Split long segments (>6s) into smaller chunks for better TTS")
    ap.add_argument("--max_segment_dur", type=float, default=6.0, help="Max segment duration before splitting (default: 6s)")
    ap.add_argument("--min_segment_dur", type=float, default=2.0, help="Min segment duration (default: 2s)")
    ap.add_argument("--auto_split_max_dur", type=float, default=10.0, help="Automaticky rozbiť segmenty dlhšie než táto hodnota (default: 10s). 0 = vypnuté.")
    ap.add_argument("--merge_segments", action="store_true", help="Merge adjacent short segments before translation")
    ap.add_argument("--merge_max_gap_s", type=float, default=0.0, help="Max gap (s) between segments to merge (0=disabled, set by content-type profile)")
    ap.add_argument("--merge_max_dur_s", type=float, default=0.0, help="Max merged segment duration (s) (0=disabled, set by content-type profile)")
    ap.add_argument("--sliding_window", action="store_true", help="Use sliding window context for translation (2 segments before/after)")
    ap.add_argument("--window_size", type=int, default=2, help="Number of segments before/after for context (default: 2)")
    ap.add_argument("--clone_voice", action="store_true", help="Clone voice from original video for better quality")
    ap.add_argument("--no_voice_clone", action="store_true", help="Disable auto voice cloning (no audio_prompt_path). Use model default voice — recommended for SK fine-tuned model to avoid artifacts from English reference audio.")
    ap.add_argument("--voice_sample_start", type=float, default=0, help="Start time (seconds) for voice sample extraction")
    ap.add_argument("--voice_sample_duration", type=float, default=10, help="Duration (seconds) of voice sample for cloning")
    ap.add_argument("--emotion_clone", action="store_true", help="Transfer emotion/style from the original video. With default/original voice it prefers a stable source-voice sample plus style transfer; with an explicit custom WAV it keeps that voice and only adapts speaking style/prosody.")
    ap.add_argument("--emotion_clone_pad_s", type=float, default=0.3, help="Padding (seconds) added around each segment when extracting emotion reference audio (default: 0.3)")
    ap.add_argument("--emotion_llm", action="store_true", help="LLM detekuje emocionálnu intenzitu segmentov a upraví Chatterbox exaggeration (low=0.50, medium=0.65, high=0.78)")
    ap.add_argument("--emotion_llm_model", default="gemma4:26b", help="Ollama model pre emotion detekciu (default: gemma4:26b)")
    ap.add_argument("--voice_eq", action="store_true", help="Apply EQ to balance voice frequencies (reduce bass/treble)")
    ap.add_argument("--tts_eq_profile", default="", help=f"EQ profile applied during polish (choices: {', '.join(TTS_EQ_PROFILES)})")
    ap.add_argument("--no_denoise", action="store_true", help="Skip FFT denoise post-processing")
    ap.add_argument("--denoise_preset", default="mild", choices=["mild", "strong"], help="Denoise strength preset")
    ap.add_argument("--use_piper", action="store_true", help="Use Piper TTS (faster, local)")
    ap.add_argument("--no_enhance", action="store_true", help="Skip Piper audio enhancement (keep raw 22kHz output)")
    ap.add_argument(
        "--piper_model", default=str(SCRIPT_DIR / "models" / "piper" / "sk_SK" / "lili" / "medium" / "sk_SK-lili-medium.onnx"), help="Path to Piper ONNX model"
    )
    ap.add_argument("--use_piper_rs", action="store_true", help="Use piper-rs CLI (pygoruut models)")
    ap.add_argument(
        "--piper_rs_cli", default="", help="Path to piper-rs CLI exe"
    )
    ap.add_argument(
        "--piper_rs_model",
        default="", help="Path to piper-rs ONNX model"
    )
    ap.add_argument(
        "--piper_rs_config",
        default="", help="Path to piper-rs config json"
    )
    ap.add_argument(
        "--piper_rs_ort_dll",
        default="", help="Path to onnxruntime.dll for piper-rs (1.23.x)"
    )
    ap.add_argument("--use_chatterbox", action="store_true", help="Use Chatterbox TTS (high quality Czech/Slovak)")
    ap.add_argument(
        "--chatterbox_model", default=str(SCRIPT_DIR / "models" / "chatterbox" / "t3_sk_v2.2.safetensors"), help="Path to Chatterbox MTL model weights"
    )
    ap.add_argument("--use_s2_pro", action="store_true", help="Use Fish Speech S2-Pro TTS (base checkpoint)")
    ap.add_argument(
        "--s2_pro_model",
        default=str(_DEFAULT_S2_PRO_MODEL),
        help="Path to the Fish Speech S2-Pro checkpoint directory",
    )
    ap.add_argument(
        "--s2_pro_server_python",
        default=str(Path.home() / "miniforge3" / "envs" / "fish_env" / "bin" / "python"),
        help="Path to fish_env python used to run the local Fish Speech API server",
    )
    ap.add_argument(
        "--s2_pro_server_port",
        type=int,
        default=8091,
        help="Local port for the Fish Speech S2-Pro API server",
    )
    ap.add_argument("--use_chatterbox_turbo", action="store_true", help="Use Chatterbox TURBO TTS (~3.4× faster than MTL)")
    ap.add_argument(
        "--chatterbox_turbo_model", default=str(SCRIPT_DIR / "models" / "chatterbox" / "t3_sk_turbo_v1_numbers_years.safetensors"), help="Path to Chatterbox Turbo fine-tuned weights"
    )
    ap.add_argument(
        "--turbo_reference_voice", default="", help="Slovak reference voice WAV for Turbo TTS (overrides built-in reference_sk.wav)"
    )
    ap.add_argument(
        "--turbo_auto_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turbo only: automatically route risky segments (numbers, abbreviations, short tech lines, SQL) to multilingual Chatterbox MTL. Default off keeps a single Turbo voice.",
    )
    ap.add_argument(
        "--turbo_fallback_model",
        default="",
        help="Turbo only: optional Chatterbox MTL model path for fallback segments. Empty = use --chatterbox_model.",
    )
    ap.add_argument(
        "--turbo_fallback_device",
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Turbo only: device for MTL fallback segments. Default cpu avoids dual-model VRAM spikes.",
    )
    ap.add_argument("--use_f5", action="store_true", help="Use F5-TTS SK fine-tuned (DiT flow-matching, ~30K stepov SK fine-tune)")
    ap.add_argument("--f5_model", default="/mnt/tts_data/miniforge3/envs/f5tts_env/lib/python3.11/ckpts/f5_sk_v4/model_415000.pt", help="Path to F5-TTS SK fine-tuned ckpt (default: v4 415K)")
    ap.add_argument("--f5_ref_audio", default="/mnt/tts_data/sk_tts/f5_sk_dataset/wavs/audiobook_Dominik_D_n___V_tieni_47_0001.wav", help="F5-TTS SK reference audio for voice cloning prompt (clean 3.3s, SNR 71dB)")
    ap.add_argument("--f5_ref_text", default="Nemám pištol, priznal sa Kanis, si normálny.", help="Transcript of f5_ref_audio (must match exactly)")
    ap.add_argument("--f5_python", default="/mnt/tts_data/miniforge3/envs/f5tts_env/bin/python", help="Path to f5tts_env python (subprocess call)")

    # OmniVoice TTS (k2-fsa, Apache 2.0, 600+ jazykov zero-shot)
    ap.add_argument("--use_omnivoice", action="store_true", help="Use OmniVoice TTS (zero-shot multilingual, najvyšší SK score na 34 testoch)")
    ap.add_argument("--omnivoice_repo", default="k2-fsa/OmniVoice", help="OmniVoice HF repo alebo local path")
    ap.add_argument("--omnivoice_ref_audio", default="/mnt/tts_data/VideoTranslator_studio/voices/juraj_sk_studio.wav", help="Reference audio (cleaned cez cleanup_reference_audio)")
    ap.add_argument("--omnivoice_ref_text", default="Náhrávka vznikla na základe predlohy publikovanej knižne vo vydavateľstve Slovart v roku dvetisíc dvadsaťpäť.", help="Transcript of ref_audio")
    ap.add_argument("--omnivoice_language", default="sk", help="OmniVoice language code (sk, en, cs, ...)")
    ap.add_argument("--omnivoice_num_step", type=int, default=64, help="Diffusion steps (32 default, 64 lepšia kvalita)")
    ap.add_argument("--omnivoice_guidance", type=float, default=2.5, help="Classifier-free guidance (default 2.0, 2.5 silnejšie ref tracking)")
    ap.add_argument("--omnivoice_python", default="/mnt/tts_data/miniforge3/envs/f5tts_env/bin/python", help="Path to env with omnivoice package")
    ap.add_argument("--omnivoice_instruct", default="", help="Voice design tags (napr. 'male, young adult, moderate pitch'). Validné: male/female, child/teenager/young adult/middle-aged/elderly, very low/low/moderate/high/very high pitch, whisper, *_accent")
    ap.add_argument("--omnivoice_speed", type=float, default=1.0, help="Speech speed multiplier (1.0=normal, 1.12=fast, 0.9=slow)")
    ap.add_argument("--omnivoice_expressive", action="store_true", help="Pridaj čiarky pred 'teda/napríklad/ale' pre expresívnejšiu prozódiu")

    ap.add_argument("--use_xtts", action="store_true", help="Use XTTS v2 TTS (multilingual, no artifacts)")
    ap.add_argument("--xtts_lang", default="cs", help="XTTS language code (cs, sk, en...)")
    ap.add_argument("--xtts_speaker_wav", default="", help="Reference voice WAV for XTTS voice cloning")
    ap.add_argument("--xtts_speaker", default="Damien Black", help="Built-in XTTS speaker name (used when no speaker_wav)")
    ap.add_argument("--xtts_pitch_shift", type=float, default=0.0, help="XTTS pitch shift in semitones (e.g. -3 = lower)")
    ap.add_argument("--xtts_python", default="", help="Path to xtts_env python (auto-detected if empty)")
    ap.add_argument("--pydub_join", action="store_true", help="Use pydub trim+crossfade for buffered Chatterbox joins")
    ap.add_argument("--no_torch_compile", action="store_true", help="Disable torch.compile() optimization for Chatterbox")
    ap.add_argument("--use_lmstudio_translate", action="store_true", help="Use LM Studio OpenAI-compatible API for translation (port 1234)")
    ap.add_argument("--lmstudio_translate_url", default="http://localhost:1234/v1", help="LM Studio API base URL (default: http://localhost:1234/v1)")
    ap.add_argument("--lmstudio_translate_model", default="", help="LM Studio model name for API calls (empty = use loaded model)")
    ap.add_argument("--lmstudio_model_key", default="gemma-4-26b-a4b-it", help="LM Studio model key for auto-load via lms CLI (default: gemma-4-26b-a4b-it)")
    ap.add_argument("--lmstudio_translate_timeout", type=int, default=180, help="Timeout per batch for LM Studio translation (default: 180s)")
    ap.add_argument("--ls_batch_size", type=int, default=50, help="Segmenty na 1 LM Studio dávku (default 50, vyššie = viac kontextu ale väčší prompt)")
    ap.add_argument("--whole_doc_translate", action="store_true", help="Posielaj celý dokument v 1 dávke pre max kontextu (overrides --ls_batch_size)")
    ap.add_argument("--chatterbox_s2_lastresort", action="store_true", help="Hybrid: Chatterbox primary + S2 Pro ako last-resort fallback ak Chatterbox QA zlyhá (namiesto silence)")
    ap.add_argument("--use_translategemma", action="store_true", help="Use TranslateGemma as translation engine (dedicated translation model, EN→55 languages)")
    ap.add_argument("--use_translategemma_fulldoc", action="store_true", help="TranslateGemma full-document mode: translate all segments at once with §N§ markers for context consistency")
    ap.add_argument("--sw_batch_size", type=int, default=40, help="Sliding window: počet segmentov v jednej dávke (default 40)")
    ap.add_argument("--sw_memory_size", type=int, default=15, help="Sliding window: počet preložených segmentov ako kontext/pamäť (default 15)")
    ap.add_argument("--translategemma_model", default="", help="Path to TranslateGemma GGUF model file")
    ap.add_argument("--translategemma_gpu_layers", type=int, default=-1, help="GPU layers for TranslateGemma (-1 = all)")
    ap.add_argument("--refine_translation", action="store_true", help="Second-pass: refine translated text with LLM for natural grammar/declension (Slovak/Czech)")
    ap.add_argument("--refine_model", default="", help="LLM model path for refinement pass (default: same as --llama_model)")
    ap.add_argument("--sk_corrector_model", default="sk-corrector-v2", help="Ollama model name for SK grammar corrector pass after LM Studio translation (default: sk-corrector-v2, empty string = vypnuté)")
    ap.add_argument("--use_finetuned_translator", action="store_true", help="Use fine-tuned 26b-translator via ollama (raw API) instead of LM Studio base 26B")
    ap.add_argument("--finetuned_translator_model", default="26b-translator-q6", help="Ollama model name for fine-tuned translator (default: 26b-translator-q6 — Q6_K, najvyššia kvalita; alternatívy: 26b-translator-q5l, 26b-translator-q5, 26b-translator [Q4_K_M])")
    # G3 12B v1 — Gemma 3 12B IT QLoRA fine-tuned EN→SK translator + length adjuster (2026-04-26)
    ap.add_argument("--use_g3_v1_translator", action="store_true", help="Use Gemma 3 12B v1 translator (g3-12b-translator-v1 ollama) — namiesto 26B. Rýchlejší (~0.5s/segment), lepšia kvalita, žiadny HTML leak.")
    ap.add_argument("--g3_v1_translator_model", default="g3-12b-translator-v1", help="Ollama model name pre G3 v1 translator (default: g3-12b-translator-v1)")
    # G3 27B v2 — single-step multitask (translate + length-aware) — nahrádza 12B + length adjuster
    # v2 oproti v1: negation transfer fix ("I knew zero X" → "Nevedel som o X vôbec nič")
    ap.add_argument("--use_g3_27b_translator", action="store_true", help="Use Gemma 3 27B multitask v2 (g3-27b-multitask-v2 ollama) — single-step EN→SK + length-aware. Lepšia kvalita než 12B, ~1s/segment.")
    ap.add_argument("--g3_27b_translator_model", default="g3-27b-multitask-v2", help="Ollama model name pre G3 27B (default: g3-27b-multitask-v2)")
    ap.add_argument("--use_g3_v1_length_adjuster", action="store_true", help="Použiť Gemma 3 12B length adjuster v1 (g3-12b-length-adjuster-v1) pre compress + expand. Bez expandu pri kratších prekladoch by audio malo ticho.")
    ap.add_argument("--g3_v1_length_adjuster_model", default="g3-12b-length-adjuster-v1", help="Ollama model name pre G3 v1 length adjuster (default: g3-12b-length-adjuster-v1)")
    ap.add_argument("--compress_overflow", action="store_true", default=True, help="Skráť segmenty ktoré prekračujú CPS limit (D=deterministic + C=26B fallback). Default zapnuté.")
    ap.add_argument("--no_compress_overflow", dest="compress_overflow", action="store_false", help="Vypne kompresiu overflow segmentov.")
    ap.add_argument("--compress_target_cps", type=float, default=13.0, help="Cieľový CPS (chars per second) pre kompresiu. Vyššie = dlhšie titulky, tolerantnejšie (default: 13.0)")
    ap.add_argument("--refine_gpu_layers", type=int, default=-1, help="GPU layers for refinement model (-1 = all on GPU, default: -1)")
    ap.add_argument("--refine_context", default="", help="Context hint for refinement pass (e.g. speaker gender, topic)")
    ap.add_argument("--gender_fix", default="", help="Deterministická oprava rodu po preklade. Hodnoty: 'sk_f' (slovenčina, ženský rod) — opraví 'som videl'→'som videla', 'bol som'→'bola som' atď.")
    ap.add_argument("--tts_safe_mode", action="store_true", help="Chatterbox safe mode: lower temperature (0.68) + lower exaggeration (0.55) — fewer early-EOS artifacts, more stable output")
    ap.add_argument("--tts_ultra_safe", action="store_true", help="Chatterbox ultra-safe mode: very low temperature (0.45) — maximum stability, less expressive")
    ap.add_argument(
        "--tts_timing_retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After TTS render, retry one too-long segment with a shorter text candidate. Default: True.",
    )
    ap.add_argument(
        "--tts_timing_retry_max_attempts",
        type=int,
        default=1,
        help="Maximum number of post-TTS timing retries per segment. Default: 1.",
    )
    ap.add_argument("--chatterbox_exaggeration", type=float, default=-1.0, help="Chatterbox exaggeration (0.0–1.0, default model default=0.5, pipeline default=0.75). Nižšia = prirodzenejší hlas, menej artefaktov.")
    ap.add_argument("--chatterbox_cfg_weight", type=float, default=-1.0, help="Chatterbox cfg_weight (0.0–1.0, default=0.5). Vyššia = silnejšia voice guidance.")
    ap.add_argument("--chatterbox_temperature", type=float, default=-1.0, help="Chatterbox temperature (default=0.60). Nižšia = stabilnejší výstup.")
    ap.add_argument("--use_chatterbox_onnx", action="store_true", help="Use Chatterbox ONNX TTS (faster, English)")
    ap.add_argument(
        "--chatterbox_onnx_precision",
        default="fp32",
        choices=["fp32", "fp16", "q4", "q4f16"],
        help="ONNX model precision (fp32 safest, q4 fastest)",
    )
    ap.add_argument("--onnx_reference_voice", default="", help="Path to reference voice sample for ONNX (overrides auto extraction)")
    ap.add_argument("--use_openai_tts", action="store_true", help="Use OpenAI ChatGPT TTS API")
    ap.add_argument("--openai_api_key", default="", help="OpenAI API key (optional, can use OPENAI_API_KEY env)")
    ap.add_argument("--openai_tts_model", default="gpt-4o-mini-tts", help="OpenAI TTS model")
    ap.add_argument("--openai_tts_voice", default="alloy", help="OpenAI TTS voice")
    ap.add_argument("--use_api_tts", action="store_true", help="Use OpenAI-compatible API TTS (ChatGPT/Grok/Gemini)")
    ap.add_argument("--api_tts_provider", default="chatgpt", help="API TTS provider label (chatgpt|grok|gemini)")
    ap.add_argument("--api_tts_key", default="", help="API TTS key (optional, can use API_TTS_KEY env)")
    ap.add_argument("--api_tts_base_url", default="https://api.openai.com/v1", help="API TTS base URL")
    ap.add_argument("--api_tts_model", default="gpt-4o-mini-tts", help="API TTS model")
    ap.add_argument("--api_tts_voice", default="alloy", help="API TTS voice")
    ap.add_argument("--use_api_translate", action="store_true", help="Use OpenAI-compatible API LLM for translation (ChatGPT/Grok/Gemini)")
    ap.add_argument("--api_translate_provider", default="chatgpt", help="API translate provider label (chatgpt|grok|gemini)")
    ap.add_argument("--api_translate_key", default="", help="API translate key (optional, can use API_TRANS_KEY env)")
    ap.add_argument("--api_translate_base_url", default="https://api.openai.com/v1", help="API translate base URL")
    ap.add_argument("--api_translate_model", default="gpt-5-mini-2025-08-07", help="API translate model")
    ap.add_argument(
        "--multi_voice",
        action="store_true",
        help="Enable multi-voice TTS: cluster segments by speaker using MFCC + KMeans and assign a dedicated voice reference per speaker cluster",
    )
    ap.add_argument("--multi_voice_n", type=int, default=2, help="Number of speaker voices to detect for --multi_voice (default: 2)")
    ap.add_argument("--diarization_json", default="", help="Path to pre-computed Level 0b JSON (speaker_id, speaker_ref_audio, emotion). If set, skips re-clustering.")
    ap.add_argument(
        "--multi_voice_male_wav",
        default="",
        help="Optional male fallback WAV for multi-voice Chatterbox routing. Empty = use cloned cluster reference.",
    )
    ap.add_argument(
        "--multi_voice_female_wav",
        default="",
        help="Optional female fallback WAV for multi-voice Chatterbox routing. Empty = use cloned cluster reference.",
    )
    ap.add_argument("--pass1_prompt", default="", help="Custom pass-1 translation prompt (overrides content_type profile; auto-set by profile if empty)")
    ap.add_argument("--pass2_prompt", default="", help="Custom pass-2 polish prompt (overrides content_type profile; auto-set by profile if empty)")
    # NLLB → Gemma → QA pipeline
    ap.add_argument("--use_nllb_gemma", action="store_true", help="Use NLLB-200 → Gemma cleanup → QA 3-stage pipeline (no refusals, preserves tone)")
    ap.add_argument("--nllb_model", default="facebook/nllb-200-1.3B", help="NLLB-200 model name or path (1.3B recommended, 3.3B for better quality)")
    ap.add_argument("--nllb_src_lang", default="auto", help="NLLB source language (auto=use Whisper detected lang, or explicit NLLB code e.g. eng_Latn)")
    ap.add_argument("--nllb_device", default="cpu", choices=["cpu", "cuda"], help="Device for NLLB-200 (default: cpu to leave VRAM for Gemma)")
    ap.add_argument("--nllb_qa_model_path", default="", help="Path to QA Llama GGUF model for NLLB pipeline. Empty = skip QA stage.")
    ap.add_argument("--nllb_qa_gpu_layers", type=int, default=32, help="GPU layers for QA Llama model in NLLB pipeline (default: 32)")
    # Generic QA stage — works with any translation pipeline
    ap.add_argument("--use_qa", action="store_true", help="Run QA tone-check (Llama 8B) after any translation to verify explicitness was not softened")
    ap.add_argument("--qa_model_path", default="", help="Path to QA Llama GGUF model (Llama 3.1 8B recommended). Required when --use_qa is set.")
    ap.add_argument("--qa_gpu_layers", type=int, default=32, help="GPU layers for QA Llama model (default: 32)")
    ap.add_argument("--use_text_adaptation", action=argparse.BooleanOptionalAction, default=True,
                    help="Text adaptation layer: timing-aware rewrite if TTS estimate overflows slot. "
                         "Uses --llama_model for concise/dub-friendly rewrite (2 rounds). "
                         "Fallback to stretch if no LLM or rewrite fails. "
                         "Default: True. Use --no-use_text_adaptation to disable.")
    ap.add_argument(
        "--use_mini_level11",
        action="store_true",
        help="Run mini_level11 pre-TTS rewrite pass (EN source -> SK spoken rewrite + SQL logic guard) before text adaptation.",
    )
    ap.add_argument(
        "--use_level6_plus",
        action="store_true",
        help="Run Level 12/13 timing+rewrite stack before TTS, Level 5-11 orchestration before TTS, and post-TTS analysis JSONs.",
    )
    ap.add_argument(
        "--level_memory_file",
        default=str(SCRIPT_DIR / "memory" / "project_memory.json"),
        help="Path to persistent Level 7/8/9/11 project memory JSON.",
    )
    ap.add_argument(
        "--level_policy_registry_file",
        default=str(SCRIPT_DIR / "memory" / "policy_registry.json"),
        help="Path to Level 10/11 policy registry JSON.",
    )
    ap.add_argument(
        "--level11_max_retries",
        type=int,
        default=1,
        help="Maximum autonomous retry rounds per segment in Level 11 sidecar.",
    )
    ap.add_argument(
        "--use_ollama_adaptation",
        action="store_true",
        help="Use local Ollama HTTP API for text adaptation instead of llama.cpp GGUF.",
    )
    ap.add_argument(
        "--adapt_provider",
        default="llama_cpp",
        choices=["llama_cpp", "ollama"],
        help="Configured text adaptation provider. Used by mini_level11 as well.",
    )
    ap.add_argument(
        "--adapt_ollama_url",
        default="http://localhost:11434/api/generate",
        help="Ollama generate endpoint for text adaptation.",
    )
    ap.add_argument(
        "--adapt_ollama_model",
        default="sk-gemma4-e4b-v6",
        help="Ollama model used for timing-aware text adaptation.",
    )
    ap.add_argument(
        "--adapt_ollama_timeout",
        type=int,
        default=120,
        help="Timeout in seconds for one Ollama adaptation request.",
    )
    ap.add_argument(
        "--cps_expand_threshold",
        type=float,
        default=9.0,
        help="Pre-TTS CPS expand pass: segmenty s CPS < prah dostanú ollama expand (default 9.0, 0=vypnuté).",
    )
    ap.add_argument(
        "--cps_expand_min_slot",
        type=float,
        default=2.0,
        help="Pre-TTS CPS expand: minimálna dĺžka slotu v sekundách pre expanziu (default 2.0).",
    )
    ap.add_argument(
        "--no_cps_expand",
        action="store_true",
        help="Vypne pre-TTS CPS expansion pass.",
    )
    ap.add_argument(
        "--cps_repair_model",
        default="llama3.1:8b",
        help="Ollama model pre CPS repair po gemma4 preklade (default llama3.1:8b, CPU-friendly).",
    )
    # MADLAD → Gemma polish → QA  batch pipeline (memory-efficient: each model fully unloaded before next)
    ap.add_argument(
        "--use_madlad_gemma_qa",
        action="store_true",
        help=(
            "3-phase batch pipeline: "
            "Phase 1 MADLAD translates all segments → free; "
            "Phase 2 Gemma polishes all → free; "
            "Phase 3 QA Llama checks all → free. "
            "Uses --madlad_model, --llama_model/--llama_gpu_layers/--llama_ctx, --qa_model_path/--qa_gpu_layers."
        ),
    )
    return ap
