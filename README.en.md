# VideoTranslator Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

🌐 **Languages:** [🇸🇰 Slovak](README.md) · 🇬🇧 English

Local-first **EN → SK** (and other languages) video-dubbing pipeline. STT → translation → TTS → finalization. Fully local, no cloud.

🌐 **Web:** https://vojtech-pet.github.io/videotranslator-studio

![VideoTranslator Studio v2.0 GUI](docs/screenshot.png)

---

## ⚡ Quick install

### 🐧 Linux

```bash
bash <(curl -sL https://raw.githubusercontent.com/Vojtech-Pet/videotranslator-studio/main/install.sh)
```

After install: applications menu → **VideoTranslator Studio**, or:
```bash
~/.local/share/videotranslator-studio/launch_studio.sh
```

### 🪟 Windows (PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/Vojtech-Pet/videotranslator-studio/main/install.ps1 | iex
```

Prerequisites (if missing):
- **Git for Windows** — https://git-scm.com/download/win
- **ffmpeg** — `winget install ffmpeg` (or from https://www.gyan.dev/ffmpeg/builds/)
- **NVIDIA driver** — for GPU acceleration (Game Ready or Studio driver)

After install: **Start menu → VideoTranslator Studio**.

### 📦 Portable (no install — just unpack and run)

For users without conda / pip / internet:

1. Download from [Releases](https://github.com/Vojtech-Pet/videotranslator-studio/releases) **2 files:**
   - `videotranslator-source-vX.Y.Z.tar.gz` (~3 MB)
   - `videotranslator-env-linux-x64.tar.gz` (~2 GB) ← contains Python + PyQt6 + all deps
2. Unpack into any folder:
   ```bash
   mkdir -p ~/VideoTranslatorStudio
   cd ~/VideoTranslatorStudio
   tar xzf ~/Downloads/videotranslator-source-vX.Y.Z.tar.gz --strip 1
   mkdir env && tar xzf ~/Downloads/videotranslator-env-linux-x64.tar.gz -C env/
   ```
3. Run:
   ```bash
   ./launch_studio.sh
   ```

First launch performs `conda-unpack` (~10–30 s, fixes relocation paths). Subsequent launches are instant.

Works **without system Python, without Qt6 install, without root**. Whole project removable via `rm -rf ~/VideoTranslatorStudio`.

### 🐧 Windows via WSL2 (alternative)

If you prefer Linux env:
1. `wsl --install` in PowerShell
2. In Ubuntu/WSL run the same `install.sh` as for Linux
3. GUI works through WSLg (Windows 11) or X-Server (Windows 10)

---

### What the install script does

- Downloads **Miniforge3** (if missing)
- Creates conda environment `videotranslator`
- Installs Python deps + CUDA libs (if GPU)
- Clones the repo
- Adds **menu shortcut**

---

## 🎬 Features

| Module | Description |
|---|---|
| 📁 **Local file** | 4-step wizard (STT → Translation → TTS → Finalize), batch mode, checkpoints |
| ⬇️ **Video download** | YouTube, Vimeo, TikTok, X/Twitter, Facebook + 1000+ more via yt-dlp |
| 🎙️ **Narrator** | Text-to-speech with reference voice, style-instruct (gender, age, pitch, accent) |
| 🔧 **Tools** | Audio mastering, EQ presets, API key management, checkpoint manager |

---

## 🛠 Pipeline

1. **STT** — Faster-Whisper large-v3-turbo with context-aware transcription
2. **Translation** — Gemma 4 26B fine-tuned (local) / GPT-5 / Grok / Gemini (API)
3. **Adaptation** — slot-aware rewrite, adaptive CPS calibration
4. **TTS** — OmniVoice (zero-shot, primary) or Chatterbox SK 2.2 (backup)
5. **Assembly** — preserve gaps, sync-safe timeline, ffmpeg mux

---

## 💻 Hardware

- **GPU:** NVIDIA with 24 GB VRAM recommended (RTX 4090 / 3090 / A6000). Works on 12 GB (smaller models).
- **RAM:** 32+ GB
- **Disk:** ~80 GB for base models (optional — install.sh asks what to download)

CPU-only fallback exists but is significantly slower.

---

## 🔑 API keys (optional)

VTS works **fully local** with fine-tuned Gemma 4 26B + OmniVoice. To use cloud APIs as alternative, set env vars:

```bash
export OPENAI_API_KEY=sk-...
export GROK_API_KEY=xai-...
export GEMINI_API_KEY=...
export HF_API_KEY=hf_...   # for HF gated models
```

Copy `.env.example` to `.env` and fill in. (`.env` is in `.gitignore`.)

---

## 🗂 Paths (auto-detect)

VTS auto-detects paths on first launch. Override via `~/.config/videotranslator/paths.json`.

Default structure:
```
~/miniforge3                              # conda
~/.local/share/videotranslator-studio/    # source code
~/.config/videotranslator/paths.json      # paths config
$models_root/                             # auto-detect: /mnt/tts_data/.../models, ~/Ai/models, etc.
```

---

## 📚 Documentation

- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Bug map:** [docs/BUG_MAP.md](docs/BUG_MAP.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **CLI flags:** `python main.py --help`

---

## 🤝 Contributing

Issues and PRs welcome. Before opening a PR please run:
```bash
python -m pytest tests/   # if present
python -c "import ast; ast.parse(open('VideoTranslator_studio.py').read())"   # syntax check
```

---

## 📄 License

[MIT License](LICENSE) — © 2026 Vojtech Petrik

---

## 🔗 Links

- **Web:** https://vojtech-pet.github.io/videotranslator-studio
- **Issues:** https://github.com/Vojtech-Pet/videotranslator-studio/issues
- **Email:** pekiskol@gmail.com
