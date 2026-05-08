# VideoTranslator Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Automatický **EN→SK** (a iné jazyky) video-dubbing pipeline. STT → preklad → TTS → finalizácia. Plne lokálne, bez cloudu.

🌐 **Web:** https://vojtech-pet.github.io/videotranslator-studio

![VideoTranslator Studio v2.0 GUI](docs/screenshot.png)

---

## ⚡ Rýchla inštalácia

### 🐧 Linux

```bash
bash <(curl -sL https://raw.githubusercontent.com/Vojtech-Pet/videotranslator-studio/main/install.sh)
```

Po inštalácii: aplikačné menu → **VideoTranslator Studio**, alebo:
```bash
~/.local/share/videotranslator-studio/launch_studio.sh
```

### 🪟 Windows (PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/Vojtech-Pet/videotranslator-studio/main/install.ps1 | iex
```

Pred-prerekvizity (ak chýbajú):
- **Git for Windows** — https://git-scm.com/download/win
- **ffmpeg** — `winget install ffmpeg` (alebo z https://www.gyan.dev/ffmpeg/builds/)
- **NVIDIA driver** — pre GPU akceleráciu (Game Ready alebo Studio driver)

Po inštalácii: **Štart menu → VideoTranslator Studio**.

### 📦 Portable (žiadny install — len rozbaľ a spusti)

Pre používateľov bez conda / pip / internet pripojenia:

1. Stiahni z [Releases](https://github.com/Vojtech-Pet/videotranslator-studio/releases) **2 súbory:**
   - `videotranslator-source-vX.Y.Z.tar.gz` (~3 MB)
   - `videotranslator-env-linux-x64.tar.gz` (~2 GB) ← obsahuje Python + PyQt6 + všetky deps
2. Rozbaľ do ľubovoľného foldera:
   ```bash
   mkdir -p ~/VideoTranslatorStudio
   cd ~/VideoTranslatorStudio
   tar xzf ~/Downloads/videotranslator-source-vX.Y.Z.tar.gz --strip 1
   mkdir env && tar xzf ~/Downloads/videotranslator-env-linux-x64.tar.gz -C env/
   ```
3. Spusti:
   ```bash
   ./launch_studio.sh
   ```

Prvé spustenie urobí `conda-unpack` (~10-30 s, opraví relocation paths). Ďalšie spustenia sú instant.

Funguje **bez systémového Pythonu, bez Qt6 inštalácie, bez root práv**. Celý projekt sa dá zmazať príkazom `rm -rf ~/VideoTranslatorStudio`.

### 🐧 Windows cez WSL2 (alternatíva)

Ak preferuješ Linux prostredie:
1. `wsl --install` v PowerShell
2. V Ubuntu/WSL spusti rovnaký `install.sh` ako pre Linux
3. GUI funguje cez WSLg (Windows 11) alebo X-Server (Windows 10)

---

### Čo install skript spraví

- Stiahne **Miniforge3** (ak chýba)
- Vytvorí conda environment `videotranslator`
- Nainštaluje Python deps + CUDA libs (ak je GPU)
- Klonuje repo
- Pridá **shortcut do menu**

---

## 🎬 Funkcie

| Modul | Popis |
|---|---|
| 📁 **Lokálny súbor** | 4-krokový wizard (STT → Preklad → TTS → Finalizácia), batch režim, checkpointy |
| ⬇️ **Stiahnutie videa** | YouTube, Vimeo, TikTok, X/Twitter, Facebook + 1000+ ďalších cez yt-dlp |
| 🎙️ **Narrátor** | Text-to-speech s referenčným hlasom, štýl-instruct (gender, vek, pitch, accent) |
| 🔧 **Nástroje** | Audio mastering, EQ presety, správa API kľúčov, checkpoint manager |

---

## 🛠 Pipeline

1. **STT** — Faster-Whisper large-v3-turbo s context-aware transcription
2. **Preklad** — Gemma 4 26B fine-tuned (lokálne) / GPT-5 / Grok / Gemini (API)
3. **Adaptácia** — slot-aware rewrite, adaptívna CPS kalibrácia
4. **TTS** — OmniVoice (zero-shot, primary) alebo Chatterbox SK 2.2 (backup)
5. **Assembly** — preserve gaps, sync-safe timeline, ffmpeg mux

---

## 💻 Hardware

- **GPU:** NVIDIA s 24 GB VRAM odporúčané (RTX 4090 / 3090 / A6000). Funguje aj s 12 GB (menšie modely).
- **RAM:** 32+ GB
- **Disk:** ~80 GB pre základné modely (voliteľne — install.sh sa spýta čo stiahnuť)

CPU-only fallback existuje, ale je výrazne pomalší.

---

## 🔑 API kľúče (voliteľné)

VTS funguje **plne lokálne** s fine-tuned Gemma 4 26B + OmniVoice. Ak chceš použiť cloud API ako alternatívu, nastav env premenné:

```bash
export OPENAI_API_KEY=sk-...
export GROK_API_KEY=xai-...
export GEMINI_API_KEY=...
export HF_API_KEY=hf_...   # pre HF gated modely
```

Skopíruj `.env.example` na `.env` a vyplň. (`.env` je v `.gitignore`.)

---

## 🗂 Cesty (auto-detect)

VTS auto-detekuje cesty pri prvom spustení. Override cez `~/.config/videotranslator/paths.json`.

Štruktúra (default):
```
~/miniforge3                              # conda
~/.local/share/videotranslator-studio/    # source code
~/.config/videotranslator/paths.json      # cesty config
$models_root/                             # auto-detect: /mnt/tts_data/.../models, ~/Ai/models, atď.
```

---

## 📚 Dokumentácia

- **Architektúra:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Bug map:** [docs/BUG_MAP.md](docs/BUG_MAP.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **CLI flagy:** `python main.py --help`

---

## 🤝 Prispievanie

Issues a PR sú vítané. Pred PR prosím spusti:
```bash
python -m pytest tests/   # ak existujú
python -c "import ast; ast.parse(open('VideoTranslator_studio.py').read())"   # syntax check
```

---

## 📄 Licencia

[MIT License](LICENSE) — © 2026 Vojtech Petrik

---

## 🔗 Linky

- **Web:** https://vojtech-pet.github.io/videotranslator-studio
- **Issues:** https://github.com/Vojtech-Pet/videotranslator-studio/issues
- **Email:** pekiskol@gmail.com
