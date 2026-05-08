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
| 🔍 **Auto-detection** | Pitch analýza + Whisper sample + keyword regex → predvyplní hlas, voice design, typ obsahu, diffusion steps a guidance scale |
| 🔧 **Nástroje** | Audio mastering, EQ presety, správa API kľúčov, checkpoint manager |

---

## 🛠 Pipeline

1. **STT** — Faster-Whisper large-v3-turbo s context-aware transcription
2. **Preklad** — Gemma 4 26B fine-tuned (lokálne) / GPT-5 / Grok / Gemini (API), s gender-aware prompt-om pre konzistentný rod cez celé video
3. **Adaptácia** — slot-aware rewrite, adaptívna CPS kalibrácia
4. **TTS** — Chatterbox SK 2.2 (ref-generator) → OmniVoice (zero-shot, primary)
5. **Assembly** — preserve gaps, sync-safe timeline, per-segment atempo, center-short placement, ffmpeg mux

### 🎯 Voice cloning architecture v2 (2026-05-08)

Pre dabovanie videí kde chceš zachovať identitu pôvodného speakera + natívny SK akcent:

1. **Demucs** vyseparuje vokály z videa → `video_clone.wav` (americký prizvuk)
2. **Chatterbox SK 2.2** (PRO config) vyrobí SK-flavored ref — SK akcent zapečený v modeli + video timbre cez audio_prompt
3. **OmniVoice** klonuje z tejto ref → finálne TTS audio (SK akcent + identita pôvodného speakera + OmniVoice quality)
4. **PRO Master** (open clarity EQ): bass body + de-nasal cut + presence + sparkle highs

Auto-fallback: ak Chatterbox model chýba, OmniVoice použije raw video clone (americký prizvuk, ale pipeline funguje).

### 🔍 Smart Auto-detection (jeden klik)

Pred Spustiť stačí kliknúť tlačidlo *Detekuj hovoriaceho* v TTS Engine sekcii — analyzuje zdroj a predvyplní GUI:

- **Pitch analýza** (librosa pyin na demucs vokáloch) → median F0, multispeaker, voice design preset (🇸🇰 muž / žena / multi-voice)
- **Whisper sample 90 s** + keyword regex (EN + SK) → typ obsahu (general / programming / technical / sql / educational / podcast / review / news)
- **Decision tree** pre OmniVoice params: tech content → 96 steps + 2.8 guidance, multispeaker → 96 + 3.0
- **Speaker gender** sa propaguje cez celý preklad: do prompt-u, do refine pass-u (G3-12B), plus deterministický regex post-fix („som dokončil" → „som dokončila") pre konzistenciu cez celé video

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
