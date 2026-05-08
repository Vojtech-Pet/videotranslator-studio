"""
VideoTranslator Studio — Qt6 · Full-featured port
Dark purple-gray · Green accent · Sidebar nav · All VideoTranslator features
"""
from __future__ import annotations
import sys, json, re, os, subprocess, time, math, shutil
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QComboBox, QLineEdit, QTextEdit, QPlainTextEdit, QSlider, QCheckBox, QRadioButton,
    QHBoxLayout, QVBoxLayout, QGridLayout, QStackedWidget, QButtonGroup,
    QFileDialog, QScrollArea, QListWidget, QListWidgetItem, QSplitter,
    QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget, QInputDialog,
    QMessageBox, QSizePolicy, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QRect, QFileSystemWatcher
from PyQt6.QtGui import QFont, QColor, QPainter, QIcon, QPixmap, QPen, QBrush, QPainterPath, QTextCursor, QAction

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
SCRIPT_DIR = BASE_DIR
sys.path.insert(0, str(BASE_DIR))
from scripts.paths import PATHS  # auto-detect or load from ~/.config/videotranslator/paths.json
VOICES_DIR             = BASE_DIR / "voices"
VOICES_DIR.mkdir(exist_ok=True)
CONFIG_FILE            = BASE_DIR / "musetalk_gui_config.json"
PROFILES_DIR           = BASE_DIR / "profiles"
NORMAL_DEFAULT_PROFILE = "normal_default"
CHECKPOINT_FILE        = BASE_DIR / "batch_checkpoint.json"
VIDEO_LISTS_FILE       = BASE_DIR / "video_lists.json"
SAVED_CHECKPOINTS_FILE = BASE_DIR / "saved_checkpoints.json"
LANG_DIR               = BASE_DIR / "languages"

# ── i18n ───────────────────────────────────────────────────────────────────────
def _available_languages() -> Dict[str, str]:
    """Returns {code: name} for all JSON files in languages/."""
    result: Dict[str, str] = {}
    if LANG_DIR.exists():
        for f in sorted(LANG_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                meta = data.get("_meta", {})
                code = meta.get("code", f.stem)
                name = meta.get("name", f.stem)
                result[code] = name
            except Exception:
                pass
    return result

def _load_language(code: str) -> Dict[str, str]:
    path = LANG_DIR / f"{code}.json"
    if not path.exists():
        path = LANG_DIR / "sk.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        return {}

# Loaded at startup; pages read from this dict
T: Dict[str, str] = {}

# Voice picker sentinels — stored canonically as Slovak in settings
# but displayed translated in comboboxes.
_VOICE_SENTINELS = {
    "(predvolený)":      "voice_default",
    "(žiadny)":          "voice_none",
    "(originál video)":  "voice_original_video",
    "🎬 Cloning z videa": "voice_clone_from_video",
}

def _voice_disp(canonical: str) -> str:
    """Convert canonical voice value to translated display text."""
    key = _VOICE_SENTINELS.get(canonical)
    return T.get(key, canonical) if key else canonical

def _voice_canon(display: str) -> str:
    """Convert displayed voice text back to canonical SK form."""
    for canonical, key in _VOICE_SENTINELS.items():
        if display == T.get(key, canonical):
            return canonical
    return display


# OmniVoice voice design — per-language presety mapované na EN tagy podporované modelom.
# OmniVoice nemá explicit slovak/czech accent → použijeme russian (najbližší slavský).
_VOICE_DESIGN_MAP = {
    # SK presets
    "🇸🇰 slovenský muž":         "male, russian accent",
    "🇸🇰 slovenská žena":        "female, russian accent",
    "🇸🇰 muž, mladý dospelý":    "male, young adult, moderate pitch, russian accent",
    "🇸🇰 muž, stredný vek":      "male, middle-aged, moderate pitch, russian accent",
    "🇸🇰 muž, starší":           "male, elderly, low pitch, russian accent",
    "🇸🇰 žena, mladá dospelá":   "female, young adult, moderate pitch, russian accent",
    "🇸🇰 žena, stredný vek":     "female, middle-aged, moderate pitch, russian accent",
    "🇸🇰 žena, staršia":         "female, elderly, russian accent",
    "🇸🇰 muž, hlboký hlas":      "male, low pitch, russian accent",
    "🇸🇰 žena, vysoký hlas":     "female, high pitch, russian accent",
    "🇸🇰 šepkanie":              "whisper, russian accent",
    # CZ presets
    "🇨🇿 český muž":             "male, russian accent",
    "🇨🇿 česká žena":            "female, russian accent",
    "🇨🇿 muž, mladý dospělý":    "male, young adult, moderate pitch, russian accent",
    "🇨🇿 muž, střední věk":      "male, middle-aged, moderate pitch, russian accent",
    "🇨🇿 muž, starší":           "male, elderly, low pitch, russian accent",
    "🇨🇿 žena, mladá dospělá":   "female, young adult, moderate pitch, russian accent",
    "🇨🇿 žena, střední věk":     "female, middle-aged, moderate pitch, russian accent",
    "🇨🇿 žena, starší":          "female, elderly, russian accent",
    "🇨🇿 muž, hluboký hlas":     "male, low pitch, russian accent",
    "🇨🇿 žena, vysoký hlas":     "female, high pitch, russian accent",
    "🇨🇿 šepot":                 "whisper, russian accent",
    # GB (British) presets — common pre obe SK aj CZ jazyky
    "🇬🇧 muž, britský prízvuk":  "male, british accent",
    "🇬🇧 žena, britský prízvuk": "female, british accent",
}


_VOICE_DESIGN_PRESETS_SK = [
    "",
    "🇸🇰 slovenský muž",
    "🇸🇰 slovenská žena",
    "🇸🇰 muž, mladý dospelý",
    "🇸🇰 muž, stredný vek",
    "🇸🇰 muž, starší",
    "🇸🇰 žena, mladá dospelá",
    "🇸🇰 žena, stredný vek",
    "🇸🇰 žena, staršia",
    "🇸🇰 muž, hlboký hlas",
    "🇸🇰 žena, vysoký hlas",
    "🇸🇰 šepkanie",
    "🇬🇧 muž, britský prízvuk",
    "🇬🇧 žena, britský prízvuk",
]
_VOICE_DESIGN_PRESETS_CZ = [
    "",
    "🇨🇿 český muž",
    "🇨🇿 česká žena",
    "🇨🇿 muž, mladý dospělý",
    "🇨🇿 muž, střední věk",
    "🇨🇿 muž, starší",
    "🇨🇿 žena, mladá dospělá",
    "🇨🇿 žena, střední věk",
    "🇨🇿 žena, starší",
    "🇨🇿 muž, hluboký hlas",
    "🇨🇿 žena, vysoký hlas",
    "🇨🇿 šepot",
    "🇬🇧 muž, britský prízvuk",
    "🇬🇧 žena, britský prízvuk",
]


def _voice_design_presets_for(lang: str) -> list:
    """Vráti zoznam voice design presetov pre cieľový jazyk."""
    lang_low = (lang or "").strip().lower()
    if lang_low in ("cs", "cz", "ces"):
        return _VOICE_DESIGN_PRESETS_CZ
    return _VOICE_DESIGN_PRESETS_SK  # default SK + ostatné jazyky


def _voice_design_canon(text: str) -> str:
    """Map voice design preset → EN OmniVoice tags.
    Ak text nie je v mape, predpoklad je že user už zadal EN tagy → vráti as-is."""
    return _VOICE_DESIGN_MAP.get(text.strip(), text.strip())

IS_WINDOWS = sys.platform == "win32"
_HOME = Path.home()
# Miniforge root cez paths modul (auto-detect alebo override v ~/.config/videotranslator/paths.json)
_MINIFORGE = Path(PATHS.miniforge_root)

if IS_WINDOWS:
    _DEFAULT_PY = _HOME / "miniconda3" / "envs" / "heygen_local" / "python.exe"
else:
    # chatterbox_env has torch 2.10 — required for Turbo/Chatterbox TTS (MeanFlow decoder)
    # musetalk_env has torch 2.0.1 which produces looping audio with Turbo
    _DEFAULT_PY = _MINIFORGE / "envs" / "chatterbox_env" / "bin" / "python3"
    if not _DEFAULT_PY.exists():
        _DEFAULT_PY = _MINIFORGE / "envs" / "musetalk_env" / "python"
if not _DEFAULT_PY.exists():
    _DEFAULT_PY = Path(sys.executable)
_DEFAULT_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_DEFAULT_CHATTERBOX_MODEL_NAME = "t3_sk_v2.2.safetensors"
_EDGE_CHATTERBOX_MODEL_NAME = "t3_sk_v2.5-edge.safetensors"
_DEFAULT_ADAPT_MODEL = str(PATHS.gemma3_12b_q8)
_DEFAULT_CHATTERBOX_MODEL = _HOME / "Ai" / "models" / "chatterbox" / _DEFAULT_CHATTERBOX_MODEL_NAME
if not _DEFAULT_CHATTERBOX_MODEL.exists():
    _DEFAULT_CHATTERBOX_MODEL = BASE_DIR / "models" / "chatterbox" / _DEFAULT_CHATTERBOX_MODEL_NAME
if not _DEFAULT_CHATTERBOX_MODEL.exists():
    _DEFAULT_CHATTERBOX_MODEL = _HOME / "Ai" / "models" / "chatterbox" / _EDGE_CHATTERBOX_MODEL_NAME
if not _DEFAULT_CHATTERBOX_MODEL.exists():
    _DEFAULT_CHATTERBOX_MODEL = BASE_DIR / "models" / "chatterbox" / _EDGE_CHATTERBOX_MODEL_NAME


def _migrate_chatterbox_model_path(path_str: str) -> str:
    current = (path_str or "").strip()
    preferred_candidates = [
        _HOME / "Ai" / "models" / "chatterbox" / _DEFAULT_CHATTERBOX_MODEL_NAME,
        BASE_DIR / "models" / "chatterbox" / _DEFAULT_CHATTERBOX_MODEL_NAME,
    ]
    edge_candidates = {
        str(_HOME / "Ai" / "models" / "chatterbox" / _EDGE_CHATTERBOX_MODEL_NAME),
        str(BASE_DIR / "models" / "chatterbox" / _EDGE_CHATTERBOX_MODEL_NAME),
        _EDGE_CHATTERBOX_MODEL_NAME,
        f"models/chatterbox/{_EDGE_CHATTERBOX_MODEL_NAME}",
    }

    preferred_existing = next((str(p) for p in preferred_candidates if p.exists()), str(_DEFAULT_CHATTERBOX_MODEL))
    if not current or current in edge_candidates:
        return preferred_existing
    return current

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#1c1c26"
SIDEBAR = "#161620"
CARD    = "#242432"
BORDER  = "#2e2e40"
MUTED   = "#484860"
FG2     = "#8888a8"
FG      = "#f0f0f8"
ACCENT  = "#2a9d8f"
ACCENTL = "#3bbfaf"
ACCENTH = "#1f7a6e"
GREEN   = "#5ab5aa"
RED     = "#f87171"
YELLOW  = "#fbbf24"

EQ_BUILTIN_PRESETS: Dict[str, List[str]] = {
    "Default (Voice EQ)": ["highpass=f=45","equalizer=f=90:t=q:w=1:g=2.8","equalizer=f=180:t=q:w=1:g=1.5","lowpass=f=9500"],
    "Warm Voice": ["highpass=f=45","equalizer=f=90:t=q:w=1:g=3.2","equalizer=f=220:t=q:w=1:g=1.8","lowpass=f=10500"],
    "Speech Clarity": ["highpass=f=60","equalizer=f=3200:t=q:w=1:g=2.2","equalizer=f=4500:t=q:w=1:g=1.6","lowpass=f=12000"],
    "De-ess Soft": ["highpass=f=55","equalizer=f=4200:t=q:w=1.2:g=-2.2","equalizer=f=6200:t=q:w=1.2:g=-2.4","lowpass=f=13000"],
    "Flat (Bypass-ish)": ["highpass=f=30","lowpass=f=16000"],
    "Podcast Deep": ["highpass=f=40","equalizer=f=85:t=q:w=0.9:g=3.6","equalizer=f=2800:t=q:w=1:g=-1.4","lowpass=f=10000"],
    "Chatterbox Natural": ["highpass=f=80","equalizer=f=350:t=q:w=1.2:g=-2.5","equalizer=f=2500:t=q:w=1.0:g=+1.5","highshelf=f=9000:g=+2.0","lowpass=f=11500"],
    "Chatterbox Voice Clone": ["highpass=f=80","equalizer=f=280:t=q:w=1.2:g=-3.5","equalizer=f=2200:t=q:w=1.0:g=+1.8","highshelf=f=9000:g=+2.5","lowpass=f=11000"],
    "Chatterbox Audiobook": ["highpass=f=80","equalizer=f=350:t=q:w=1.5:g=-3.0","equalizer=f=2500:t=q:w=1.0:g=+1.0","highshelf=f=9500:g=+2.0","lowpass=f=11500"],
    "Chatterbox Podcast": ["highpass=f=80","equalizer=f=200:t=q:w=1.0:g=+1.5","equalizer=f=3000:t=q:w=1.0:g=+2.0","highshelf=f=9000:g=+1.5","lowpass=f=11500"],
    "Hegen": ["highpass=f=100","equalizer=f=450:t=q:w=1.0:g=+1.2","equalizer=f=2200:t=q:w=1.0:g=+1.0","equalizer=f=4000:t=q:w=1.2:g=-1.5","lowpass=f=12000"],
}

# ── Global QSS ─────────────────────────────────────────────────────────────────
QSS = f"""
QMainWindow, QWidget#root {{ background-color: {BG}; }}
QWidget#sidebar {{ background-color: {SIDEBAR}; }}
QFrame#sep {{ background-color: {ACCENT}; }}
QWidget {{ background-color: {BG}; color: {FG}; }}
QGroupBox {{
    background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    margin-top: 8px; padding: 8px 8px 8px 8px;
    font-size: 11px; color: {FG2};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {FG2}; background: {CARD}; }}
QLabel {{ color: {FG}; background: transparent; }}
QLineEdit {{
    background-color: {SIDEBAR}; border: 1px solid {BORDER};
    border-radius: 6px; color: {FG}; padding: 3px 8px; font-size: 12px;
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit::placeholder {{ color: {MUTED}; }}
QComboBox {{
    background-color: {SIDEBAR}; border: 1px solid {BORDER};
    border-radius: 6px; color: {FG}; padding: 2px 8px;
    font-size: 12px; min-height: 26px;
}}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {FG2}; margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {CARD}; border: 1px solid {BORDER};
    color: {FG}; selection-background-color: {ACCENT}; selection-color: #000; outline: none;
}}
QTextEdit {{
    background-color: {SIDEBAR}; border: 1px solid {BORDER};
    border-radius: 6px; color: {FG2};
    font-family: 'DejaVu Sans Mono', monospace; font-size: 11px; padding: 4px;
}}
QListWidget {{
    background-color: {CARD}; border: 1px solid {BORDER};
    border-radius: 6px; color: {FG}; font-size: 12px;
    outline: none;
}}
QListWidget::item {{ padding: 2px 6px; }}
QListWidget::item:selected {{ background-color: rgba(34,197,94,0.2); color: {ACCENTL}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACCENT}; border: none; width: 14px; height: 14px;
    border-radius: 7px; margin: -5px 0;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QCheckBox {{ color: {FG}; font-size: 12px; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid {BORDER}; background: {CARD};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QRadioButton {{ color: {FG}; font-size: 12px; spacing: 6px; }}
QRadioButton::indicator {{
    width: 14px; height: 14px; border-radius: 7px;
    border: 1px solid {BORDER}; background: {CARD};
}}
QRadioButton::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QSpinBox, QDoubleSpinBox {{
    background-color: {CARD}; border: 1px solid {BORDER};
    border-radius: 5px; color: {FG}; padding: 2px 6px; font-size: 12px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QProgressBar {{
    background-color: {CARD}; border: 1px solid {BORDER};
    border-radius: 5px; height: 7px; text-align: center; color: {FG};
    font-size: 11px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 4px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; background: {CARD}; }}
QTabBar::tab {{
    background: {SIDEBAR}; color: {FG2}; padding: 4px 12px;
    border-top-left-radius: 5px; border-top-right-radius: 5px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {CARD}; color: {FG}; }}
QScrollBar:vertical {{ background: {BG}; width: 6px; border-radius: 3px; }}
QScrollBar::handle:vertical {{ background: {MUTED}; border-radius: 3px; min-height: 20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: {BG}; height: 6px; border-radius: 3px; }}
QScrollBar::handle:horizontal {{ background: {MUTED}; border-radius: 3px; min-width: 20px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


def _pill(bg=ACCENT, hover=ACCENTL, pressed=ACCENTH, text="#000000", r=14):
    return f"""
    QPushButton {{
        background-color: {bg}; color: {text}; border: none;
        border-radius: {r}px; font-size: 12px; font-weight: bold; padding: 0 12px;
    }}
    QPushButton:hover   {{ background-color: {hover}; }}
    QPushButton:pressed {{ background-color: {pressed}; }}
    QPushButton:disabled {{ background-color: {MUTED}; color: {FG2}; }}
    """


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60:02d}s"
    return f"{s//3600}h {(s%3600)//60:02d}m"


def _probe_media_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    try:
        res = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        out = (res.stdout or "").strip()
        return float(out) if out else None
    except Exception:
        return None

def _pill_sm(bg=CARD, hover="#2e2e40", pressed="#3a3a50", text=FG, r=8):
    return f"""
    QPushButton {{
        background-color: {bg}; color: {text}; border: 1px solid {BORDER};
        border-radius: {r}px; font-size: 12px; padding: 0 10px;
    }}
    QPushButton:hover   {{ background-color: {hover}; }}
    QPushButton:pressed {{ background-color: {pressed}; }}
    """


def _nav_qss(active=False):
    if active:
        return f"""QPushButton {{
            background-color: rgba(42,157,143,0.12); color: {FG};
            border: none; border-left: 3px solid {ACCENT};
            border-radius: 0px; border-top-right-radius: 6px; border-bottom-right-radius: 6px;
            font-size: 12px; font-weight: bold;
            text-align: left; padding: 0 10px 0 9px;
        }} QPushButton:hover {{ background-color: rgba(42,157,143,0.18); }}"""
    return f"""QPushButton {{
        background-color: transparent; color: {FG2};
        border: none; border-left: 3px solid transparent;
        border-radius: 0px; border-top-right-radius: 6px; border-bottom-right-radius: 6px;
        font-size: 12px;
        text-align: left; padding: 0 10px 0 9px;
    }} QPushButton:hover {{ background-color: rgba(255,255,255,0.04); color: {FG}; }}"""


# ── Icons ──────────────────────────────────────────────────────────────────────
def _make_icon(draw_fn, size=18, color=FG2) -> QIcon:
    px = QPixmap(size, size); px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    draw_fn(p, size, QColor(r,g,b,210)); p.end()
    return QIcon(px)

def _ico_folder(p,s,c):
    pen=QPen(c,1.5); pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(2,6,s-4,s-9,2,2)
    path=QPainterPath(); path.moveTo(2,8); path.lineTo(2,5); path.lineTo(7,5); path.lineTo(9,7); path.lineTo(2,7)
    p.drawPath(path)
def _ico_play(p,s,c):
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(c))
    path=QPainterPath(); path.moveTo(4,3); path.lineTo(s-3,s//2); path.lineTo(4,s-3); path.closeSubpath(); p.drawPath(path)
def _ico_key(p,s,c):
    pen=QPen(c,1.5); p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(2,2,9,9); p.drawLine(11,11,s-2,s-2); p.drawLine(s-5,s-5,s-5,s-2); p.drawLine(s-8,s-3,s-5,s-3)
def _ico_gear(p,s,c):
    pen=QPen(c,1.5); p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    cx,cy,r=s//2,s//2,s//2-4; p.drawEllipse(cx-r+2,cy-r+2,(r-2)*2,(r-2)*2)
    for i in range(8):
        a=i*math.pi/4; p.drawLine(int(cx+(r-1)*math.cos(a)),int(cy+(r-1)*math.sin(a)),int(cx+(r+2)*math.cos(a)),int(cy+(r+2)*math.sin(a)))
def _ico_tools(p,s,c):
    pen=QPen(c,1.5); pen.setCapStyle(Qt.PenCapStyle.RoundCap); p.setPen(pen)
    p.drawLine(4,4,s-4,s-4); p.drawLine(s-7,4,s-4,7); p.drawLine(4,s-7,7,s-4)
def _ico_eq(p,s,c):
    pen=QPen(c,1.5); pen.setCapStyle(Qt.PenCapStyle.RoundCap); p.setPen(pen)
    for x,h in [(3,10),(7,5),(11,14),(15,8)]:
        p.drawLine(x,s-3,x,s-3-h); p.drawLine(x-1,s-3-h+4,x+1,s-3-h+4)
def _ico_qc(p,s,c):
    pen=QPen(c,1.5); pen.setCapStyle(Qt.PenCapStyle.RoundCap); pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(2,2,s-4,s-4,3,3)
    p.drawLine(5,7,s-5,7); p.drawLine(5,10,s-5,10); p.drawLine(5,13,int((s-5)*0.6),13)
    cp=QPainterPath(); cp.moveTo(s-8,s-6); cp.lineTo(s-6,s-4); cp.lineTo(s-3,s-9)
    p.drawPath(cp)
def _ico_train(p,s,c):
    pen=QPen(c,1.5); pen.setCapStyle(Qt.PenCapStyle.RoundCap); p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    # Brain outline
    p.drawEllipse(4,3,s-8,s-10)
    # Neural lines
    cx=s//2; cy=(s-4)//2
    p.drawLine(cx,3,cx,s-7)
    p.drawLine(4,cy,s-4,cy)
    p.drawLine(int(4+1.5),int(3+1.5),int(s-4-1.5),int(s-7-1.5))
    p.drawLine(int(s-4-1.5),int(3+1.5),int(4+1.5),int(s-7-1.5))
    # Dots at intersections
    p.setBrush(QBrush(c))
    p.setPen(Qt.PenStyle.NoPen)
    for dx,dy in [(cx,3),(cx,s-7),(4,cy),(s-4,cy)]:
        p.drawEllipse(dx-2,dy-2,4,4)

def _ico_narrator(p,s,c):
    """Mikrofón / hovorca ikona pre Narrátor tab."""
    pen=QPen(c,1.5); pen.setCapStyle(Qt.PenCapStyle.RoundCap); p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
    cx=s//2
    # mikrofón hlava
    p.drawRoundedRect(cx-3,3,6,9,3,3)
    # stojan
    p.drawArc(cx-6,8,12,8,0,-180*16)
    p.drawLine(cx,14,cx,s-3)
    p.drawLine(cx-3,s-3,cx+3,s-3)

_DRAW_FNS = [_ico_folder, _ico_play, _ico_key, _ico_tools, _ico_eq, _ico_gear, _ico_narrator]
_NAV_LABEL_KEYS = ["nav_local","nav_download","nav_api","nav_tools","nav_eq","nav_settings","nav_narrator"]
_NAV_EMOJIS = ["📁","⬇️","🔑","🔧","🎚️","⚙️","🎙️"]

# Public verzia — 6 hlavných translation engineov (zobrazené v UI dropdowne).
# Pokročilé (madlad, multislav5lang, translategemma, llama, hybrid, nllb, grammar_fix)
# sú dostupné cez settings.trans_engine ak ich nastavíš ručne v paths.json / config.
_PUBLIC_ENGINES = ["gemma", "lmstudio", "google", "chatgpt", "gemini", "grok"]
_ALL_ENGINES = ["google","madlad","multislav5lang","translategemma","llama","gemma",
                "lmstudio","hybrid","nllb","chatgpt","grok","gemini","grammar_fix"]


def _emoji_icon(emoji: str, size: int = 18) -> QIcon:
    px = QPixmap(size, size); px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    f = QFont("Noto Color Emoji")
    f.setPixelSize(int(size * 0.85))
    p.setFont(f)
    p.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, emoji)
    p.end()
    return QIcon(px)


# ── Thumbnail helper ────────────────────────────────────────────────────────────
def _extract_thumbnail(path: str, w: int = 72, h: int = 40) -> "Optional[QPixmap]":
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".jpg"))
    try:
        r = subprocess.run(
            [_DEFAULT_FFMPEG, "-ss", "2", "-i", path,
             "-frames:v", "1", "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
             "-y", str(tmp)],
            capture_output=True, timeout=6,
        )
        if r.returncode == 0 and tmp.exists():
            px = QPixmap(str(tmp))
            return px if not px.isNull() else None
    except Exception:
        pass
    finally:
        try: tmp.unlink()
        except: pass
    return None


# ── Step bar ───────────────────────────────────────────────────────────────────
class StepBar(QWidget):
    _STEP_KEYS = [("step_stt","STT"), ("step_translate","Preklad"),
                  ("step_tts","TTS"), ("step_finalize","Finalizácia")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._step = -1
        self.setFixedHeight(52)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    @property
    def _STEPS(self):
        return [T.get(k, fb) for k, fb in self._STEP_KEYS]

    def set_step(self, step: int):
        if self._step != step:
            self._step = step
            self.update()

    def reset(self):
        self.set_step(-1)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self._STEPS)
        W = self.width()
        step_w = W / n
        R = 12
        cy = 18

        for i, label in enumerate(self._STEPS):
            cx = int(step_w * i + step_w / 2)
            done   = (i < self._step)
            active = (i == self._step)

            # Connector to next step
            if i < n - 1:
                nx = int(step_w * (i + 1) + step_w / 2)
                col = ACCENTL if done else BORDER
                p.setPen(QPen(QColor(col), 2))
                p.drawLine(cx + R, cy, nx - R, cy)

            # Circle
            if active:
                p.setPen(QPen(QColor(ACCENT), 2)); p.setBrush(QBrush(QColor(ACCENT)))
            elif done:
                p.setPen(QPen(QColor(ACCENTL), 2)); p.setBrush(QBrush(QColor(ACCENTL)))
            else:
                p.setPen(QPen(QColor(BORDER), 2)); p.setBrush(QBrush(QColor(CARD)))
            p.drawEllipse(cx - R, cy - R, R * 2, R * 2)

            # Symbol
            if done:
                p.setPen(QPen(QColor("#000"), 1))
            else:
                p.setPen(QPen(QColor(FG if active else FG2)))
            p.setFont(QFont("DejaVu Sans", 8, QFont.Weight.Bold))
            p.drawText(cx - R, cy - R, R * 2, R * 2, Qt.AlignmentFlag.AlignCenter,
                       "✓" if done else str(i + 1))

            # Label
            col = FG if active else (ACCENTL if done else FG2)
            p.setPen(QPen(QColor(col)))
            p.setFont(QFont("DejaVu Sans", 8))
            p.drawText(int(cx - step_w / 2), cy + R + 4, int(step_w), 14,
                       Qt.AlignmentFlag.AlignCenter, label)
        p.end()


# ── Drop zone ──────────────────────────────────────────────────────────────────
class DropZone(QWidget):
    files_dropped = pyqtSignal(list)
    clicked_add   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._dragging = False
        self.setMinimumHeight(90)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked_add.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._dragging = True; self.update()

    def dragLeaveEvent(self, _):
        self._dragging = False; self.update()

    def dropEvent(self, e):
        _exts = {'.mp4','.mkv','.avi','.mov','.webm','.mp3','.wav','.m4a','.flac'}
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in _exts]
        if paths: self.files_dropped.emit(paths)
        self._dragging = False; self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(2, 2, -2, -2)
        bg = QColor(ACCENT + "25") if self._dragging else QColor(CARD)
        p.setBrush(QBrush(bg))
        border = QColor(ACCENTL) if self._dragging else QColor(BORDER)
        p.setPen(QPen(border, 1.5, Qt.PenStyle.DashLine))
        p.drawRoundedRect(r, 10, 10)
        cx, cy = self.width() // 2, self.height() // 2 - 8
        ic = QColor(ACCENTL) if self._dragging else QColor(MUTED)
        p.setPen(QPen(ic, 2, Qt.PenStyle.SolidLine))
        p.drawLine(cx, cy - 10, cx, cy + 6)
        p.drawLine(cx - 5, cy - 4, cx, cy - 10)
        p.drawLine(cx + 5, cy - 4, cx, cy - 10)
        p.drawLine(cx - 9, cy + 8, cx + 9, cy + 8)
        tc = QColor(ACCENTL) if self._dragging else QColor(FG2)
        p.setPen(QPen(tc)); p.setFont(QFont("DejaVu Sans", 10))
        p.drawText(0, cy + 12, self.width(), 22, Qt.AlignmentFlag.AlignCenter,
                   T.get("drop_release","Pustiť sem  ↓") if self._dragging else T.get("drop_hint","Presuň video sem  ·  alebo  ·  klikni pre výber"))
        p.setPen(QPen(QColor(MUTED))); p.setFont(QFont("DejaVu Sans", 8))
        p.drawText(0, cy + 30, self.width(), 14, Qt.AlignmentFlag.AlignCenter,
                   T.get("drop_formats","MP4 · MKV · AVI · MOV · WEBM · MP3 · WAV"))
        p.end()


# ── Collapsible section ────────────────────────────────────────────────────────
class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, parent=None, collapsed: bool = True):
        super().__init__(parent)
        self._title = title
        self._collapsed = collapsed
        self._content = content
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._btn = QPushButton()
        self._btn.setFixedHeight(24)
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.setStyleSheet(f"""
            QPushButton {{
                background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
                color: {FG}; font-size: 12px; font-weight: bold;
                text-align: left; padding: 0 12px;
            }}
            QPushButton:hover {{ border-color: {ACCENT}; background: {SIDEBAR}; }}
        """)
        self._btn.clicked.connect(self._toggle)
        lay.addWidget(self._btn)
        content.setVisible(not collapsed)
        lay.addWidget(content)
        self._refresh_label()

    def _refresh_label(self):
        arrow = "▶" if self._collapsed else "▼"
        self._btn.setText(f"  {arrow}  {self._title}")

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._content.setVisible(not self._collapsed)
        self._refresh_label()

    def expand(self):
        self._collapsed = False; self._content.setVisible(True); self._refresh_label()

    def set_title(self, t: str):
        self._title = t; self._refresh_label()

    def setTitle(self, t: str):
        self.set_title(t)


# ── Data ───────────────────────────────────────────────────────────────────────
@dataclass
class BatchItem:
    path: str
    status: str = "pending"  # pending/processing/done/failed
    error: str = ""


@dataclass
class AppSettings:
    py: str = str(_DEFAULT_PY)
    ffmpeg: str = _DEFAULT_FFMPEG
    def_tts_engine: str = "omnivoice"  # default OmniVoice (najlepšia SK kvalita od 2026-05-06)
    models_skip_dialog: bool = False   # ak True, nezobrazovať dialog modelov pri štarte
    refine_engine_choice: str = "Gemma 3 12B v1 (Ollama, fine-tune, default)"  # 2-LLM refine model voľba
    use_llamacpp_server: bool = True   # built-in open-source server (default ON, súčasť VTS)
    def_trans_engine: str = "lmstudio"  # LM Studio Gemma 4 26B base — overený 2026-05-06
    chatterbox_model: str = str(_DEFAULT_CHATTERBOX_MODEL)
    cb_voice: str = "(predvolený)"
    cb_default_voice: str = "chatterbox_sk_narrative.wav"
    cb_male_voice: str = "(žiadny)"
    cb_female_voice: str = "(žiadny)"
    cb_clone_auto: bool = True
    cb_clone_start: float = 0.0
    cb_clone_duration: float = 10.0
    # OmniVoice TTS (k2-fsa, Apache 2.0, 600+ jazykov zero-shot)
    ov_num_step: int = 64
    ov_timeline_mode: str = "Dynamický (per-segment)"
    ov_manual_speed: float = 1.18
    ov_dyn_max_factor: float = 1.40
    ov_dyn_min_factor: float = 0.95
    ov_european_voice: bool = False
    ov_guidance: float = 2.5
    ov_language: str = "sk"
    ov_instruct: str = ""
    ov_speed: float = 1.10
    ov_expressive: bool = False
    # Multi-voice: pri detekcii viacerých speakerov vyrobí per-speaker SK ref
    # cez Chatterbox + použije per-segment v OmniVoice batch
    ov_multi_voice: bool = False
    ov_multi_voice_n: int = 0  # 0 = auto-detect, >0 = explicit speaker count
    openai_api_key: str = ""
    openai_translate_model: str = "gpt-5-mini-2025-08-07"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    openai_tts_base_url: str = "https://api.openai.com/v1"
    grok_api_key: str = ""
    grok_translate_model: str = "grok-3-mini"
    grok_tts_model: str = "grok-3-mini"
    grok_tts_voice: str = "alloy"
    grok_tts_base_url: str = "https://api.x.ai/v1"
    gemini_api_key: str = ""
    gemini_translate_model: str = "gemini-2.5-flash"
    gemini_tts_model: str = "gemini-2.5-flash"
    gemini_tts_voice: str = "alloy"
    gemini_tts_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    hf_api_key: str = ""
    def_speech_gain: str = "2.5"
    whisper_model: str = "large-v3-turbo"
    whisper_models: List[str] = field(default_factory=lambda: [
        "large-v3-turbo", "large-v3", "large-v2", "medium", "small", "base", "tiny",
    ])
    # Vlastné TTS enginy — user-managed v Settings, appear v TTS dropdown
    # Each: {"name": "my_engine", "type": "chatterbox|omnivoice", "path": "/path/to/model"}
    custom_tts_engines: List[Dict[str, str]] = field(default_factory=list)
    def_music_volume: str = "0.08"
    def_backaudio_volume: str = "0.0"
    def_base_tempo: str = "1.00"
    def_stretch_min: str = "0.85"
    def_stretch_max: str = "1.50"
    def_pause_gap_s: str = "5.00"
    def_content_type: str = "general"
    def_vad_min_speech_ms: str = "4000"
    def_vad_min_silence_ms: str = "5000"
    def_vad_max_speech_s: str = "90.0"
    def_denoise_preset: str = "mild"
    chk_lipsync: bool = False
    # Defaults overené z benchmark testov 2026-05-06 (LM Studio + OmniVoice + tech video)
    chk_keep_music: bool = True   # Default ON — prevažne videá majú background music ktorú user chce zachovať
    vocal_separator: str = "demucs"
    chk_burn_subtitles: bool = False
    chk_context: bool = False
    chk_refine: bool = False     # 2-LLM refine (G3-12B v1 batch) — default OFF, opt-in (zdvojnásobí čas)
    chk_voice_eq: bool = True    # ON — pre OmniVoice Soft Highs EQ (anti-ostré výšky)
    chk_multi_voice: bool = False
    multi_voice_n: int = 2
    chk_voice_clone: bool = False
    chk_denoise: bool = False
    chk_auto_gain: bool = True
    chk_srt_align: bool = True   # ON — generuje SRT pre dabovaný video (užitočné pri editácii)
    chk_fill_slot: bool = False
    chk_mini_level11: bool = True   # ON — pre-TTS multi-agent rewrite zlepšuje plynulosť segmentov
    chk_level6_plus: bool = True
    chk_mixed_technical_terms: bool = True
    ui_language: str = "sk"
    window_width: int = 1280
    window_height: int = 900
    cb_exaggeration: float = 0.50
    cb_cfg: float = 0.65
    cb_temp: float = 0.75
    cb_emo_clone: bool = False
    cb_emotion_llm: bool = False
    src_lang: str = "auto (Whisper)"
    tgt_lang: str = "cs"
    trans_engine: str = "translategemma"
    eq_profile: str = "Soft Highs"   # Default — anti-ostré výšky + bass boost pre OmniVoice
    denoise_preset: str = "mild"
    adapt_provider: str = "ollama"
    adapt_ollama_url: str = "http://localhost:11434/api/generate"
    adapt_ollama_model: str = "sk-gemma4-e4b-v6"
    mux_resolution: str = ""
    ui_mode: str = "normal"
    text_adapt_enabled: bool = True
    tgemma_models_e4b: List[str] = field(default_factory=lambda: [
        str(PATHS.e4b_v9_translate),
        "scripts/models/translategemma/translategemma-12b-it-Q8_0.gguf",
        str(PATHS.gemma3_12b_q8),
    ])
    # Ollama model names pre engine=gemma (ollama beží na :11434)
    # 12B v1 default (osvedčený). 27B v1 EXPERIMENTAL (negation issue, treba retrain v2).
    tgemma_models_26b: List[str] = field(default_factory=lambda: [
        "g3-12b-translator-v1:latest",
        "g3-27b-multitask-v1:latest",
        "26b-translator-q5:latest",
        "26b-translator-q5l:latest",
        "26b-translator-q6:latest",
        "26b-translator:latest",
    ])

    def _secret_free_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("openai_api_key", "grok_api_key", "gemini_api_key"):
            data[key] = ""
        return data

    def load(self):
        if CONFIG_FILE.exists():
            try:
                d = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                d.setdefault("openai_translate_model", "gpt-5-mini-2025-08-07")
                d.setdefault("grok_translate_model", d.get("grok_tts_model", "grok-3-mini"))
                d.setdefault("gemini_translate_model", d.get("gemini_tts_model", "gemini-2.5-flash"))
                for k in asdict(self):
                    if k in ("openai_api_key", "grok_api_key", "gemini_api_key"):
                        continue
                    if k in d:
                        setattr(self, k, d[k])
                # Migrácia: starší engine name → chatterbox (default)
                # Migrácia legacy hodnôt — všetko nepodporované → omnivoice (nový default)
                if self.def_tts_engine in ("s2_pro", "chatterbox_turbo", "turbo", "xtts", "f5"):
                    self.def_tts_engine = "omnivoice"
                self.chatterbox_model = _migrate_chatterbox_model_path(self.chatterbox_model)
                if self.adapt_provider not in ("llama_cpp", "ollama"):
                    self.adapt_provider = "ollama"
                # Migrácia: tgemma_models_26b sa zmenil z GGUF ciest na ollama mená
                if any(str(p).endswith(".gguf") for p in (self.tgemma_models_26b or [])):
                    self.tgemma_models_26b = [
                        "g3-27b-multitask-v1:latest",
                        "g3-12b-translator-v1:latest",
                        "26b-translator-q5:latest",
                        "26b-translator-q5l:latest",
                        "26b-translator-q6:latest",
                        "26b-translator:latest",
                    ]
                # Migrácia: g3-12b-translator-v1 pridanie do existujúcich saved settings
                if "g3-12b-translator-v1:latest" not in self.tgemma_models_26b:
                    self.tgemma_models_26b.insert(0, "g3-12b-translator-v1:latest")
                # Migrácia: g3-27b-multitask-v1 ako 2. (experimental — negation issue)
                if "g3-27b-multitask-v1:latest" not in self.tgemma_models_26b:
                    self.tgemma_models_26b.insert(1, "g3-27b-multitask-v1:latest")
                # Migrácia engine name: gemma4-26b → gemma (po rename 2026-04-26)
                if self.trans_engine == "gemma4-26b":
                    self.trans_engine = "gemma"
            except Exception as e:
                print(f"Config load error: {e}")

    def save(self):
        CONFIG_FILE.write_text(json.dumps(self._secret_free_dict(), indent=2), encoding="utf-8")


def _load_video_lists() -> Dict[str, List[str]]:
    if VIDEO_LISTS_FILE.exists():
        try: return json.loads(VIDEO_LISTS_FILE.read_text(encoding="utf-8"))
        except: pass
    return {}

def _save_video_lists(d: Dict[str,List[str]]):
    VIDEO_LISTS_FILE.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

def _load_saved_checkpoints() -> Dict[str, Any]:
    if SAVED_CHECKPOINTS_FILE.exists():
        try: return json.loads(SAVED_CHECKPOINTS_FILE.read_text(encoding="utf-8"))
        except: pass
    return {}

def _save_saved_checkpoints(d: Dict[str,Any]):
    SAVED_CHECKPOINTS_FILE.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

def _save_checkpoint(items: List[BatchItem], idx: int, settings: Dict[str,Any]):
    data = {"items":[{"path":i.path,"status":i.status,"error":i.error} for i in items],
            "current_index":idx, "settings":settings, "timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
    CHECKPOINT_FILE.write_text(json.dumps(data,indent=2,ensure_ascii=False), encoding="utf-8")

def _load_checkpoint():
    if not CHECKPOINT_FILE.exists(): return None
    try:
        d = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        items = [BatchItem(**{k:v for k,v in i.items() if k in ("path","status","error")}) for i in d["items"]]
        return items, d["current_index"], d.get("settings", {})
    except: return None


# ── Runner thread ──────────────────────────────────────────────────────────────
class Runner(QThread):
    log_line = pyqtSignal(str)
    progress = pyqtSignal(float)
    finished = pyqtSignal(bool)

    def __init__(self, cmd: List[str], env: Optional[Dict]=None):
        super().__init__()
        self._cmd = cmd
        self._env = env
        self._proc: Optional[subprocess.Popen] = None

    def run(self):
        seg_re  = re.compile(r'[Ss]egment\s+(\d+)\s*/\s*(\d+)')
        step_re = re.compile(r'\[STEP\s+(\d+)/(\d+)\]', re.I)
        pct_re  = re.compile(r'(\d+)%\|')
        tqdm_re = re.compile(r'(\d+)/(\d+)\s*\[')
        # tqdm progress bars contain these patterns — suppress from log widget to avoid
        # flooding Qt event loop with hundreds of signals/sec (causes GUI freeze / 100% CPU)
        tqdm_noise_re = re.compile(r'%\||it/s\]|\[[\d:]+<[\d:]+')
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # Pridaj CUDA knižnice pre všetky conda envs (llama_cpp potrebuje libcudart/cublas)
        _cuda_paths = []
        for _env_pyver in [
            ("musetalk_env", "python3.10"),
            ("chatterbox_env", "python3.11"),
            ("llama_cpp_env", "python3.11"),
        ]:
            _base = f"{PATHS.miniforge_root}/envs/{_env_pyver[0]}/lib/{_env_pyver[1]}/site-packages/nvidia"
            import pathlib as _pl
            for _sub in ("cuda_runtime", "cublas", "cusparse", "nvjitlink", "cuda_nvrtc"):
                _p = f"{_base}/{_sub}/lib"
                if _pl.Path(_p).exists():
                    _cuda_paths.append(_p)
        if _cuda_paths:
            _existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(_cuda_paths) + (":" + _existing if _existing else "")
        if self._env: env.update(self._env)
        try:
            flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess,"CREATE_NO_WINDOW") else 0
            self._proc = subprocess.Popen(
                self._cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(BASE_DIR), env=env, creationflags=flags,
            )
            for raw in self._proc.stdout:
                # universal newlines (text=True) treats both \n and \r as separators —
                # tqdm uses \r to overwrite lines, so each tqdm tick becomes a "line".
                line = raw.rstrip('\r\n')
                if not line:
                    continue
                # Extract progress from tqdm lines but don't emit them to the log widget
                is_tqdm = bool(tqdm_noise_re.search(line))
                if not is_tqdm:
                    self.log_line.emit(line)
                for pattern, grp in [(seg_re,None),(step_re,None),(pct_re,"pct"),(tqdm_re,None)]:
                    m = pattern.search(line)
                    if m:
                        if grp == "pct":
                            self.progress.emit(float(m.group(1)))
                        elif m.lastindex and m.lastindex >= 2:
                            cur, tot = int(m.group(1)), int(m.group(2))
                            if tot > 0: self.progress.emit(cur / tot * 100)
                        break
            self._proc.wait()
            self.finished.emit(self._proc.returncode == 0)
        except Exception as e:
            self.log_line.emit(f"[ERROR] {e}")
            self.finished.emit(False)

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                if IS_WINDOWS:
                    subprocess.run(["taskkill","/F","/T","/PID",str(self._proc.pid)],
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    self._proc.terminate()
            except: self._proc.kill()


# ── Main window ────────────────────────────────────────────────────────────────
class StudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = AppSettings(); self.settings.load()
        self.batch_items: List[BatchItem] = []
        self._runner: Optional[Runner] = None
        self._start_time: float = 0.0
        self._last_progress: float = 0.0
        self._proc_timer = QTimer(self)
        self._proc_timer.setInterval(800)
        self._proc_timer.timeout.connect(self._on_timer_tick)
        self._pulse_state: bool = False
        self.setWindowTitle("VideoTranslator Studio")
        self.resize(self.settings.window_width, self.settings.window_height)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(QSS)
        # Top menu bar — File / Tools / Help
        self._build_menu_bar()
        # Bootstrap check — ak chýbajú esenciálne cesty (miniforge/models/knihy), upozorni
        self._check_bootstrap_paths()
        # First-run model check — ak chýbajú HF modely, ponúkni download
        self._check_models_first_run()
        self._build()
        # Aplikuj UI mód po vybudovaní widgetov
        self._apply_ui_mode(self.settings.ui_mode)
        # Auto-hide neaktívnych polí v Settings tabe
        self._apply_settings_visibility()
        # Voice folder watcher — keď user pridá/zmaže WAV v voices/, dropdowny sa auto-aktualizujú
        self._voice_watcher = QFileSystemWatcher(self)
        self._voice_watcher.addPath(str(VOICES_DIR))
        self._voice_watcher.directoryChanged.connect(lambda *_: self._refresh_voices())

    # ── Layout ─────────────────────────────────────────────────────────────────
    # ── i18n helpers ──────────────────────────────────────────────────────────
    def _tr_grp(self, key: str, fallback: str = "") -> QGroupBox:
        w = QGroupBox(T.get(key, fallback)); self._tr_widgets.append((w, key, fallback)); return w

    def _tr_lbl(self, key: str, fallback: str = "") -> QLabel:
        w = self._lbl(T.get(key, fallback)); self._tr_widgets.append((w, key, fallback)); return w

    def _tr_btn(self, key: str, fallback: str = "") -> QPushButton:
        w = QPushButton(T.get(key, fallback)); self._tr_widgets.append((w, key, fallback)); return w

    def _tr_chk(self, key: str, fallback: str = "") -> QCheckBox:
        w = QCheckBox(T.get(key, fallback)); self._tr_widgets.append((w, key, fallback)); return w

    def _tr_rb(self, key: str, fallback: str = "") -> QRadioButton:
        w = QRadioButton(T.get(key, fallback)); self._tr_widgets.append((w, key, fallback)); return w

    def _tr_reg(self, widget, key: str, fallback: str = ""):
        """Register an already-created widget for retranslation."""
        self._tr_widgets.append((widget, key, fallback)); return widget

    def _tr_tip(self, widget, key: str, fallback: str = ""):
        """Set a translatable tooltip and register it for retranslation."""
        widget.setToolTip(T.get(key, fallback))
        if not hasattr(self, "_tr_tooltips"):
            self._tr_tooltips = []
        self._tr_tooltips.append((widget, key, fallback))
        return widget

    def _on_language_changed(self, *_):
        new_lang = self._lang_cb.currentData()
        if not new_lang or new_lang == self.settings.ui_language:
            return
        self.settings.ui_language = new_lang
        self.settings.save()
        self._retranslate()

    def _retranslate(self):
        global T
        # Capture old translation table to canonicalize current widget state
        # before swapping T to the new language.
        T_old = T
        # Resolve voice combo items to canonical SK using the old T
        _voice_combo_state: Dict[str, str] = {}
        for combo_attr in ("_cb_voice",
                           "_cb_default_voice", "_cb_male_voice", "_cb_female_voice"):
            combo = getattr(self, combo_attr, None)
            if combo is not None:
                txt = combo.currentText()
                canonical = txt
                for can, key in _VOICE_SENTINELS.items():
                    if txt == can or txt == T_old.get(key, can):
                        canonical = can; break
                _voice_combo_state[combo_attr] = canonical
        T = _load_language(self.settings.ui_language)
        for widget, key, fallback in self._tr_widgets:
            text = T.get(key, fallback)
            if hasattr(widget, 'setTitle'):
                widget.setTitle(text)
            elif hasattr(widget, 'setText'):
                widget.setText(text)
        # Update nav buttons
        for i, k in enumerate(_NAV_LABEL_KEYS):
            if i < len(self._nav_btns):
                self._nav_btns[i].setText(T.get(k, k))
        # Placeholders + paint-based widgets
        if hasattr(self, "_seg_src"):
            self._seg_src.setPlaceholderText(T.get("ph_seg_src", self._seg_src.placeholderText()))
        if hasattr(self, "_seg_tgt"):
            self._seg_tgt.setPlaceholderText(T.get("ph_seg_tgt", self._seg_tgt.placeholderText()))
        if hasattr(self, "_step_bar"):
            self._step_bar.update()
        if hasattr(self, "_drop_zone"):
            self._drop_zone.update()
        # UI mode dropdown items (preserve selected index)
        if hasattr(self, "_ui_mode_cb"):
            idx = self._ui_mode_cb.currentIndex()
            self._ui_mode_cb.blockSignals(True)
            self._ui_mode_cb.clear()
            self._ui_mode_cb.addItems([T.get("mode_normal","Normal"), T.get("mode_extended","Rozšírený")])
            self._ui_mode_cb.setCurrentIndex(max(0, idx))
            self._ui_mode_cb.blockSignals(False)
        # YouTube cookies dropdown
        if hasattr(self, "_yt_cookies"):
            idx = self._yt_cookies.currentIndex()
            self._yt_cookies.blockSignals(True)
            self._yt_cookies.clear()
            self._yt_cookies.addItems([T.get("yt_cookies_none","(žiadne)"),"firefox","chrome","chromium","brave","edge"])
            self._yt_cookies.setCurrentIndex(max(0, idx))
            self._yt_cookies.blockSignals(False)
        # Mux resolution: only the first item ("(original)") needs translating
        if hasattr(self, "_mux_resolution"):
            idx = self._mux_resolution.currentIndex()
            self._mux_resolution.setItemText(0, T.get("mux_res_original","(pôvodné)"))
            self._mux_resolution.setCurrentIndex(max(0, idx))
        # Voice comboboxes: re-translate sentinel items (filenames stay)
        for combo_attr in ("_cb_voice",
                           "_cb_default_voice", "_cb_male_voice", "_cb_female_voice"):
            combo = getattr(self, combo_attr, None)
            if combo is None:
                continue
            for i in range(combo.count()):
                txt = combo.itemText(i)
                # Identify sentinel by checking against canonical and old translation
                for can, key in _VOICE_SENTINELS.items():
                    if txt == can or txt == T_old.get(key, can):
                        combo.setItemText(i, T.get(key, can))
                        break
            # Restore the previously-selected canonical (now in the new language)
            target_canonical = _voice_combo_state.get(combo_attr)
            if target_canonical:
                target_disp = _voice_disp(target_canonical) if target_canonical in _VOICE_SENTINELS else target_canonical
                combo.blockSignals(True)
                idx = combo.findText(target_disp)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                combo.blockSignals(False)
        # Tooltips registered via _tr_tip
        for widget, key, fallback in getattr(self, "_tr_tooltips", []):
            widget.setToolTip(T.get(key, fallback))
        # Refresh dynamic emo_clone label/tooltip after language switch
        if hasattr(self, "_sync_clone_controls"):
            self._sync_clone_controls()

    def _build(self):
        self._tr_widgets: list = []
        self._tr_tooltips: list = []
        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QHBoxLayout(root); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._build_sidebar())
        sep = QFrame(); sep.setObjectName("sep"); sep.setFixedWidth(3)
        lay.addWidget(sep)
        self._stack = QStackedWidget()
        for fn in [self._page_local, self._page_youtube, self._page_api,
                   self._page_tools, self._page_eq, self._page_settings,
                   self._page_narrator]:
            self._stack.addWidget(fn())
        self._stack.setCurrentIndex(0)
        lay.addWidget(self._stack, 1)

        # ── Persistent status bar ──────────────────────────────────────────────
        sb = self.statusBar()
        sb.setStyleSheet(f"""
            QStatusBar {{ background:{SIDEBAR}; border-top:1px solid {BORDER};
                          color:{FG2}; padding:0 10px; min-height:28px; }}
            QStatusBar::item {{ border:none; }}
        """)
        self._badge = self._tr_reg(
            QLabel(T.get("status_ready","● Pripravené")), "status_ready", "● Pripravené")
        self._badge.setFont(QFont("DejaVu Sans", 10, QFont.Weight.Bold))
        self._badge.setStyleSheet(f"color:{GREEN}; background:transparent;")
        sb.addWidget(self._badge)
        self._time_label = QLabel(""); self._time_label.setStyleSheet(f"color:{FG2}; font-size:11px; background:transparent;")
        sb.addWidget(self._time_label)
        self._progress = QProgressBar(); self._progress.setRange(0, 100); self._progress.setValue(0)
        self._progress.setFixedWidth(200); self._progress.setFixedHeight(7)
        self._progress.setTextVisible(False)
        sb.addPermanentWidget(self._progress)

    def _build_sidebar(self) -> QWidget:
        sb = QWidget(); sb.setObjectName("sidebar"); sb.setFixedWidth(210)
        lay = QVBoxLayout(sb); lay.setContentsMargins(12,16,12,14); lay.setSpacing(2)
        t1 = QLabel("VideoTranslator"); t1.setFont(QFont("DejaVu Sans",13,QFont.Weight.Bold))
        t1.setStyleSheet(f"color:{FG};")
        t2 = QLabel("Studio"); t2.setFont(QFont("DejaVu Sans",9))
        t2.setStyleSheet(f"color:{FG2};")
        lay.addWidget(t1); lay.addWidget(t2); lay.addSpacing(12)
        sect = self._tr_reg(QLabel(T.get("side_nav","NAVIGÁCIA")), "side_nav", "NAVIGÁCIA")
        sect.setFont(QFont("DejaVu Sans",8,QFont.Weight.Bold))
        sect.setStyleSheet(f"color:{FG2}; letter-spacing:1px;")
        lay.addWidget(sect); lay.addSpacing(2)
        self._nav_btns: List[QPushButton] = []
        for i, label in enumerate(T.get(k, k) for k in _NAV_LABEL_KEYS):
            btn = QPushButton(label); btn.setFixedHeight(24)
            btn.setIcon(_emoji_icon(_NAV_EMOJIS[i],18)); btn.setIconSize(QSize(18,18))
            btn.setStyleSheet(_nav_qss(active=(i==0)))
            btn.clicked.connect(lambda _,idx=i: self._select_view(idx))
            self._nav_btns.append(btn); lay.addWidget(btn)
        lay.addStretch()
        # UI režim toggle
        mode_lbl = self._tr_reg(QLabel(T.get("side_mode","REŽIM")), "side_mode", "REŽIM")
        mode_lbl.setFont(QFont("DejaVu Sans",8,QFont.Weight.Bold))
        mode_lbl.setStyleSheet(f"color:{FG2}; letter-spacing:1px;")
        lay.addWidget(mode_lbl)
        self._ui_mode_cb = QComboBox()
        self._ui_mode_cb.addItems([T.get("mode_normal","Normal"), T.get("mode_extended","Rozšírený")])
        self._ui_mode_cb.setCurrentIndex(1 if self.settings.ui_mode == "extended" else 0)
        self._ui_mode_cb.setFixedHeight(24)
        self._ui_mode_cb.currentTextChanged.connect(self._on_ui_mode_changed)
        lay.addWidget(self._ui_mode_cb)
        lay.addSpacing(8)
        ver = QLabel("v2.0 · Qt6"); ver.setFont(QFont("DejaVu Sans",9))
        ver.setStyleSheet(f"color:{MUTED};"); ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(ver)
        return sb

    def _select_view(self, idx: int):
        for i,btn in enumerate(self._nav_btns):
            active=(i==idx)
            btn.setIcon(_emoji_icon(_NAV_EMOJIS[i],18))
            btn.setStyleSheet(_nav_qss(active=active))
        self._stack.setCurrentIndex(idx)

    def _persist_batch_state(self):
        snap = self._checkpoint_settings_snapshot()
        done_count = sum(1 for i in self.batch_items if i.status == "done")
        _save_checkpoint(self.batch_items, done_count, snap)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 0 — Local file
    # ══════════════════════════════════════════════════════════════════════════
    def _page_local(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(16,14,16,14); lay.setSpacing(10)

        # ── Step indicator ────────────────────────────────────────────────────
        self._step_bar = StepBar()
        lay.addWidget(self._step_bar)

        # ── Batch file list ──────────────────────────────────────────────────
        grp_files = self._tr_grp("grp_files","Vstupné súbory (batch)")
        fl = QVBoxLayout(grp_files); fl.setSpacing(5)

        # Drop zone (visible when list is empty)
        self._drop_zone = DropZone()
        self._drop_zone.clicked_add.connect(self._batch_add)
        self._drop_zone.files_dropped.connect(self._on_files_dropped)
        fl.addWidget(self._drop_zone)

        # File list (visible when files present)
        self._batch_list = QListWidget()
        self._batch_list.setIconSize(QSize(72, 40))
        self._batch_list.setMaximumHeight(160)
        self._batch_list.setVisible(False)
        fl.addWidget(self._batch_list)

        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        for key, fb, fn in [("btn_add","Pridať",self._batch_add),("btn_remove","Odstrániť",self._batch_remove),
                             ("btn_clear","Vymazať",self._batch_clear),("btn_save_list","Uložiť zoznam",self._batch_save_list),
                             ("btn_load_list","Načítať zoznam",self._batch_load_list)]:
            b = self._tr_btn(key, fb); b.setFixedHeight(24); b.setStyleSheet(_pill_sm())
            b.clicked.connect(fn); btn_row.addWidget(b)
        btn_row.addStretch()
        b_chk = self._tr_btn("btn_checkpoint","Checkpoint ▾"); b_chk.setFixedHeight(24); b_chk.setStyleSheet(_pill_sm())
        b_chk.clicked.connect(self._checkpoint_menu); btn_row.addWidget(b_chk)
        fl.addLayout(btn_row)
        # Output dir
        orow = QHBoxLayout(); orow.setSpacing(8)
        orow.addWidget(self._tr_lbl("lbl_output_dir","Výstupný priečinok:"))
        self._out_dir = QLineEdit(); self._out_dir.setPlaceholderText(T.get("lbl_output_dir_ph","(rovnaký ako vstup)"))
        self._out_dir.setFixedHeight(26)
        b_out = QPushButton("…"); b_out.setFixedSize(32,32); b_out.setStyleSheet(_pill_sm())
        b_out.clicked.connect(lambda: self._out_dir.setText(QFileDialog.getExistingDirectory(self,"Výstupný priečinok") or self._out_dir.text()))
        orow.addWidget(self._out_dir,1); orow.addWidget(b_out)
        fl.addLayout(orow)
        lay.addWidget(grp_files)

        # ── Languages + Engine ───────────────────────────────────────────────
        grp_lang = self._tr_grp("grp_langs","Jazyky a prekladový engine")
        ll = QGridLayout(grp_lang); ll.setSpacing(5)
        # Comprehensive languages (OmniVoice 646 supported, Whisper STT pre populárne).
        # Logické poradie: hlavné slavské + EU + globálne.
        langs = [
            # Slavské + okolité
            "sk", "cs", "pl", "hu", "ru", "uk", "be", "hr", "sl", "sr", "bg", "mk",
            # Hlavné EU
            "en", "de", "fr", "es", "it", "pt", "nl",
            # Severské
            "sv", "no", "da", "fi",
            # Pobaltské
            "et", "lv", "lt",
            # Ďalšie EU
            "ro", "el", "tr", "sq",
            # Globálne
            "zh", "ja", "ko", "arb", "hi", "vi", "th", "id", "he", "fa",
        ]
        ll.addWidget(self._tr_lbl("lbl_src_lang","Zdrojový jazyk"),0,0)
        self._src_lang = QComboBox(); self._src_lang.addItems(["auto (Whisper)"] + langs)
        self._src_lang.setCurrentText(self.settings.src_lang if self.settings.src_lang in ["auto (Whisper)"]+langs else "auto (Whisper)")
        self._src_lang.currentTextChanged.connect(self._save_chk_state)
        self._src_lang.setFixedHeight(26); ll.addWidget(self._src_lang,0,1)
        ll.addWidget(self._tr_lbl("lbl_tgt_lang","Cieľový jazyk"),0,2)
        self._tgt_lang = QComboBox(); self._tgt_lang.addItems(langs)
        self._tgt_lang.setCurrentText(self.settings.tgt_lang if self.settings.tgt_lang in langs else "cs")
        self._tgt_lang.currentTextChanged.connect(self._save_chk_state)
        self._tgt_lang.currentTextChanged.connect(self._refresh_voice_design_presets)
        self._tgt_lang.setFixedHeight(26); ll.addWidget(self._tgt_lang,0,3)
        ll.addWidget(self._tr_lbl("lbl_trans_engine","Prekladový engine"),1,0)
        self._trans_engine = QComboBox()
        # Public verzia: 5 hlavných engineov. Pokročilé (madlad, nllb, hybrid, ...) v Settings.
        # Cesta k modelu (GGUF / API model) sa nastavuje výhradne v Settings → Predvolené hodnoty.
        self._trans_engine.addItems(_PUBLIC_ENGINES)
        saved_engine = self.settings.trans_engine if self.settings.trans_engine in _PUBLIC_ENGINES else "gemma"
        self._trans_engine.setCurrentText(saved_engine)
        self._trans_engine.currentTextChanged.connect(self._save_chk_state)
        self._trans_engine.currentTextChanged.connect(self._on_engine_changed)
        self._trans_engine.setFixedHeight(26); ll.addWidget(self._trans_engine,1,1, 1, 3)

        # Model dropdowns — nemiestnené v hlavnom layoute, len v Settings.
        # Widgety ostávajú v pamäti, downstream kód (CLI build, settings sync) ich číta.
        self._madlad_model = QComboBox()
        self._madlad_model.setEditable(True)
        self._madlad_model.addItems(list(self.settings.tgemma_models_e4b))
        self._madlad_model.lineEdit().editingFinished.connect(self._on_tgemma_text_edited)
        self._nllb_model_cb = QComboBox()
        self._nllb_model_cb.addItems(["facebook/nllb-200-1.3B","facebook/nllb-200-3.3B"])
        # Inicializácia obsahu modelového zoznamu podľa enginu (pre Settings sync)
        self._on_engine_changed(self._trans_engine.currentText())
        lay.addWidget(grp_lang)

        # ── Content type ─────────────────────────────────────────────────────
        self._grp_ct = self._tr_grp("grp_content_type","Typ obsahu (štýl prekladu — pre všetky engines)")
        grp_ct = self._grp_ct
        ctl = QHBoxLayout(grp_ct); ctl.setSpacing(4)
        self._content_grp = QButtonGroup(self)
        for i,(key,fb,val) in enumerate([("ct_general","Všeobecný","general"),("ct_educational","Náučné","educational"),
                                          ("ct_podcast","Podcast","podcast"),("ct_review","Recenzia","review"),
                                          ("ct_technical","Technický","technical"),("ct_programming","Programovanie","programming"),
                                          ("ct_news","Správy","news"),
                                          ("ct_entertainment","Zábava","entertainment"),("ct_sql","SQL/DB","sql")]):
            rb = self._tr_rb(key, fb); rb.setProperty("val",val)
            if val == self.settings.def_content_type: rb.setChecked(True)
            self._content_grp.addButton(rb,i); ctl.addWidget(rb)
        ctl.addStretch()
        lay.addWidget(grp_ct)

        # ── Existujúce titulky (SRT) — preskočí STT + translation ───────────
        grp_srt = self._tr_grp("grp_srt", "Existujúce titulky (SRT) — preskočí STT + preklad")
        srt_l = QHBoxLayout(grp_srt); srt_l.setSpacing(5)
        self._srt_in = QLineEdit("")
        self._srt_in.setPlaceholderText("Cesta k SRT (cieľový jazyk SK/CS — iba TTS, žiadny preklad)")
        self._srt_in.setFixedHeight(26)
        self._srt_in.editingFinished.connect(self._save_chk_state)
        srt_l.addWidget(self._srt_in)
        b_srt = QPushButton("…"); b_srt.setFixedSize(30, 30); b_srt.setStyleSheet(_pill_sm())
        b_srt.clicked.connect(lambda: self._srt_in.setText(
            QFileDialog.getOpenFileName(self, "Vyber SRT titulky", "",
                                         "SRT (*.srt);;All (*)")[0] or self._srt_in.text()))
        srt_l.addWidget(b_srt)
        b_srt_clear = QPushButton("✕"); b_srt_clear.setFixedSize(28, 28); b_srt_clear.setStyleSheet(_pill_sm())
        self._tr_tip(b_srt_clear, "tip_srt_clear", "Vymazať SRT — pipeline pôjde Whisper STT cestou")
        b_srt_clear.clicked.connect(lambda: self._srt_in.setText(""))
        srt_l.addWidget(b_srt_clear)
        lay.addWidget(grp_srt)

        # ── TTS engine ───────────────────────────────────────────────────────
        grp_tts = self._tr_grp("grp_tts","TTS Engine")
        ttsl = QVBoxLayout(grp_tts); ttsl.setSpacing(5)
        tts_radio_row = QHBoxLayout()
        self._tts_grp = QButtonGroup(self)
        # OmniVoice ako default + prvý v poradí (od 2026-05-06 — najlepšia SK kvalita)
        self._rb_omnivoice = QRadioButton("OmniVoice 🌍 (predvolený)")
        self._rb_chatterbox = QRadioButton("Chatterbox")
        self._tts_grp.addButton(self._rb_omnivoice, 0)
        self._tts_grp.addButton(self._rb_chatterbox, 1)
        tts_radio_row.addWidget(self._rb_omnivoice); tts_radio_row.addWidget(self._rb_chatterbox)
        tts_radio_row.addStretch(); ttsl.addLayout(tts_radio_row)
        self._rb_omnivoice.setChecked(True)  # default OmniVoice

        # Chatterbox options (stacked)
        self._tts_stack = QStackedWidget()
        cb_w = QWidget(); cb_l = QGridLayout(cb_w); cb_l.setSpacing(5)
        cb_l.addWidget(self._tr_lbl("lbl_model","Model:"),0,0)
        self._cb_model = QLineEdit(self.settings.chatterbox_model); self._cb_model.setFixedHeight(26)
        self._cb_model.editingFinished.connect(self._save_chk_state)
        b_cbm = QPushButton("…"); b_cbm.setFixedSize(30,30); b_cbm.setStyleSheet(_pill_sm())
        b_cbm.clicked.connect(self._pick_cb_model)
        cb_l.addWidget(self._cb_model,0,1); cb_l.addWidget(b_cbm,0,2)
        cb_l.addWidget(self._tr_lbl("cb_voice_lbl","Hlas (WAV):"),1,0)
        self._cb_voice = QComboBox(); self._cb_voice.setFixedHeight(26)
        self._cb_voice.addItems(self._get_voices())
        if self.settings.cb_voice in [self._cb_voice.itemText(i) for i in range(self._cb_voice.count())]:
            self._cb_voice.setCurrentText(self.settings.cb_voice)
        b_cbv = QPushButton("↺"); b_cbv.setFixedSize(30,30); b_cbv.setStyleSheet(_pill_sm())
        b_cbv.clicked.connect(self._refresh_voices)
        cb_l.addWidget(self._cb_voice,1,1); cb_l.addWidget(b_cbv,1,2)
        cb_l.addWidget(self._tr_lbl("lbl_default","Predvolený:"),2,0)
        self._cb_default_voice = QComboBox(); self._cb_default_voice.setFixedHeight(24)
        _cb_wav_only = [_voice_disp("(žiadny)")] + sorted(p.name for p in VOICES_DIR.glob("*.wav"))
        self._cb_default_voice.addItems(_cb_wav_only)
        self._tr_tip(self._cb_default_voice, "tip_default_voice", "Hlas použitý keď je hlavný výber '(predvolený)' — namiesto auto-extrakcie z videa.")
        self._set_combo_value(self._cb_default_voice, self.settings.cb_default_voice, "(žiadny)")
        self._cb_default_voice.currentTextChanged.connect(self._save_chk_state)
        cb_l.addWidget(self._cb_default_voice,2,1,1,2)
        self._cb_male_voice_lbl = self._tr_lbl("lbl_male_fallback","Mužský fallback:")
        cb_l.addWidget(self._cb_male_voice_lbl,3,0)
        self._cb_male_voice = QComboBox(); self._cb_male_voice.setFixedHeight(24)
        self._cb_male_voice.addItems(_cb_wav_only)
        self._tr_tip(self._cb_male_voice, "tip_male_voice", "Voliteľný mužský WAV pre Multi-voice. Aktivuje sa zaškrtnutím 'Multi-voice' v Možnostiach.")
        self._set_combo_value(self._cb_male_voice, self.settings.cb_male_voice, "(žiadny)")
        self._cb_male_voice.currentTextChanged.connect(self._save_chk_state)
        cb_l.addWidget(self._cb_male_voice,3,1,1,2)
        self._cb_female_voice_lbl = self._tr_lbl("lbl_female_fallback","Ženský fallback:")
        cb_l.addWidget(self._cb_female_voice_lbl,4,0)
        self._cb_female_voice = QComboBox(); self._cb_female_voice.setFixedHeight(24)
        self._cb_female_voice.addItems(_cb_wav_only)
        self._tr_tip(self._cb_female_voice, "tip_female_voice", "Voliteľný ženský WAV pre Multi-voice. Aktivuje sa zaškrtnutím 'Multi-voice' v Možnostiach.")
        self._set_combo_value(self._cb_female_voice, self.settings.cb_female_voice, "(žiadny)")
        self._cb_female_voice.currentTextChanged.connect(self._save_chk_state)
        cb_l.addWidget(self._cb_female_voice,4,1,1,2)
        self._cb_clone_auto = self._tr_chk("chk_clone_auto_best","Clone z videa: auto najlepší sample")
        self._cb_clone_auto.setChecked(self.settings.cb_clone_auto)
        cb_l.addWidget(self._cb_clone_auto,5,0,1,3)
        self._cb_clone_start_lbl = self._tr_lbl("lbl_clone_start_s","Clone start (s):")
        self._cb_clone_start = QDoubleSpinBox(); self._cb_clone_start.setRange(0, 99999); self._cb_clone_start.setDecimals(2)
        self._cb_clone_start.setSingleStep(1.0); self._cb_clone_start.setValue(self.settings.cb_clone_start); self._cb_clone_start.setFixedHeight(26)
        self._cb_clone_dur_lbl = self._tr_lbl("lbl_clone_duration_s","Clone dĺžka (s):")
        self._cb_clone_dur = QDoubleSpinBox(); self._cb_clone_dur.setRange(1, 600); self._cb_clone_dur.setDecimals(2)
        self._cb_clone_dur.setSingleStep(1.0); self._cb_clone_dur.setValue(self.settings.cb_clone_duration); self._cb_clone_dur.setFixedHeight(26)
        cb_l.addWidget(self._cb_clone_start_lbl,6,0); cb_l.addWidget(self._cb_clone_start,6,1)
        cb_l.addWidget(self._cb_clone_dur_lbl,7,0); cb_l.addWidget(self._cb_clone_dur,7,1)
        self._cb_emo_clone = self._tr_chk("chk_emo_video","Emócia z videa")
        # Emócie — inline row
        # Inline emotion sliders (Expr/CFG/Temp) presunuté do Settings → Chatterbox TTS parametre
        cb_l.setColumnStretch(1,1); self._tts_stack.addWidget(cb_w)

        # OmniVoice options (stack index 1)
        ov_w = QWidget(); ov_l = QGridLayout(ov_w); ov_l.setSpacing(5)
        ov_l.addWidget(self._tr_lbl("cb_voice_lbl","Hlas (WAV):"), 0, 0)
        # Voice picker — mirror _cb_voice (zmena tu propaguje aj do _cb_voice cez sync handler)
        ov_voice_row = QWidget()
        ov_vh = QHBoxLayout(ov_voice_row); ov_vh.setContentsMargins(0,0,0,0); ov_vh.setSpacing(4)
        self._ov_voice = QComboBox(); self._ov_voice.setFixedHeight(26)
        self._ov_voice.addItems(self._get_voices())
        self._ov_voice.currentTextChanged.connect(self._sync_voice_from_omnivoice)
        ov_vh.addWidget(self._ov_voice, 1)
        # ➕ pridať voice (file dialog, copy do voices/)
        b_add_voice = QPushButton("+"); b_add_voice.setFixedSize(26, 26)
        b_add_voice.setToolTip("Pridať WAV súbor do voices/ adresára")
        b_add_voice.clicked.connect(self._add_voice_file)
        ov_vh.addWidget(b_add_voice)
        # ↻ refresh
        b_refresh_voice = QPushButton("↻"); b_refresh_voice.setFixedSize(26, 26)
        b_refresh_voice.setToolTip("Prečítaj voices/ priečinok znova")
        b_refresh_voice.clicked.connect(self._refresh_voices)
        ov_vh.addWidget(b_refresh_voice)
        ov_l.addWidget(ov_voice_row, 0, 1, 1, 2)
        ov_l.addWidget(self._tr_lbl("ov_diffusion_steps","Diffusion stepy:"), 1, 0)
        self._ov_num_step = QSpinBox(); self._ov_num_step.setRange(16, 256)
        self._ov_num_step.setValue(int(getattr(self.settings, "ov_num_step", 64)))
        self._ov_num_step.setFixedHeight(26); self._ov_num_step.setSingleStep(8)
        self._tr_tip(self._ov_num_step, "tip_ov_num_step", "Vyššie = lepšia kvalita, pomalšie. Default 32, odporúčané 64.")
        ov_l.addWidget(self._ov_num_step, 1, 1)
        ov_l.addWidget(self._tr_lbl("ov_step_hint", "(32=default, 64=lepšie, 128=top)"), 1, 2)
        ov_l.addWidget(self._tr_lbl("ov_guidance_lbl","Guidance scale:"), 2, 0)
        self._ov_guidance = QDoubleSpinBox(); self._ov_guidance.setRange(1.0, 5.0)
        self._ov_guidance.setSingleStep(0.1); self._ov_guidance.setDecimals(2)
        self._ov_guidance.setValue(float(getattr(self.settings, "ov_guidance", 2.5)))
        self._ov_guidance.setFixedHeight(26)
        self._tr_tip(self._ov_guidance, "tip_ov_guidance", "Vyššie = silnejšie tracking ref voice. Default 2.0, odporúčané 2.5.")
        ov_l.addWidget(self._ov_guidance, 2, 1)
        ov_l.addWidget(self._tr_lbl("ov_guidance_hint", "(2.0=default, 2.5=odporúčané)"), 2, 2)
        # OmniVoice jazyk = používa cieľový jazyk z hlavnej voľby (Cieľový jazyk).
        # Žiadny duplicitný dropdown — _ov_language odstránený, _tgt_lang je single source.

        ov_l.addWidget(self._tr_lbl("ov_voice_design_lbl","Voice design:"), 4, 0)
        self._ov_instruct = QComboBox(); self._ov_instruct.setEditable(True)
        self._ov_instruct.setFixedHeight(26)
        # Per-language presets — rebuild pri zmene cieľového jazyka cez
        # _refresh_voice_design_presets()
        for preset in _voice_design_presets_for(self.settings.tgt_lang or "sk"):
            self._ov_instruct.addItem(preset)
        # Voice design sa pri každom štarte GUI resetuje na default (prázdne).
        # Dôvod: voice design je "ad-hoc" voľba pre konkrétny dub, nemá zmysel pamätať
        # si ju medzi behmi (gender + accent musia matchovať video, čo sa mení per-vide).
        self._ov_instruct.setCurrentText("")
        self._tr_tip(self._ov_instruct, "tip_ov_instruct",
            "Popis hlasu (voice design). Vlajka označuje jazyk:\n"
            "🇸🇰 slovenský, 🇨🇿 český, 🇬🇧 britský. Mapuje sa na OmniVoice tagy.")
        ov_l.addWidget(self._ov_instruct, 4, 1)
        ov_l.addWidget(self._tr_lbl("ov_instruct_hint", "(napr. 'slovenský muž' alebo 'muž, mladý dospelý')"), 4, 2)

        # Timeline sync mode NAJPRV — určuje čo sa dá ďalej upravovať
        ov_l.addWidget(self._tr_lbl("ov_timeline_lbl","Sync s videom:"), 5, 0)
        self._ov_timeline_mode = QComboBox(); self._ov_timeline_mode.setFixedHeight(26)
        self._ov_timeline_mode.addItems([
            "Dynamický (per-segment)",
            "Manuálny (uniform speed)",
            "Žiadny (raw audio)",
        ])
        _saved_tl = getattr(self.settings, "ov_timeline_mode", "Dynamický (per-segment)")
        if self._ov_timeline_mode.findText(_saved_tl) >= 0:
            self._ov_timeline_mode.setCurrentText(_saved_tl)
        ov_l.addWidget(self._ov_timeline_mode, 5, 1)
        ov_l.addWidget(self._tr_lbl("ov_timeline_hint",
            "(Dynamický: per-segment, Manuálny: rovnaké tempo, Žiadny: bez sync)"), 5, 2)
        self._ov_timeline_mode.currentTextChanged.connect(self._save_chk_state)

        # Speed controls POD timeline mode — viditeľnosť riadi timeline mode
        self._ov_speed_lbl = self._tr_lbl("ov_speed_lbl","Rýchlosť reči:")
        ov_l.addWidget(self._ov_speed_lbl, 6, 0)
        self._ov_speed = QDoubleSpinBox(); self._ov_speed.setRange(0.5, 2.0)
        self._ov_speed.setSingleStep(0.05); self._ov_speed.setDecimals(2)
        self._ov_speed.setValue(float(getattr(self.settings, "ov_speed", 1.10)))
        self._ov_speed.setFixedHeight(26)
        ov_l.addWidget(self._ov_speed, 6, 1)
        self._ov_speed_hint = self._tr_lbl("ov_speed_hint", "(1.10=odporúčané, 1.0=pomalšie, 1.18=energický)")
        ov_l.addWidget(self._ov_speed_hint, 6, 2)

        # Manual speed factor — used iba keď timeline_mode=Manuálny
        self._ov_manual_speed_lbl = self._tr_lbl("ov_manual_speed_lbl","Manuálna rýchlosť:")
        ov_l.addWidget(self._ov_manual_speed_lbl, 7, 0)
        self._ov_manual_speed = QDoubleSpinBox(); self._ov_manual_speed.setRange(0.5, 2.0)
        self._ov_manual_speed.setSingleStep(0.05); self._ov_manual_speed.setDecimals(2)
        self._ov_manual_speed.setValue(float(getattr(self.settings, "ov_manual_speed", 1.18)))
        self._ov_manual_speed.setFixedHeight(26)
        ov_l.addWidget(self._ov_manual_speed, 7, 1)
        self._ov_manual_speed_hint = self._tr_lbl("ov_manual_speed_hint",
            "(1.18 = match 8min EN → 7min SK; iba pre Manuálny mode)")
        ov_l.addWidget(self._ov_manual_speed_hint, 7, 2)
        self._ov_manual_speed.valueChanged.connect(self._save_chk_state)

        # Dynamic mode: min/max factor limits
        self._ov_dyn_min_lbl = self._tr_lbl("ov_dyn_min_lbl","Min tempo (dynamický):")
        ov_l.addWidget(self._ov_dyn_min_lbl, 8, 0)
        self._ov_dyn_min = QDoubleSpinBox(); self._ov_dyn_min.setRange(0.85, 1.00)
        self._ov_dyn_min.setSingleStep(0.01); self._ov_dyn_min.setDecimals(2)
        self._ov_dyn_min.setValue(float(getattr(self.settings, "ov_dyn_min_factor", 0.95)))
        self._ov_dyn_min.setFixedHeight(26)
        ov_l.addWidget(self._ov_dyn_min, 8, 1)
        self._ov_dyn_min_hint = self._tr_lbl("ov_dyn_min_hint",
            "(pod touto hranicou sa NEspomalí, slot sa doplní tichom)")
        ov_l.addWidget(self._ov_dyn_min_hint, 8, 2)
        self._ov_dyn_min.valueChanged.connect(self._save_chk_state)

        self._ov_dyn_max_lbl = self._tr_lbl("ov_dyn_max_lbl","Max tempo (dynamický):")
        ov_l.addWidget(self._ov_dyn_max_lbl, 9, 0)
        self._ov_dyn_max = QDoubleSpinBox(); self._ov_dyn_max.setRange(1.10, 1.80)
        self._ov_dyn_max.setSingleStep(0.05); self._ov_dyn_max.setDecimals(2)
        self._ov_dyn_max.setValue(float(getattr(self.settings, "ov_dyn_max_factor", 1.40)))
        self._ov_dyn_max.setFixedHeight(26)
        ov_l.addWidget(self._ov_dyn_max, 9, 1)
        self._ov_dyn_max_hint = self._tr_lbl("ov_dyn_max_hint",
            "(max zrýchlenie pre dlhé segmenty; nad tým overlap do nasledujúceho slotu)")
        ov_l.addWidget(self._ov_dyn_max_hint, 9, 2)
        self._ov_dyn_max.valueChanged.connect(self._save_chk_state)

        # Accent transfer UI bola odstránená 2026-05-08 — nahradila ju kvalitnejšia
        # cesta cez Chatterbox-as-ref-generator v auto_clone_ref.py
        # (SK akcent zapečený v Chatterbox SK 2.2 modeli, OmniVoice klonuje z SK-flavored ref).

        # ── Multi-voice (per-speaker cloning) ───────────────────────────
        self._ov_multi_voice = self._tr_chk("ov_multi_voice",
            "🎭 Multi-voice (per-speaker SK clone)")
        self._ov_multi_voice.setChecked(bool(getattr(self.settings, "ov_multi_voice", False)))
        self._tr_tip(self._ov_multi_voice, "tip_ov_multi_voice",
                     "Diarizácia (Resemblyzer + KMeans) → per-speaker Chatterbox SK ref-gen → "
                     "OmniVoice TTS s rôznym hlasom per speaker. Auto-zaškrtne sa pri detekcii "
                     "multispeaker v Auto-detection.")
        self._ov_multi_voice.toggled.connect(self._save_chk_state)
        ov_l.addWidget(self._ov_multi_voice, 10, 0, 1, 3)
        # Show/hide widgets podľa timeline mode
        def _upd_manual_visible():
            mode = self._ov_timeline_mode.currentText()
            show_manual = mode.startswith("Manuálny")
            show_speech = not mode.startswith("Dynamický")
            show_dyn = mode.startswith("Dynamický")
            # Manual speed (iba Manuálny mode)
            self._ov_manual_speed_lbl.setVisible(show_manual)
            self._ov_manual_speed.setVisible(show_manual)
            self._ov_manual_speed_hint.setVisible(show_manual)
            # Speech speed (Manuálny + Žiadny)
            self._ov_speed_lbl.setVisible(show_speech)
            self._ov_speed.setVisible(show_speech)
            self._ov_speed_hint.setVisible(show_speech)
            # Dynamic min/max (iba Dynamický)
            self._ov_dyn_min_lbl.setVisible(show_dyn)
            self._ov_dyn_min.setVisible(show_dyn)
            self._ov_dyn_min_hint.setVisible(show_dyn)
            self._ov_dyn_max_lbl.setVisible(show_dyn)
            self._ov_dyn_max.setVisible(show_dyn)
            self._ov_dyn_max_hint.setVisible(show_dyn)
            # Force ov_speed=1.0 keď je Dynamický
            if not show_speech and abs(self._ov_speed.value() - 1.0) > 0.001:
                self._ov_speed.setValue(1.0)
        self._ov_timeline_mode.currentTextChanged.connect(lambda _: _upd_manual_visible())
        QTimer.singleShot(0, _upd_manual_visible)

        # Expresívna prozódia presunutá do Settings → "Pokročilé OmniVoice nastavenia"
        # (vytvorí sa tam dolu, widget už nie je v hlavnom OmniVoice taby)
        self._ov_expressive = self._tr_chk("ov_expressive", "Expresívna prozódia (čiarky pred 'teda/napríklad/ale')")
        self._ov_expressive.setChecked(bool(getattr(self.settings, "ov_expressive", False)))
        # widget NIE JE pridaný do ov_l — bude pridaný v _page_settings()

        ov_l.setColumnStretch(1, 1)
        self._tts_stack.addWidget(ov_w)

        self._ov_num_step.valueChanged.connect(self._save_chk_state)
        self._ov_guidance.valueChanged.connect(self._save_chk_state)
        self._ov_instruct.currentTextChanged.connect(self._save_chk_state)
        self._ov_speed.valueChanged.connect(self._save_chk_state)
        self._ov_expressive.stateChanged.connect(self._save_chk_state)

        self._cb_voice.currentTextChanged.connect(self._save_chk_state)
        self._cb_voice.currentTextChanged.connect(self._sync_clone_controls)
        self._cb_clone_auto.stateChanged.connect(self._save_chk_state)
        self._cb_clone_auto.stateChanged.connect(self._sync_clone_controls)
        self._cb_clone_start.valueChanged.connect(self._save_chk_state)
        self._cb_clone_dur.valueChanged.connect(self._save_chk_state)
        self._sync_clone_controls()

        _saved_tts = self.settings.def_tts_engine
        if _saved_tts == "omnivoice":
            self._rb_omnivoice.setChecked(True)
        else:
            self._rb_chatterbox.setChecked(True)

        ttsl.addWidget(self._tts_stack)
        def _upd_tts_stack():
            if self._rb_omnivoice.isChecked():
                self._tts_stack.setCurrentIndex(1)
            else:
                self._tts_stack.setCurrentIndex(0)
        self._rb_chatterbox.toggled.connect(lambda _: _upd_tts_stack())
        self._rb_omnivoice.toggled.connect(lambda _: _upd_tts_stack())
        self._rb_chatterbox.toggled.connect(self._save_chk_state)
        self._rb_omnivoice.toggled.connect(self._save_chk_state)
        # Engine-specific Možnosti visibility update
        self._rb_chatterbox.toggled.connect(lambda _: self._update_engine_options_visibility())
        self._rb_omnivoice.toggled.connect(lambda _: self._update_engine_options_visibility())
        _upd_tts_stack()

        # Auto-detect speakers tlačidlo — small, kompakt, zarovnané vľavo
        self._auto_detect_btn = self._tr_btn("btn_auto_detect", "🔍  Detekuj hovoriaceho")
        self._auto_detect_btn.setFixedHeight(26)
        self._auto_detect_btn.setFixedWidth(220)
        self._auto_detect_btn.setStyleSheet(_pill(r=6))
        self._auto_detect_btn.clicked.connect(self._on_auto_detect_speaker)
        self._tr_tip(self._auto_detect_btn, "tip_auto_detect",
            "Spustí pitch analysis na prvom videu v batch zozname.\n"
            "Auto-nastaví voice design (slovenský muž/žena) podľa detekovaného pohlavia.\n"
            "Trvá ~10s pre 8min video (cached demucs vocals + librosa pyin).")
        # Wrap v hbox aby bolo zarovnané vľavo (s stretch napravo)
        from PyQt6.QtWidgets import QHBoxLayout as _QHB
        _detect_row = _QHB()
        _detect_row.addWidget(self._auto_detect_btn)
        _detect_row.addStretch(1)
        ttsl.addLayout(_detect_row)

        self._tts_section = CollapsibleSection(T.get("grp_tts","TTS Engine"), grp_tts, collapsed=False)
        lay.addWidget(self._tts_section)

        # ── Options ──────────────────────────────────────────────────────────
        grp_opt = self._tr_grp("grp_options","Možnosti")
        optl = QGridLayout(grp_opt); optl.setSpacing(6)
        # Public verzia: 7 aktívnych checkboxov v 3-stĺpcovej mriežke.
        # Odstránené (mŕtve / legacy): Lipsync, Kontextový preklad, Auto speech gain,
        # Fill slot, Level 5-13, Titulky burn-in, Emócia (LLM).
        opts = [
            ("chk_keep_music","Zachovať hudbu","_chk_keep_music",0,0),
            ("chk_refine","Opraviť gramatiku (2× LLM)","_chk_refine",0,1),
            ("chk_voice_eq","Voice EQ","_chk_voice_eq",0,2),
            ("chk_multi_voice","Multi-voice (Chatterbox)","_chk_multi_voice",1,0),
            ("chk_srt_align","WhisperX SRT alignment","_chk_srt_align",1,1),
            ("chk_mini_level11","Mini Level11 pre-TTS rewrite","_chk_mini_level11",1,2),
            ("chk_mixed_technical_terms","Mixed SK+EN technical terms","_chk_mixed_technical_terms",2,0),
            ("chk_stop_after_translate","Iba translate (audio si vyrobíš sám)","_chk_stop_after_translate",2,1),
        ]
        chk_defaults = {
            "_chk_keep_music": self.settings.chk_keep_music,
            "_chk_refine": self.settings.chk_refine,
            "_chk_voice_eq": self.settings.chk_voice_eq,
            "_chk_multi_voice": self.settings.chk_multi_voice,
            "_chk_srt_align": self.settings.chk_srt_align,
            "_chk_mini_level11": self.settings.chk_mini_level11,
            "_chk_mixed_technical_terms": self.settings.chk_mixed_technical_terms,
            "_chk_stop_after_translate": False,  # default OFF — keď je ON, pipeline preskočí TTS
        }
        for key,fb,attr,r,c in opts:
            cb = self._tr_chk(key, fb)
            cb.setChecked(chk_defaults.get(attr, False))
            cb.stateChanged.connect(self._save_chk_state)
            setattr(self,attr,cb); optl.addWidget(cb,r,c)

        # Engine-specific options visibility
        # OmniVoice path: keep_music + voice_eq + srt_align + stop_after_translate
        #   (multi_voice je IBA Chatterbox, mini_level11 + mixed_terms platí pre obe)
        # Chatterbox path: všetky checkboxy okrem stop_after_translate (ten je iba pre OmniVoice)
        # Engines exclusive: multi_voice (CB only), stop_after_translate (OV only orchestrate)
        self._opt_omnivoice_only = ["_chk_stop_after_translate"]
        self._opt_chatterbox_only = ["_chk_multi_voice"]
        # Refine = 2× LLM gramatika — má zmysel len pre Chatterbox path (OmniVoice ide cez standalone)
        self._opt_chatterbox_only.append("_chk_refine")

        # Vocal separator dropdown (next to "Zachovať hudbu")
        from PyQt6.QtWidgets import QComboBox as _QCB
        self._vocal_sep = _QCB()
        self._vocal_sep.addItems(["demucs", "htdemucs_ft", "BS-Roformer"])
        _cur_vsep = getattr(self.settings, "vocal_separator", "demucs")
        # "mdx_net" is legacy label — map to display name
        if _cur_vsep == "mdx_net": _cur_vsep = "BS-Roformer"
        _idx = self._vocal_sep.findText(_cur_vsep)
        if _idx >= 0: self._vocal_sep.setCurrentIndex(_idx)
        self._tr_tip(self._vocal_sep, "tip_vocal_sep", "Vocal separation backend:\ndemucs – rýchly\nhtdemucs_ft – lepšia kvalita\nBS-Roformer – najlepší (SDR 12.9)")
        self._vocal_sep.setFixedWidth(110)
        self._vocal_sep.currentIndexChanged.connect(self._save_chk_state)
        optl.addWidget(self._vocal_sep, 0, 3)

        self._multi_voice_n_lbl = self._tr_lbl("lbl_voices_count","Počet hlasov:")
        self._multi_voice_n = QSpinBox()
        self._multi_voice_n.setRange(2, 6)
        self._multi_voice_n.setValue(max(2, int(getattr(self.settings, "multi_voice_n", 2) or 2)))
        self._multi_voice_n.setFixedHeight(26)
        self._multi_voice_n.valueChanged.connect(self._save_chk_state)
        self._multi_voice_n_lbl.setVisible(self._chk_multi_voice.isChecked())
        self._multi_voice_n.setVisible(self._chk_multi_voice.isChecked())
        optl.addWidget(self._multi_voice_n_lbl, 6, 1)
        optl.addWidget(self._multi_voice_n, 6, 2)
        self._chk_multi_voice.stateChanged.connect(
            lambda *_: self._update_engine_options_visibility()
        )
        # Initial visibility based on current engine
        QTimer.singleShot(0, self._update_engine_options_visibility)
        self._chk_multi_voice.stateChanged.connect(
            lambda *_: self._multi_voice_n.setVisible(self._chk_multi_voice.isChecked())
        )
        self._chk_multi_voice.stateChanged.connect(self._sync_clone_controls)
        self._cb_emo_clone.setChecked(self.settings.cb_emo_clone)
        self._cb_emo_clone.stateChanged.connect(self._save_chk_state)
        optl.addWidget(self._cb_emo_clone,2,1)
        # Emócia (LLM) — odstránené z public verzie (legacy, prekryté Mini Level11)
        # EQ profil — presunutý vedľa Voice EQ na rovnakom riadku
        self._eq_profile_lbl = self._tr_lbl("lbl_eq_profile","EQ profil:")
        optl.addWidget(self._eq_profile_lbl,2,2)
        self._eq_profile = QComboBox(); self._eq_profile.addItems([""] + list(EQ_BUILTIN_PRESETS.keys()))
        self._eq_profile.setCurrentText(self.settings.eq_profile)
        self._eq_profile.currentTextChanged.connect(self._save_chk_state)
        self._eq_profile.setFixedHeight(24); optl.addWidget(self._eq_profile,3,2)
        self._sync_clone_controls()
        self._opt_section = CollapsibleSection(T.get("grp_options","Možnosti"), grp_opt, collapsed=True)
        self._tr_widgets.append((self._opt_section, "grp_options", "Možnosti"))
        lay.addWidget(self._opt_section)

        # ── Action buttons ────────────────────────────────────────────────────
        self._start_btn = self._tr_btn("btn_start","▶  Spustiť preklad")
        self._start_btn.setFixedHeight(42)
        self._start_btn.setFont(QFont("DejaVu Sans", 13, QFont.Weight.Bold))
        self._start_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._start_btn.setStyleSheet(_pill(r=10)); self._start_btn.clicked.connect(self._on_start)
        lay.addWidget(self._start_btn)
        btn_row2 = QHBoxLayout(); btn_row2.setSpacing(10)
        self._stop_btn = self._tr_btn("btn_stop","■  Zastaviť"); self._stop_btn.setFixedHeight(24)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(_pill(bg="#3d1a1a",hover="#5a2020",pressed="#7a2020",text=RED,r=8))
        self._stop_btn.clicked.connect(self._on_stop)
        ckpt_exists = _load_checkpoint() is not None
        self._resume_btn = self._tr_btn("btn_resume","⟳  Pokračovať"); self._resume_btn.setFixedHeight(24)
        self._resume_btn.setStyleSheet(_pill(bg="#1a3a1a",hover="#2a5a2a",pressed="#1a3a1a",text=ACCENTL,r=8))
        self._resume_btn.setEnabled(ckpt_exists)
        self._tr_tip(self._resume_btn, "tip_resume", "Načítať auto-checkpoint a pokračovať kde sa skončilo")
        self._resume_btn.clicked.connect(self._on_resume)
        btn_row2.addWidget(self._resume_btn); btn_row2.addWidget(self._stop_btn,1)
        lay.addLayout(btn_row2)

        # ── Log ───────────────────────────────────────────────────────────────
        lay.addWidget(self._sect("LOG"))
        self._log = QTextEdit(); self._log.setReadOnly(True); self._log.setMinimumHeight(130)
        self._log.document().setMaximumBlockCount(2000)
        lay.addWidget(self._log)

        # ── Segment dual panel ────────────────────────────────────────────────
        seg_w = QWidget()
        seg_l = QVBoxLayout(seg_w); seg_l.setContentsMargins(0,8,0,0); seg_l.setSpacing(6)
        seg_top = QHBoxLayout()
        seg_top.addWidget(self._tr_lbl("lbl_original","Originál"))
        seg_top.addWidget(self._tr_lbl("lbl_translation","Preklad"))
        seg_top.addStretch()
        btn_load_seg = self._tr_btn("btn_load_json","Načítať JSON…"); btn_load_seg.setFixedHeight(26)
        btn_load_seg.setStyleSheet(_pill_sm()); btn_load_seg.clicked.connect(self._load_segments_json)
        seg_top.addWidget(btn_load_seg)
        seg_l.addLayout(seg_top)
        seg_split = QSplitter(Qt.Orientation.Horizontal)
        self._seg_src = QTextEdit(); self._seg_src.setReadOnly(True); self._seg_src.setMinimumHeight(160)
        self._seg_src.setPlaceholderText(T.get("ph_seg_src","Originálny text sa zobrazí po načítaní JSON..."))
        self._seg_tgt = QTextEdit(); self._seg_tgt.setMinimumHeight(160)
        self._seg_tgt.setPlaceholderText(T.get("ph_seg_tgt","Preložený text sa zobrazí po načítaní JSON..."))
        seg_split.addWidget(self._seg_src); seg_split.addWidget(self._seg_tgt)
        seg_split.setSizes([1, 1])
        seg_l.addWidget(seg_split)
        self._seg_section = CollapsibleSection(T.get("coll_segments","Segmenty (originál · preklad)"), seg_w, collapsed=True)
        self._tr_widgets.append((self._seg_section, "coll_segments", "Segmenty (originál · preklad)"))
        lay.addWidget(self._seg_section)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        scroll.setWidget(inner)
        return scroll

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Stiahnutie videa
    # ══════════════════════════════════════════════════════════════════════════
    def _page_youtube(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(16,14,16,14); lay.setSpacing(10)
        lay.addWidget(self._tr_reg(self._sect(T.get("sect_download","STIAHNUTIE VIDEA")), "sect_download","STIAHNUTIE VIDEA"))
        hint = self._tr_reg(QLabel(T.get("lbl_download_hint","Podporované: YouTube, Vimeo, TikTok, Twitter/X, Facebook, Twitch, Dailymotion, PornHub a 1000+ ďalších (yt-dlp)")), "lbl_download_hint","Podporované: YouTube, Vimeo, TikTok, Twitter/X, Facebook, Twitch, Dailymotion, PornHub a 1000+ ďalších (yt-dlp)")
        hint.setStyleSheet("color:#7a9cbf; font-size:12px;"); lay.addWidget(hint)

        grp = self._tr_grp("grp_download_url","URL a nastavenia")
        gl = QGridLayout(grp); gl.setSpacing(5)
        gl.addWidget(self._tr_lbl("lbl_url","URL:"),0,0)
        self._yt_url = QLineEdit(); self._yt_url.setPlaceholderText("https://...")
        self._yt_url.setFixedHeight(24); gl.addWidget(self._yt_url,0,1,1,3)

        gl.addWidget(self._tr_lbl("lbl_format","Formát:"),1,0)
        self._yt_fmt = QComboBox(); self._yt_fmt.addItems([T.get("fmt_video","Video (MP4)"),T.get("fmt_audio","Iba audio (MP3)")])
        self._yt_fmt.setFixedHeight(26); gl.addWidget(self._yt_fmt,1,1)

        gl.addWidget(self._tr_lbl("lbl_quality","Max kvalita:"),1,2)
        self._yt_quality = QComboBox(); self._yt_quality.addItems(["2160","1440","1080","720","480","360"])
        self._yt_quality.setCurrentText("1080"); self._yt_quality.setFixedHeight(26); gl.addWidget(self._yt_quality,1,3)

        gl.addWidget(self._tr_lbl("lbl_out_folder","Výstupný priečinok:"),2,0)
        self._yt_outdir = QLineEdit(); self._yt_outdir.setText(str(Path.home() / "Stiahnuté"))
        self._yt_outdir.setFixedHeight(26); gl.addWidget(self._yt_outdir,2,1,1,2)
        btn_browse = QPushButton("…"); btn_browse.setFixedSize(36,32)
        btn_browse.clicked.connect(lambda: self._yt_outdir.setText(d) if (d := QFileDialog.getExistingDirectory(self,"Výstupný priečinok",self._yt_outdir.text())) else None)
        gl.addWidget(btn_browse,2,3)
        lay.addWidget(grp)

        grp2 = self._tr_grp("grp_download_opts","Možnosti")
        ol = QHBoxLayout(grp2); ol.setSpacing(16)
        self._yt_playlist = self._tr_chk("opt_playlist","Stiahnuť celý playlist"); ol.addWidget(self._yt_playlist)
        ol.addWidget(self._tr_lbl("lbl_cookies","Cookies z prehliadača:"))
        self._yt_cookies = QComboBox(); self._yt_cookies.addItems([T.get("yt_cookies_none","(žiadne)"),"firefox","chrome","chromium","brave","edge"])
        self._yt_cookies.setFixedHeight(26); ol.addWidget(self._yt_cookies)
        cookies_hint = self._tr_reg(QLabel(T.get("lbl_cookies_hint","(pre prihlásené stránky)")), "lbl_cookies_hint","(pre prihlásené stránky)"); cookies_hint.setStyleSheet("color:#7a9cbf; font-size:12px;"); ol.addWidget(cookies_hint)
        ol.addStretch()
        lay.addWidget(grp2)

        self._yt_start = self._tr_btn("btn_download","▶  Stiahnuť")
        self._yt_start.setFixedHeight(42)
        self._yt_start.setFont(QFont("DejaVu Sans", 13, QFont.Weight.Bold))
        self._yt_start.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._yt_start.setStyleSheet(_pill(r=10)); self._yt_start.clicked.connect(self._on_yt_start)
        lay.addWidget(self._yt_start)
        yt_btn_row = QHBoxLayout()
        self._yt_stop = self._tr_btn("btn_stop","■  Zastaviť"); self._yt_stop.setFixedHeight(24)
        self._yt_stop.setEnabled(False)
        self._yt_stop.setStyleSheet(_pill(bg="#3d1a1a",hover="#5a2020",pressed="#7a2020",text=RED,r=8))
        self._yt_stop.clicked.connect(self._on_stop)
        yt_btn_row.addWidget(self._yt_stop,1)
        lay.addLayout(yt_btn_row)
        lay.addWidget(self._sect("LOG"))
        self._yt_log = QTextEdit(); self._yt_log.setReadOnly(True); self._yt_log.setMinimumHeight(200)
        lay.addWidget(self._yt_log)
        lay.addStretch()

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        scroll.setWidget(inner)
        return scroll

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — API keys
    # ══════════════════════════════════════════════════════════════════════════
    def _page_api(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(16,14,16,14); lay.setSpacing(10)
        lay.addWidget(self._tr_reg(self._sect(T.get("sect_api","API KĽÚČE A MODELY")), "sect_api","API KĽÚČE A MODELY"))

        providers = [
            ("OpenAI / ChatGPT", [
                ("API kľúč","_oa_key",self.settings.openai_api_key,True),
                ("Model (preklad)","_oa_translate_model",self.settings.openai_translate_model,False),
                ("Model (TTS)","_oa_tts_model",self.settings.openai_tts_model,False),
                ("TTS hlas","_oa_voice",self.settings.openai_tts_voice,False),
                ("Base URL","_oa_url",self.settings.openai_tts_base_url,False),
            ]),
            ("Grok (xAI)", [
                ("API kľúč","_grok_key",self.settings.grok_api_key,True),
                ("Model (preklad)","_grok_translate_model",self.settings.grok_translate_model,False),
                ("Model (TTS)","_grok_tts_model",self.settings.grok_tts_model,False),
                ("TTS hlas","_grok_voice",self.settings.grok_tts_voice,False),
                ("Base URL","_grok_url",self.settings.grok_tts_base_url,False),
            ]),
            ("Gemini (Google)", [
                ("API kľúč","_gem_key",self.settings.gemini_api_key,True),
                ("Model (preklad)","_gem_translate_model",self.settings.gemini_translate_model,False),
                ("Model (TTS)","_gem_tts_model",self.settings.gemini_tts_model,False),
                ("TTS hlas","_gem_voice",self.settings.gemini_tts_voice,False),
                ("Base URL","_gem_url",self.settings.gemini_tts_base_url,False),
            ]),
            ("Hugging Face", [
                ("API kľúč","_hf_key",self.settings.hf_api_key,True),
            ]),
        ]
        key_placeholders = {
            "_oa_key": "Použi OPENAI_API_KEY alebo API_TRANS_KEY",
            "_grok_key": "Použi GROK_API_KEY, XAI_API_KEY alebo API_TRANS_KEY",
            "_gem_key": "Použi GEMINI_API_KEY, GOOGLE_API_KEY alebo API_TRANS_KEY",
            "_hf_key": "Použi HF_TOKEN alebo HUGGINGFACE_HUB_TOKEN",
        }
        for grp_name, fields in providers:
            grp = QGroupBox(grp_name); gl = QGridLayout(grp); gl.setSpacing(5)
            for row,(lbl_txt,attr,default,secret) in enumerate(fields):
                gl.addWidget(self._lbl(lbl_txt+":"),row,0)
                e = QLineEdit(default); e.setFixedHeight(26)
                if secret: e.setEchoMode(QLineEdit.EchoMode.Password)
                if attr in key_placeholders:
                    e.setPlaceholderText(key_placeholders[attr])
                setattr(self,attr,e); gl.addWidget(e,row,1)
            gl.setColumnStretch(1,1); lay.addWidget(grp)

        save_btn = self._tr_btn("btn_save_api_settings","Uložiť API nastavenia"); save_btn.setFixedHeight(26)
        save_btn.setStyleSheet(_pill()); save_btn.clicked.connect(self._save_api)
        lay.addWidget(save_btn)
        lay.addStretch()

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        scroll.setWidget(inner)
        return scroll

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Tools
    # ══════════════════════════════════════════════════════════════════════════
    def _page_tools(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(16,14,16,14); lay.setSpacing(10)
        lay.addWidget(self._tr_reg(self._sect(T.get("sect_tools","NÁSTROJE")), "sect_tools","NÁSTROJE"))

        # GPU / VRAM cleanup
        grp_gpu = self._tr_grp("grp_gpu","GPU / VRAM")
        gl = QHBoxLayout(grp_gpu); gl.setSpacing(8)
        gl.addWidget(self._tr_lbl("lbl_gpu_info", "Po pádových chybách / orphan procesoch:"))
        b_release_vram = self._tr_btn("btn_release_vram","Uvoľniť VRAM (Ollama + LM Studio + orphan procesy)")
        b_release_vram.setFixedHeight(28); b_release_vram.setStyleSheet(_pill())
        b_release_vram.clicked.connect(self._on_release_vram_clicked)
        gl.addWidget(b_release_vram, 1)
        lay.addWidget(grp_gpu)

        # Modely manager
        grp_models = self._tr_grp("grp_models","Modely")
        ml = QHBoxLayout(grp_models); ml.setSpacing(8)
        ml.addWidget(self._tr_lbl("lbl_models_info","Stiahnuť/aktualizovať modely z HuggingFace:"))
        b_models = self._tr_btn("btn_models_manage","Spravovať modely")
        b_models.setFixedHeight(28); b_models.setStyleSheet(_pill())
        b_models.clicked.connect(self._on_manage_models_clicked)
        ml.addWidget(b_models, 1)
        lay.addWidget(grp_models)

        # FPS conversion
        grp_fps = self._tr_grp("grp_fps_conv","Konverzia FPS na 25fps")
        fl = QGridLayout(grp_fps); fl.setSpacing(5)
        fl.addWidget(self._tr_lbl("lbl_input","Vstup:"),0,0)
        self._fps_in = QLineEdit(); self._fps_in.setFixedHeight(26)
        b1 = QPushButton("…"); b1.setFixedSize(32,32); b1.setStyleSheet(_pill_sm())
        b1.clicked.connect(lambda: self._fps_in.setText(QFileDialog.getOpenFileName(self,"Video","","Video (*.mp4 *.mkv *.avi *.mov);;All (*)")[0] or self._fps_in.text()))
        fl.addWidget(self._fps_in,0,1); fl.addWidget(b1,0,2)
        fl.addWidget(self._tr_lbl("lbl_output","Výstup:"),1,0)
        self._fps_out = QLineEdit(); self._fps_out.setPlaceholderText("(auto: _25fps.mp4)"); self._fps_out.setFixedHeight(26)
        b2 = QPushButton("…"); b2.setFixedSize(32,32); b2.setStyleSheet(_pill_sm())
        b2.clicked.connect(lambda: self._fps_out.setText(QFileDialog.getSaveFileName(self,"Výstup","","Video (*.mp4);;All (*)")[0] or self._fps_out.text()))
        fl.addWidget(self._fps_out,1,1); fl.addWidget(b2,1,2)
        b_fps = self._tr_btn("btn_convert_25fps","Konvertovať na 25fps"); b_fps.setFixedHeight(24); b_fps.setStyleSheet(_pill())
        b_fps.clicked.connect(self._tool_fps); fl.addWidget(b_fps,2,0,1,3)
        fl.setColumnStretch(1,1); lay.addWidget(grp_fps)

        # Video split
        grp_split = self._tr_grp("grp_split_video","Rozdeliť video podľa dĺžky")
        sl = QGridLayout(grp_split); sl.setSpacing(5)
        sl.addWidget(self._tr_lbl("lbl_input","Vstup:"),0,0)
        self._split_in = QLineEdit(); self._split_in.setFixedHeight(26)
        b3 = QPushButton("…"); b3.setFixedSize(32,32); b3.setStyleSheet(_pill_sm())
        b3.clicked.connect(lambda: self._split_in.setText(QFileDialog.getOpenFileName(self,"Video","","Video (*.mp4 *.mkv *.avi *.mov);;All (*)")[0] or self._split_in.text()))
        sl.addWidget(self._split_in,0,1); sl.addWidget(b3,0,2)
        sl.addWidget(self._tr_lbl("lbl_part_length_min","Dĺžka časti (min):"),1,0)
        self._split_mins = QSpinBox(); self._split_mins.setRange(1,120); self._split_mins.setValue(10)
        self._split_mins.setFixedHeight(26); sl.addWidget(self._split_mins,1,1)
        b_split = self._tr_btn("btn_split_video","Rozdeliť video"); b_split.setFixedHeight(24); b_split.setStyleSheet(_pill())
        b_split.clicked.connect(self._tool_split); sl.addWidget(b_split,2,0,1,3)
        sl.setColumnStretch(1,1); lay.addWidget(grp_split)

        # Video join
        grp_join = self._tr_grp("grp_join_video","Zlúčiť časti videa")
        jl = QGridLayout(grp_join); jl.setSpacing(5)
        jl.addWidget(self._tr_lbl("lbl_parts_folder","Priečinok s časťami:"),0,0)
        self._join_dir = QLineEdit(); self._join_dir.setFixedHeight(26)
        b4 = QPushButton("…"); b4.setFixedSize(32,32); b4.setStyleSheet(_pill_sm())
        b4.clicked.connect(lambda: self._join_dir.setText(QFileDialog.getExistingDirectory(self,"Priečinok") or self._join_dir.text()))
        jl.addWidget(self._join_dir,0,1); jl.addWidget(b4,0,2)
        jl.addWidget(self._tr_lbl("lbl_output","Výstup:"),1,0)
        self._join_out = QLineEdit(); self._join_out.setPlaceholderText("(auto: joined.mp4)"); self._join_out.setFixedHeight(26)
        jl.addWidget(self._join_out,1,1)
        b_join = self._tr_btn("btn_join_video","Zlúčiť video"); b_join.setFixedHeight(24); b_join.setStyleSheet(_pill())
        b_join.clicked.connect(self._tool_join); jl.addWidget(b_join,2,0,1,3)
        jl.setColumnStretch(1,1); lay.addWidget(grp_join)

        # ONNX export
        grp_onnx = self._tr_grp("grp_onnx","Export ONNX modelu (MuseTalk)")
        ol = QGridLayout(grp_onnx); ol.setSpacing(5)
        ol.addWidget(self._tr_lbl("lbl_export_type","Typ exportu:"),0,0)
        self._onnx_type = QComboBox(); self._onnx_type.addItems(["ONNX","ONNX + TensorRT"])
        self._onnx_type.setFixedHeight(26); ol.addWidget(self._onnx_type,0,1)
        ol.addWidget(self._tr_lbl("lbl_models_label","Modely:"),1,0)
        self._onnx_models = QComboBox(); self._onnx_models.addItems(["All","UNet only","VAE only","BiSeNet only","Whisper only"])
        self._onnx_models.setFixedHeight(26); ol.addWidget(self._onnx_models,1,1)
        self._onnx_fp16 = QCheckBox("FP16"); ol.addWidget(self._onnx_fp16,2,0)
        b_onnx = self._tr_btn("btn_export_onnx","Exportovať ONNX"); b_onnx.setFixedHeight(24); b_onnx.setStyleSheet(_pill())
        b_onnx.clicked.connect(self._tool_onnx); ol.addWidget(b_onnx,3,0,1,2)
        ol.setColumnStretch(1,1); lay.addWidget(grp_onnx)

        # Diarization + emotion
        grp_diar = self._tr_grp("grp_diar","Diarizácia rečníkov + Emócie (Level 0 / 0b)")
        dl = QGridLayout(grp_diar); dl.setSpacing(5)
        dl.addWidget(self._tr_lbl("lbl_video","Video:"), 0, 0)
        self._diar_in = QLineEdit(); self._diar_in.setFixedHeight(26)
        b_diar_in = QPushButton("…"); b_diar_in.setFixedSize(32, 32); b_diar_in.setStyleSheet(_pill_sm())
        b_diar_in.clicked.connect(lambda: self._diar_in.setText(
            QFileDialog.getOpenFileName(self, "Video", "", "Video/Audio (*.mp4 *.mkv *.avi *.mov *.mp3 *.wav);;All (*)")[0]
            or self._diar_in.text()
        ))
        dl.addWidget(self._diar_in, 0, 1); dl.addWidget(b_diar_in, 0, 2)
        dl.addWidget(self._tr_lbl("lbl_src_lang_short","Jazyk zdroja:"), 1, 0)
        self._diar_lang = QLineEdit("en"); self._diar_lang.setFixedHeight(26); self._diar_lang.setMaximumWidth(60)
        dl.addWidget(self._diar_lang, 1, 1)
        dl.addWidget(self._tr_lbl("lbl_min_speakers","Min rečníkov:"), 2, 0)
        self._diar_min_spk = QSpinBox(); self._diar_min_spk.setRange(1, 8); self._diar_min_spk.setValue(1)
        self._diar_min_spk.setFixedHeight(26); dl.addWidget(self._diar_min_spk, 2, 1)
        dl.addWidget(self._tr_lbl("lbl_max_speakers","Max rečníkov:"), 3, 0)
        self._diar_max_spk = QSpinBox(); self._diar_max_spk.setRange(1, 8); self._diar_max_spk.setValue(3)
        self._diar_max_spk.setFixedHeight(26); dl.addWidget(self._diar_max_spk, 3, 1)
        b_diar = self._tr_btn("btn_diar_run","🎙 Diarizuj & Analyzuj emócie"); b_diar.setFixedHeight(28); b_diar.setStyleSheet(_pill())
        b_diar.clicked.connect(self._tool_diarize); dl.addWidget(b_diar, 4, 0, 1, 3)
        dl.setColumnStretch(1, 1); lay.addWidget(grp_diar)

        # Tools log
        lay.addWidget(self._sect("LOG"))
        self._tools_log = QTextEdit(); self._tools_log.setReadOnly(True); self._tools_log.setMinimumHeight(120)
        lay.addWidget(self._tools_log)
        lay.addStretch()

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        scroll.setWidget(inner)
        return scroll

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — EQ
    # ══════════════════════════════════════════════════════════════════════════
    def _page_eq(self) -> QWidget:

        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(16,14,16,14); lay.setSpacing(10)
        lay.addWidget(self._tr_reg(self._sect(T.get("sect_eq","EKVALIZÉR HLASU")), "sect_eq","EKVALIZÉR HLASU"))

        grp_io = self._tr_grp("grp_eq_io","Vstup / Výstup")
        il = QGridLayout(grp_io); il.setSpacing(8)
        il.addWidget(self._tr_lbl("lbl_src_audio","Zdrojový audio súbor:"),0,0)
        self._eq_in = QLineEdit(); self._eq_in.setFixedHeight(26)
        b_eqi = QPushButton("…"); b_eqi.setFixedSize(32,32); b_eqi.setStyleSheet(_pill_sm())
        b_eqi.clicked.connect(lambda: self._eq_in.setText(QFileDialog.getOpenFileName(self,"Audio","","Audio (*.wav *.mp3 *.flac *.ogg);;All (*)")[0] or self._eq_in.text()))
        il.addWidget(self._eq_in,0,1); il.addWidget(b_eqi,0,2)
        il.addWidget(self._tr_lbl("lbl_dst_audio","Výstupný audio súbor:"),1,0)
        self._eq_out = QLineEdit(); self._eq_out.setFixedHeight(26)
        b_eqo = QPushButton("…"); b_eqo.setFixedSize(32,32); b_eqo.setStyleSheet(_pill_sm())
        b_eqo.clicked.connect(lambda: self._eq_out.setText(QFileDialog.getSaveFileName(self,"Výstup","","Audio (*.wav *.mp3);;All (*)")[0] or self._eq_out.text()))
        il.addWidget(self._eq_out,1,1); il.addWidget(b_eqo,1,2)
        il.setColumnStretch(1,1); lay.addWidget(grp_io)

        grp_pre = self._tr_grp("grp_eq_presets","Presets")
        pl = QHBoxLayout(grp_pre); pl.setSpacing(8)
        pl.addWidget(self._tr_lbl("lbl_preset","Preset:"))
        self._eq_preset_cb = QComboBox(); self._eq_preset_cb.addItems(list(EQ_BUILTIN_PRESETS.keys()))
        self._eq_preset_cb.setFixedHeight(26)
        self._eq_preset_cb.currentTextChanged.connect(self._eq_load_preset)
        pl.addWidget(self._eq_preset_cb,1)
        b_apply = self._tr_btn("btn_apply_preset","Použiť preset"); b_apply.setFixedHeight(26); b_apply.setStyleSheet(_pill_sm())
        b_apply.clicked.connect(lambda: self._eq_load_preset(self._eq_preset_cb.currentText()))
        pl.addWidget(b_apply)
        lay.addWidget(grp_pre)

        grp_fil = self._tr_grp("grp_eq_filters","FFmpeg filtre (1 per riadok)")
        fl = QVBoxLayout(grp_fil); fl.setSpacing(6)
        self._eq_filters = QTextEdit(); self._eq_filters.setMinimumHeight(120)
        self._eq_filters.setStyleSheet(f"background:{CARD}; color:{FG}; font-family:'DejaVu Sans Mono'; font-size:12px;")
        self._eq_filters.setPlaceholderText("highpass=f=80\nequalizer=f=350:t=q:w=1.2:g=-2.5\nlowpass=f=11500")
        fl.addWidget(self._eq_filters)
        eq_btn_row = QHBoxLayout(); eq_btn_row.setSpacing(8)
        b_run = self._tr_btn("btn_apply_eq","▶  Aplikovať EQ"); b_run.setFixedHeight(24); b_run.setStyleSheet(_pill())
        b_run.clicked.connect(self._eq_apply)
        b_play_in = self._tr_btn("btn_play_in","▶ Prehrať vstup"); b_play_in.setFixedHeight(24); b_play_in.setStyleSheet(_pill_sm())
        b_play_in.clicked.connect(lambda: self._play_audio(self._eq_in.text()))
        b_play_out = self._tr_btn("btn_play_out","▶ Prehrať výstup"); b_play_out.setFixedHeight(24); b_play_out.setStyleSheet(_pill_sm())
        b_play_out.clicked.connect(lambda: self._play_audio(self._eq_out.text()))
        eq_btn_row.addWidget(b_run); eq_btn_row.addWidget(b_play_in); eq_btn_row.addWidget(b_play_out)
        eq_btn_row.addStretch(); fl.addLayout(eq_btn_row)
        lay.addWidget(grp_fil)
        lay.addWidget(self._sect("LOG"))
        self._eq_log = QTextEdit(); self._eq_log.setReadOnly(True); self._eq_log.setMinimumHeight(100)
        lay.addWidget(self._eq_log)
        lay.addStretch()

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        scroll.setWidget(inner)
        return scroll

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 6 — Settings
    # ══════════════════════════════════════════════════════════════════════════
    def _page_settings(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(16,14,16,14); lay.setSpacing(10)
        lay.addWidget(self._tr_reg(self._sect(T.get("sect_settings","NASTAVENIA")), "sect_settings","NASTAVENIA"))

        # Language switcher
        grp_lang_ui = self._tr_grp("grp_language","Jazyk rozhrania")
        lang_lay = QHBoxLayout(grp_lang_ui); lang_lay.setSpacing(12)
        lang_lay.addWidget(self._tr_lbl("lbl_ui_language","Jazyk rozhrania:"))
        self._lang_cb = QComboBox(); self._lang_cb.setFixedHeight(26)
        _avail = _available_languages()
        for code, name in _avail.items():
            self._lang_cb.addItem(name, userData=code)
        cur_lang = self.settings.ui_language
        for i in range(self._lang_cb.count()):
            if self._lang_cb.itemData(i) == cur_lang:
                self._lang_cb.setCurrentIndex(i); break
        # Live switch — load the JSON and retranslate immediately on dropdown change
        self._lang_cb.currentIndexChanged.connect(self._on_language_changed)
        lang_lay.addWidget(self._lang_cb)
        _hint_lbl = self._tr_reg(QLabel(T.get("lbl_restart_hint","")), "lbl_restart_hint","")
        _hint_lbl.setStyleSheet("color:#7a9cbf; font-size:11px;"); lang_lay.addWidget(_hint_lbl)
        lang_lay.addStretch()
        lay.addWidget(grp_lang_ui)

        # Window size
        grp_win = self._tr_grp("grp_window_size","Veľkosť okna pri spustení")
        wl = QHBoxLayout(grp_win); wl.setSpacing(12)
        wl.addWidget(self._tr_lbl("lbl_win_width","Šírka:"))
        self._win_width = QSpinBox(); self._win_width.setRange(900, 3840)
        self._win_width.setValue(self.settings.window_width); self._win_width.setFixedHeight(26)
        wl.addWidget(self._win_width)
        wl.addWidget(self._tr_lbl("lbl_win_height","Výška:"))
        self._win_height = QSpinBox(); self._win_height.setRange(600, 2160)
        self._win_height.setValue(self.settings.window_height); self._win_height.setFixedHeight(26)
        wl.addWidget(self._win_height)
        btn_cur = self._tr_btn("btn_use_current","Použiť aktuálnu veľkosť"); btn_cur.setFixedHeight(26)
        btn_cur.setStyleSheet(_pill_sm())
        btn_cur.clicked.connect(lambda: (
            self._win_width.setValue(self.width()),
            self._win_height.setValue(self.height()),
        ))
        wl.addWidget(btn_cur)
        wl.addStretch()
        lay.addWidget(grp_win)

        # Paths
        grp_paths = self._tr_grp("grp_paths","Systémové cesty")
        pl = QGridLayout(grp_paths); pl.setSpacing(8)
        pl.addWidget(self._tr_lbl("lbl_python","Python exe:"),0,0)
        self._set_py = QLineEdit(self.settings.py); self._set_py.setFixedHeight(26)
        b_py = QPushButton("…"); b_py.setFixedSize(32,32); b_py.setStyleSheet(_pill_sm())
        b_py.clicked.connect(lambda: self._set_py.setText(QFileDialog.getOpenFileName(self,"Python","","Python (python python3 python.exe);;All (*)")[0] or self._set_py.text()))
        pl.addWidget(self._set_py,0,1); pl.addWidget(b_py,0,2)
        pl.addWidget(self._tr_lbl("lbl_ffmpeg","FFmpeg:"),1,0)
        self._set_ffmpeg = QLineEdit(self.settings.ffmpeg); self._set_ffmpeg.setFixedHeight(26)
        b_ff = QPushButton("…"); b_ff.setFixedSize(32,32); b_ff.setStyleSheet(_pill_sm())
        b_ff.clicked.connect(lambda: self._set_ffmpeg.setText(QFileDialog.getOpenFileName(self,"FFmpeg","","Exe (ffmpeg ffmpeg.exe);;All (*)")[0] or self._set_ffmpeg.text()))
        pl.addWidget(self._set_ffmpeg,1,1); pl.addWidget(b_ff,1,2)
        pl.setColumnStretch(1,1); lay.addWidget(grp_paths)

        # ── Modely (HuggingFace download manager) ─────────────────────────────
        grp_models_hf = self._tr_grp("grp_models_hf", "Modely z HuggingFace")
        mhl = QGridLayout(grp_models_hf); mhl.setSpacing(6)
        self._chk_models_skip = QCheckBox("Pri spustení nezobrazovať dialog modelov, aj keď chýbajú")
        self._chk_models_skip.setChecked(getattr(self.settings, "models_skip_dialog", False))
        self._chk_models_skip.setToolTip(
            "Užitočné keď používaš vlastné modely a nechceš stiahnuť z HuggingFace.\n"
            "Vždy môžeš dialog vyvolať cez Nástroje → Spravovať modely…"
        )
        self._chk_models_skip.stateChanged.connect(self._on_models_skip_toggled)
        mhl.addWidget(self._chk_models_skip, 0, 0, 1, 2)
        b_show_models = QPushButton("Otvoriť dialog modelov teraz")
        b_show_models.setFixedHeight(28); b_show_models.setStyleSheet(_pill())
        b_show_models.clicked.connect(self._on_manage_models_clicked)
        mhl.addWidget(b_show_models, 1, 0)
        mhl.setColumnStretch(1, 1)
        lay.addWidget(grp_models_hf)

        # ── Pokročilé OmniVoice nastavenia ────────────────────────────────────
        grp_ov_adv = self._tr_grp("grp_ov_adv", "Pokročilé OmniVoice nastavenia")
        ovl = QVBoxLayout(grp_ov_adv); ovl.setSpacing(6)
        # _ov_expressive widget bol vytvorený v _page_local() ale NIE pridaný do ov_l —
        # pridáme ho sem do Settings
        if hasattr(self, "_ov_expressive"):
            ovl.addWidget(self._ov_expressive)
            tip = QLabel(
                "Vloží čiarku pred slová <b>teda / napríklad / ale / však / totiž / "
                "pretože / preto</b> aby OmniVoice urobil pri čítaní mikropauzu — "
                "prirodzenejšia prozódia."
            )
            tip.setWordWrap(True)
            tip.setStyleSheet(f"color:{MUTED}; font-size:11px;")
            ovl.addWidget(tip)
        lay.addWidget(grp_ov_adv)

        # ── Modely + roots (paths.py override) ────────────────────────────────
        grp_models = self._tr_grp("grp_paths_models", "Modely a koreňové cesty (paths.py)")
        ml = QGridLayout(grp_models); ml.setSpacing(8)
        ml.addWidget(self._tr_lbl("lbl_miniforge", "Miniforge3 root:"), 0, 0)
        self._path_miniforge = QLineEdit(PATHS.miniforge_root); self._path_miniforge.setFixedHeight(26)
        b_mf = QPushButton("…"); b_mf.setFixedSize(32,32); b_mf.setStyleSheet(_pill_sm())
        b_mf.clicked.connect(lambda: self._path_miniforge.setText(
            QFileDialog.getExistingDirectory(self, "Miniforge3 root", self._path_miniforge.text()) or self._path_miniforge.text()))
        ml.addWidget(self._path_miniforge, 0, 1); ml.addWidget(b_mf, 0, 2)

        ml.addWidget(self._tr_lbl("lbl_models_root", "Modely (HF):"), 1, 0)
        self._path_models = QLineEdit(PATHS.models_root); self._path_models.setFixedHeight(26)
        b_mr = QPushButton("…"); b_mr.setFixedSize(32,32); b_mr.setStyleSheet(_pill_sm())
        b_mr.clicked.connect(lambda: self._path_models.setText(
            QFileDialog.getExistingDirectory(self, "Models root", self._path_models.text()) or self._path_models.text()))
        ml.addWidget(self._path_models, 1, 1); ml.addWidget(b_mr, 1, 2)

        ml.addWidget(self._tr_lbl("lbl_knihy_root", "Knihy (GGUF, datasety):"), 2, 0)
        self._path_knihy = QLineEdit(PATHS.knihy_root); self._path_knihy.setFixedHeight(26)
        b_kh = QPushButton("…"); b_kh.setFixedSize(32,32); b_kh.setStyleSheet(_pill_sm())
        b_kh.clicked.connect(lambda: self._path_knihy.setText(
            QFileDialog.getExistingDirectory(self, "Knihy root", self._path_knihy.text()) or self._path_knihy.text()))
        ml.addWidget(self._path_knihy, 2, 1); ml.addWidget(b_kh, 2, 2)

        ml.addWidget(self._tr_lbl("lbl_ollama_bin", "Ollama bin:"), 3, 0)
        self._path_ollama = QLineEdit(PATHS.ollama_bin); self._path_ollama.setFixedHeight(26)
        b_ol = QPushButton("…"); b_ol.setFixedSize(32,32); b_ol.setStyleSheet(_pill_sm())
        b_ol.clicked.connect(lambda: self._path_ollama.setText(
            QFileDialog.getOpenFileName(self, "Ollama binary", self._path_ollama.text())[0] or self._path_ollama.text()))
        ml.addWidget(self._path_ollama, 3, 1); ml.addWidget(b_ol, 3, 2)

        b_save_paths = self._tr_btn("btn_save_paths","Uložiť cesty"); b_save_paths.setFixedHeight(28); b_save_paths.setStyleSheet(_pill())
        b_save_paths.clicked.connect(self._save_paths_config)
        ml.addWidget(b_save_paths, 4, 0, 1, 3)
        ml.setColumnStretch(1, 1); lay.addWidget(grp_models)

        # ── Vlastné TTS enginy ───────────────────────────────────────────────
        grp_tts = self._tr_grp("grp_custom_tts","Vlastné TTS enginy")
        tl = QGridLayout(grp_tts); tl.setSpacing(8)
        tl.addWidget(self._tr_lbl("lbl_custom_tts","Pridať/upraviť TTS engine — názov + typ + cesta k modelu:"), 0, 0, 1, 4)
        tl.addWidget(self._tr_lbl("lbl_ct_select","Engine:"), 1, 0)
        self._custom_tts_combo = QComboBox()
        self._custom_tts_combo.setEditable(False)
        self._custom_tts_combo.setFixedHeight(26)
        self._refresh_custom_tts_combo()
        self._custom_tts_combo.currentTextChanged.connect(self._on_custom_tts_selected)
        tl.addWidget(self._custom_tts_combo, 1, 1)
        b_ct_add = QPushButton("+"); b_ct_add.setFixedSize(28, 28); b_ct_add.setStyleSheet(_pill_sm())
        self._tr_tip(b_ct_add, "tip_ct_add", "Pridať nový custom TTS engine")
        b_ct_add.clicked.connect(self._on_custom_tts_add)
        tl.addWidget(b_ct_add, 1, 2)
        b_ct_del = QPushButton("−"); b_ct_del.setFixedSize(28, 28); b_ct_del.setStyleSheet(_pill_sm())
        self._tr_tip(b_ct_del, "tip_ct_del", "Odstrániť aktuálne vybraný custom TTS engine")
        b_ct_del.clicked.connect(self._on_custom_tts_del)
        tl.addWidget(b_ct_del, 1, 3)
        tl.addWidget(self._tr_lbl("lbl_ct_type","Typ:"), 2, 0)
        self._custom_tts_type = QComboBox()
        self._custom_tts_type.addItems(["chatterbox", "omnivoice"])
        self._custom_tts_type.setFixedHeight(26)
        self._custom_tts_type.currentTextChanged.connect(self._on_custom_tts_field_edit)
        tl.addWidget(self._custom_tts_type, 2, 1)
        tl.addWidget(self._tr_lbl("lbl_ct_path","Cesta:"), 2, 2)
        self._custom_tts_path = QLineEdit()
        self._custom_tts_path.setFixedHeight(26)
        self._custom_tts_path.editingFinished.connect(self._on_custom_tts_field_edit)
        tl.addWidget(self._custom_tts_path, 2, 3)
        b_ct_browse = QPushButton("…"); b_ct_browse.setFixedSize(28, 28); b_ct_browse.setStyleSheet(_pill_sm())
        b_ct_browse.clicked.connect(self._on_custom_tts_browse)
        tl.addWidget(b_ct_browse, 2, 4)
        tl.setColumnStretch(3, 1); lay.addWidget(grp_tts)

        # ── Správa prekladových modelov ─────────────────────────────────────
        grp_tm = self._tr_grp("grp_trans_models", "Správa prekladových modelov")
        tml = QGridLayout(grp_tm); tml.setSpacing(8)
        tml.addWidget(self._tr_lbl("lbl_tm_help",
            "Pre engine 'translategemma' alebo 'gemma' — pridaj/odstráň/refresh modely\n"
            "v dropdowne hlavného panelu. Zmeny sa hneď prejavia v hlavnom paneli."),
            0, 0, 1, 4)
        b_tm_add = self._tr_btn("btn_tm_add","+ Pridať GGUF model zo súboru"); b_tm_add.setFixedHeight(28)
        b_tm_add.clicked.connect(self._on_tgemma_add)
        tml.addWidget(b_tm_add, 1, 0)
        b_tm_del = self._tr_btn("btn_tm_del","− Odstrániť aktuálny model"); b_tm_del.setFixedHeight(28)
        b_tm_del.clicked.connect(self._on_tgemma_del)
        tml.addWidget(b_tm_del, 1, 1)
        b_tm_refresh = self._tr_btn("btn_tm_refresh","↻ Načítať ollama modely"); b_tm_refresh.setFixedHeight(28)
        self._tr_tip(b_tm_refresh, "tip_tm_refresh", "Pre engine=gemma — auto-discover nainštalované ollama modely")
        b_tm_refresh.clicked.connect(self._on_refresh_ollama_models)
        tml.addWidget(b_tm_refresh, 1, 2)
        tml.setColumnStretch(3, 1); lay.addWidget(grp_tm)

        # ── Profily a predvolené nastavenia ──────────────────────────────────
        grp_prof = self._tr_grp("grp_profiles", "Profily a predvolené nastavenia")
        prl = QGridLayout(grp_prof); prl.setSpacing(8)
        prl.addWidget(self._tr_lbl("lbl_profile", "Profil:"), 0, 0)
        self._profile_combo = QComboBox()
        self._profile_combo.setFixedHeight(26)
        self._profile_combo.setEditable(False)
        self._refresh_profile_list()
        prl.addWidget(self._profile_combo, 0, 1)
        b_load = self._tr_btn("btn_profile_load","Načítať")
        b_load.setFixedHeight(28); self._tr_tip(b_load, "tip_profile_load", "Načíta vybraný profil. Reštart GUI pre úplné aplikovanie.")
        b_load.clicked.connect(self._on_profile_load)
        prl.addWidget(b_load, 0, 2)
        b_save_as = self._tr_btn("btn_profile_save_as","Uložiť ako…")
        b_save_as.setFixedHeight(28); self._tr_tip(b_save_as, "tip_profile_save_as", "Uloží aktuálne nastavenia ako pomenovaný profil.")
        b_save_as.clicked.connect(self._on_profile_save_as)
        prl.addWidget(b_save_as, 0, 3)
        b_del = self._tr_btn("btn_profile_delete","Vymazať")
        b_del.setFixedHeight(28); self._tr_tip(b_del, "tip_profile_delete", "Vymaže vybraný profil zo zoznamu.")
        b_del.clicked.connect(self._on_profile_delete)
        prl.addWidget(b_del, 0, 4)
        # Special: set as Normal default + factory reset
        b_set_normal = self._tr_btn("btn_set_normal_default","Aktuálne nastaviť ako Normal default")
        b_set_normal.setFixedHeight(28)
        self._tr_tip(b_set_normal, "tip_set_normal_default", "Pri prepnutí UI módu na Normal sa automaticky obnovia tieto nastavenia.")
        b_set_normal.clicked.connect(self._on_set_normal_default)
        prl.addWidget(b_set_normal, 1, 0, 1, 3)
        b_factory = self._tr_btn("btn_factory_reset","Obnoviť pôvodné (factory) nastavenia")
        b_factory.setFixedHeight(28)
        self._tr_tip(b_factory, "tip_factory_reset", "Resetuje VŠETKY nastavenia na továrenské. Vyžaduje reštart GUI.")
        b_factory.clicked.connect(self._on_factory_reset)
        prl.addWidget(b_factory, 1, 3, 1, 2)
        prl.setColumnStretch(1, 1); lay.addWidget(grp_prof)

        # Defaults
        grp_def = self._tr_grp("grp_defaults","Predvolené hodnoty")
        dl = QGridLayout(grp_def); dl.setSpacing(5)
        dl.addWidget(self._tr_lbl("lbl_tts_engine","TTS engine (default pri spustení):"),0,0)
        self._set_tts = QComboBox(); self._set_tts.addItems(["omnivoice","chatterbox"])
        self._set_tts.setCurrentText(self.settings.def_tts_engine); self._set_tts.setFixedHeight(26)
        # Auto-save + sync s Local page radio buttons
        self._set_tts.currentTextChanged.connect(self._on_default_tts_changed)
        dl.addWidget(self._set_tts,0,1)
        dl.addWidget(self._tr_lbl("lbl_trans_engine","Prekladový engine:"),0,2)
        self._set_trans = QComboBox(); self._set_trans.addItems(_PUBLIC_ENGINES)
        self._set_trans.setCurrentText(self.settings.def_trans_engine if self.settings.def_trans_engine in _PUBLIC_ENGINES else "lmstudio")
        self._set_trans.setFixedHeight(26)
        dl.addWidget(self._set_trans,0,3)
        # Built-in llama-cpp-server (alternatíva LM Studio, open source, súčasť VTS)
        self._chk_llamacpp = QCheckBox("Použiť built-in llama-cpp-server (namiesto LM Studio)")
        self._chk_llamacpp.setChecked(getattr(self.settings, "use_llamacpp_server", True))
        self._chk_llamacpp.setToolTip(
            "Open-source náhrada LM Studio. Auto-spustí sa pri preklade s engine=lmstudio.\n"
            "Beží na rovnakom porte 1234 s OpenAI-compatible API. Bez 3rd party závislostí."
        )
        self._chk_llamacpp.stateChanged.connect(
            lambda *_: (setattr(self.settings, "use_llamacpp_server", self._chk_llamacpp.isChecked()),
                        self.settings.save())
        )
        dl.addWidget(self._chk_llamacpp, 3, 0, 1, 4)

        # Lokálny GGUF model — presunutý sem z hlavnej stránky (clean UX)
        dl.addWidget(self._tr_lbl("lbl_gemma_model","Lokálny GGUF model:"),1,4)
        self._set_gemma_model = QComboBox()
        self._set_gemma_model.setEditable(True)
        # Mirror items z _madlad_model (ktorý je primary widget so zoznamom)
        if hasattr(self, "_madlad_model"):
            items = [self._madlad_model.itemText(i) for i in range(self._madlad_model.count())]
            self._set_gemma_model.addItems(items)
            self._set_gemma_model.setCurrentText(self._madlad_model.currentText())
        # Two-way sync so _madlad_model
        self._set_gemma_model.currentTextChanged.connect(self._sync_gemma_model_from_settings)
        self._set_gemma_model.setFixedHeight(26)
        dl.addWidget(self._set_gemma_model,1,5)
        # Refine model — pre 2-LLM pipeline (Opraviť gramatiku 2× LLM)
        dl.addWidget(self._tr_lbl("lbl_refine_engine","Refine LLM (Opraviť gramatiku):"),2,4)
        self._set_refine_engine = QComboBox()
        # 27B multitask v2 + sk-corrector-v3 odstránené (broken refine — echoed EN, halucinácie).
        # Gemma 3 12B v1 (Ollama fine-tune) overený ako stabilný refine pre tech texty.
        self._set_refine_engine.addItems([
            "Gemma 3 12B v1 (Ollama, fine-tune, default)",
            "Gemma 3 12B Q8 (llama.cpp)",
            "Gemma 4 26B (LM Studio, primary reuse)",
            "Custom Ollama model",
        ])
        _saved_refine = getattr(self.settings, "refine_engine_choice", "Gemma 3 12B Q8 (llama.cpp, default)")
        if _saved_refine in [self._set_refine_engine.itemText(i) for i in range(self._set_refine_engine.count())]:
            self._set_refine_engine.setCurrentText(_saved_refine)
        self._set_refine_engine.setFixedHeight(26)
        self._set_refine_engine.currentTextChanged.connect(self._on_refine_engine_changed)
        dl.addWidget(self._set_refine_engine,2,5)
        # Whisper model — editovateľný + perzistovaný
        dl.addWidget(self._tr_lbl("lbl_whisper_model","Whisper model:"),0,4)
        self._set_whisper = QComboBox()
        self._set_whisper.setEditable(True)
        self._set_whisper.addItems(list(self.settings.whisper_models))
        self._set_whisper.setCurrentText(self.settings.whisper_model)
        self._set_whisper.setFixedHeight(26)
        self._tr_tip(self._set_whisper, "tip_whisper_model",
            "faster-whisper model: large-v3-turbo (rýchly), large-v3 (najlepšia kvalita),\n"
            "medium/small/base/tiny (rýchlejšie, slabšia kvalita).\n"
            "Editovateľné — môžeš zadať vlastný názov modelu.")
        # Auto-save pri zmene
        self._set_whisper.lineEdit().editingFinished.connect(self._on_whisper_model_edited)
        self._set_whisper.currentTextChanged.connect(lambda *_: setattr(self.settings, "whisper_model", self._set_whisper.currentText().strip()))
        dl.addWidget(self._set_whisper,0,5)
        dl.addWidget(self._tr_lbl("lbl_speech_gain","Speech gain:"),1,0)
        self._set_speech_gain = QLineEdit(self.settings.def_speech_gain); self._set_speech_gain.setFixedHeight(26); dl.addWidget(self._set_speech_gain,1,1)
        dl.addWidget(self._tr_lbl("lbl_music_vol","Music volume:"),1,2)
        self._set_music_vol = QLineEdit(self.settings.def_music_volume); self._set_music_vol.setFixedHeight(26); dl.addWidget(self._set_music_vol,1,3)
        dl.addWidget(self._tr_lbl("lbl_tempo","Base tempo:"),2,0)
        self._set_tempo = QLineEdit(self.settings.def_base_tempo); self._set_tempo.setFixedHeight(26); dl.addWidget(self._set_tempo,2,1)
        dl.addWidget(self._tr_lbl("lbl_stretch_min","Stretch min:"),2,2)
        self._set_str_min = QLineEdit(self.settings.def_stretch_min); self._set_str_min.setFixedHeight(26); dl.addWidget(self._set_str_min,2,3)
        dl.addWidget(self._tr_lbl("lbl_stretch_max","Stretch max:"),3,0)
        self._set_str_max = QLineEdit(self.settings.def_stretch_max); self._set_str_max.setFixedHeight(26); dl.addWidget(self._set_str_max,3,1)
        dl.addWidget(self._tr_lbl("lbl_pause_gap","Max pause gap (s):"),3,2)
        self._set_pause = QLineEdit(self.settings.def_pause_gap_s); self._set_pause.setFixedHeight(26); dl.addWidget(self._set_pause,3,3)
        dl.addWidget(self._tr_lbl("lbl_denoise","Odšumenie hlasu:"),4,0)
        self._chk_denoise = self._tr_chk("chk_denoise","Zapnúť"); self._chk_denoise.setChecked(self.settings.chk_denoise)
        dl.addWidget(self._chk_denoise,4,1)
        self._denoise_preset_lbl = self._tr_lbl("lbl_denoise_preset","Denoise preset:")
        dl.addWidget(self._denoise_preset_lbl,4,2)
        self._denoise_preset = QComboBox(); self._denoise_preset.addItems(["mild","strong"])
        self._denoise_preset.setCurrentText(self.settings.denoise_preset)
        self._denoise_preset.setFixedHeight(26); dl.addWidget(self._denoise_preset,4,3)
        self._chk_denoise.stateChanged.connect(self._apply_settings_visibility)
        dl.addWidget(self._tr_lbl("lbl_pitch","Pitch hlasu (st):"),5,0)
        self._pitch_shift = QDoubleSpinBox(); self._pitch_shift.setRange(-12,12); self._pitch_shift.setSingleStep(0.5)
        self._pitch_shift.setFixedHeight(26); dl.addWidget(self._pitch_shift,5,1)
        dl.addWidget(self._tr_lbl("lbl_mux_res","Rozlíšenie výstupu:"),5,2)
        self._mux_resolution = QComboBox()
        _orig_label = T.get("mux_res_original","(pôvodné)")
        self._mux_resolution.addItems([_orig_label,"3840x2160","2560x1440","1920x1080","1280x720","854x480","640x360"])
        _cur_res = self.settings.mux_resolution or _orig_label
        self._mux_resolution.setCurrentText(_cur_res if _cur_res in [_orig_label,"3840x2160","2560x1440","1920x1080","1280x720","854x480","640x360"] else _orig_label)
        self._mux_resolution.setFixedHeight(26); dl.addWidget(self._mux_resolution,5,3)
        dl.setColumnStretch(1,1); dl.setColumnStretch(3,1); lay.addWidget(grp_def)

        grp_adapt = self._tr_grp("grp_adapt","Text adaptation")
        al = QGridLayout(grp_adapt); al.setSpacing(5)
        self._text_adapt_enabled = self._tr_chk("chk_text_adapt","Zapnúť text adaptation (timing-aware rewrite)")
        self._text_adapt_enabled.setChecked(bool(self.settings.text_adapt_enabled))
        self._tr_tip(self._text_adapt_enabled, "tip_text_adapt",
            "Ak TTS odhadovaná dĺžka preteká slot, pokúsi sa vetu skrátiť "
            "(concise → dub-friendly rewrite). Gemma 12B. avg ratio 1.76×→0.77×.")
        self._text_adapt_enabled.stateChanged.connect(self._save_chk_state)
        al.addWidget(self._text_adapt_enabled, 0, 0, 1, 4)
        self._adapt_provider_lbl = self._tr_lbl("lbl_adapt_provider","Backend:")
        al.addWidget(self._adapt_provider_lbl,1,0)
        self._adapt_provider = QComboBox()
        self._adapt_provider.addItems(["llama_cpp","ollama"])
        self._adapt_provider.setCurrentText(self.settings.adapt_provider)
        self._adapt_provider.setFixedHeight(26)
        al.addWidget(self._adapt_provider,1,1)
        self._adapt_ollama_model_lbl = self._tr_lbl("lbl_adapt_model","Ollama model:")
        al.addWidget(self._adapt_ollama_model_lbl,1,2)
        self._adapt_ollama_model = QLineEdit(self.settings.adapt_ollama_model)
        self._adapt_ollama_model.setFixedHeight(26)
        al.addWidget(self._adapt_ollama_model,1,3)
        self._adapt_ollama_url_lbl = self._tr_lbl("lbl_adapt_url","Ollama URL:")
        al.addWidget(self._adapt_ollama_url_lbl,2,0)
        self._adapt_ollama_url = QLineEdit(self.settings.adapt_ollama_url)
        self._adapt_ollama_url.setFixedHeight(26)
        al.addWidget(self._adapt_ollama_url,2,1,1,3)
        self._text_adapt_enabled.stateChanged.connect(self._apply_settings_visibility)
        self._adapt_provider.currentTextChanged.connect(self._apply_settings_visibility)
        al.setColumnStretch(1,1); al.setColumnStretch(3,1)
        lay.addWidget(grp_adapt)

        # VAD
        grp_vad = self._tr_grp("grp_vad","VAD segmentácia")
        vl = QGridLayout(grp_vad); vl.setSpacing(5)
        vl.addWidget(self._tr_lbl("lbl_vad_speech","Min speech (ms):"),0,0)
        self._set_vad_speech = QLineEdit(self.settings.def_vad_min_speech_ms); self._set_vad_speech.setFixedHeight(26); vl.addWidget(self._set_vad_speech,0,1)
        vl.addWidget(self._tr_lbl("lbl_vad_silence","Min silence (ms):"),0,2)
        self._set_vad_silence = QLineEdit(self.settings.def_vad_min_silence_ms); self._set_vad_silence.setFixedHeight(26); vl.addWidget(self._set_vad_silence,0,3)
        vl.addWidget(self._tr_lbl("lbl_vad_max","Max segment (s):"),1,0)
        self._set_vad_max = QLineEdit(self.settings.def_vad_max_speech_s); self._set_vad_max.setFixedHeight(26); vl.addWidget(self._set_vad_max,1,1)
        vl.setColumnStretch(1,1); vl.setColumnStretch(3,1); lay.addWidget(grp_vad)

        # Chatterbox TTS params
        grp_cb = self._tr_grp("grp_cb_params","Chatterbox TTS parametre")
        cbl = QGridLayout(grp_cb); cbl.setSpacing(8)
        cbl.addWidget(self._tr_lbl("lbl_expresivity","Expresivita:"),0,0)
        self._cb_exag = QDoubleSpinBox(); self._cb_exag.setRange(0,2); self._cb_exag.setSingleStep(0.05)
        self._cb_exag.setValue(self.settings.cb_exaggeration); self._cb_exag.setFixedHeight(26)
        self._cb_exag.valueChanged.connect(self._save_chk_state)
        cbl.addWidget(self._cb_exag,0,1)
        cbl.addWidget(self._tr_lbl("lbl_cfg_weight","CFG weight:"),1,0)
        self._cb_cfg = QDoubleSpinBox(); self._cb_cfg.setRange(0,5); self._cb_cfg.setSingleStep(0.1)
        self._cb_cfg.setValue(self.settings.cb_cfg); self._cb_cfg.setFixedHeight(26)
        self._cb_cfg.valueChanged.connect(self._save_chk_state)
        cbl.addWidget(self._cb_cfg,1,1)
        cbl.addWidget(self._tr_lbl("lbl_temperature","Temperature:"),2,0)
        self._cb_temp = QDoubleSpinBox(); self._cb_temp.setRange(0,2); self._cb_temp.setSingleStep(0.05)
        self._cb_temp.setValue(self.settings.cb_temp); self._cb_temp.setFixedHeight(26)
        self._cb_temp.valueChanged.connect(self._save_chk_state)
        cbl.addWidget(self._cb_temp,2,1)
        cbl.setColumnStretch(1,1); lay.addWidget(grp_cb)

        save_btn = self._tr_btn("btn_save_settings","Uložiť nastavenia"); save_btn.setFixedHeight(26)
        save_btn.setStyleSheet(_pill()); save_btn.clicked.connect(self._save_settings)
        lay.addWidget(save_btn)
        lay.addStretch()

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        scroll.setWidget(inner)
        return scroll

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _lbl(self, txt: str) -> QLabel:
        l = QLabel(txt); l.setStyleSheet(f"color:{FG2}; font-size:12px;"); return l

    def _sect(self, txt: str) -> QLabel:
        l = QLabel(txt); l.setFont(QFont("DejaVu Sans",9,QFont.Weight.Bold))
        l.setStyleSheet(f"color:{FG2}; letter-spacing:1px;"); return l

    def _get_voices(self) -> List[str]:
        return (
            [_voice_disp("(predvolený)"), _voice_disp("🎬 Cloning z videa")]
            + sorted(p.name for p in VOICES_DIR.glob("*.wav"))
            + [_voice_disp("(originál video)")]
        )

    def _refresh_voices(self):
        _wav_only = [_voice_disp("(žiadny)")] + sorted(p.name for p in VOICES_DIR.glob("*.wav"))
        cur_cb = self._cb_voice.currentText()
        self._cb_voice.clear(); self._cb_voice.addItems(self._get_voices())
        if cur_cb: self._cb_voice.setCurrentText(cur_cb)
        if hasattr(self, "_cb_default_voice"):
            cur_def = self._cb_default_voice.currentText()
            self._cb_default_voice.clear(); self._cb_default_voice.addItems(_wav_only)
            if cur_def: self._cb_default_voice.setCurrentText(cur_def)
        if hasattr(self, "_ov_voice"):
            cur_ov = self._ov_voice.currentText()
            self._ov_voice.blockSignals(True)
            self._ov_voice.clear(); self._ov_voice.addItems(self._get_voices())
            if cur_ov: self._ov_voice.setCurrentText(cur_ov)
            self._ov_voice.blockSignals(False)

    def _sync_voice_from_omnivoice(self, txt: str):
        """OmniVoice voice picker → propaguj do hlavného _cb_voice."""
        if hasattr(self, "_cb_voice") and self._cb_voice.currentText() != txt:
            self._cb_voice.setCurrentText(txt)

    def _add_voice_file(self):
        """File picker → skopíruj WAV do voices/ → refresh dropdownov."""
        import shutil
        path, _ = QFileDialog.getOpenFileName(
            self, "Pridať voice WAV", "", "Audio (*.wav *.mp3 *.flac);;All files (*)"
        )
        if not path:
            return
        src = Path(path)
        if not src.exists():
            return
        dst = VOICES_DIR / src.name
        if dst.exists():
            r = QMessageBox.question(self, "Prepísať?",
                f"Súbor {dst.name} už existuje. Prepísať?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return
        # If non-WAV, convert via ffmpeg
        if src.suffix.lower() != ".wav":
            dst = dst.with_suffix(".wav")
            try:
                import subprocess as _sp
                _sp.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "24000", str(dst)],
                        capture_output=True, timeout=60)
            except Exception as e:
                QMessageBox.warning(self, "Konverzia zlyhala", str(e))
                return
        else:
            try:
                shutil.copy(str(src), str(dst))
            except Exception as e:
                QMessageBox.warning(self, "Kopírovanie zlyhalo", str(e))
                return
        self._refresh_voices()
        # Auto-select novo pridaný voice
        if hasattr(self, "_ov_voice"):
            self._ov_voice.setCurrentText(dst.name)
        if hasattr(self, "_cb_male_voice"):
            cur_male = self._cb_male_voice.currentText()
            self._cb_male_voice.clear(); self._cb_male_voice.addItems(_wav_only)
            if cur_male: self._cb_male_voice.setCurrentText(cur_male)
        if hasattr(self, "_cb_female_voice"):
            cur_female = self._cb_female_voice.currentText()
            self._cb_female_voice.clear(); self._cb_female_voice.addItems(_wav_only)
            if cur_female: self._cb_female_voice.setCurrentText(cur_female)
        self._sync_clone_controls()

    def _sync_clone_controls(self, *_):
        _auto_changed = False
        if hasattr(self, "_cb_voice"):
            _cb_canon = _voice_canon(self._cb_voice.currentText())
            cb_is_original = (_cb_canon == "(originál video)")
            cb_has_custom_wav = _cb_canon not in ("(predvolený)", "(originál video)")
            cb_manual = cb_is_original and not self._cb_clone_auto.isChecked()
            self._cb_clone_auto.setVisible(cb_is_original)
            self._cb_clone_start_lbl.setVisible(cb_is_original)
            self._cb_clone_start.setVisible(cb_is_original)
            self._cb_clone_dur_lbl.setVisible(cb_is_original)
            self._cb_clone_dur.setVisible(cb_is_original)
            self._cb_clone_start_lbl.setEnabled(cb_manual)
            self._cb_clone_start.setEnabled(cb_manual)
            self._cb_clone_dur_lbl.setEnabled(cb_manual)
            self._cb_clone_dur.setEnabled(cb_manual)
            if hasattr(self, "_cb_emo_clone"):
                if cb_has_custom_wav:
                    self._cb_emo_clone.setText(T.get("chk_emo_video_style","Emócia z videa (štýl)"))
                    self._cb_emo_clone.setEnabled(True)
                    self._cb_emo_clone.setToolTip(T.get("tip_emo_video_style","Pri ručne zvolenom WAV hlase zostane identita hlasu zachovaná a z videa sa prenesie len dynamika, energia a štýl reči."))
                else:
                    self._cb_emo_clone.setText(T.get("chk_emo_video","Emócia z videa"))
                    self._cb_emo_clone.setEnabled(True)
                    self._cb_emo_clone.setToolTip(T.get("tip_emo_video_full","Pri predvolenom alebo originálnom hlase použije per-segment referenciu z videa ako plný clone prompt."))
            if hasattr(self, "_chk_multi_voice"):
                if cb_has_custom_wav and self._chk_multi_voice.isChecked():
                    self._chk_multi_voice.blockSignals(True)
                    self._chk_multi_voice.setChecked(False)
                    self._chk_multi_voice.blockSignals(False)
                    _auto_changed = True
                self._chk_multi_voice.setEnabled(not cb_has_custom_wav)
                if cb_has_custom_wav:
                    self._chk_multi_voice.setToolTip(T.get("tip_multi_voice_off","Pri ručne zvolenom WAV hlase je Multi-voice vypnutý, aby sa nemiešali viaceré referencie."))
                else:
                    self._chk_multi_voice.setToolTip("")
                if hasattr(self, "_multi_voice_n_lbl"):
                    self._multi_voice_n_lbl.setVisible(self._chk_multi_voice.isChecked() and not cb_has_custom_wav)
                if hasattr(self, "_multi_voice_n"):
                    self._multi_voice_n.setVisible(self._chk_multi_voice.isChecked() and not cb_has_custom_wav)
                _gender_voice_enabled = self._chk_multi_voice.isChecked() and not cb_has_custom_wav
                for w in (
                    getattr(self, "_cb_male_voice", None),
                    getattr(self, "_cb_male_voice_lbl", None),
                    getattr(self, "_cb_female_voice", None),
                    getattr(self, "_cb_female_voice_lbl", None),
                ):
                    if w is not None: w.setVisible(_gender_voice_enabled)
        if _auto_changed:
            self._save_chk_state()

    def _pick_cb_model(self):
        p,_ = QFileDialog.getOpenFileName(self,"Chatterbox model",str(BASE_DIR/"models"/"chatterbox"),"Safetensors (*.safetensors);;All (*)")
        if p:
            try: self._cb_model.setText(str(Path(p).relative_to(BASE_DIR)))
            except: self._cb_model.setText(p)
            self._save_chk_state()  # persist immediately

    def _log_add(self, line: str, log_widget: Optional[QTextEdit] = None):
        # Guard pre prípad keď _log_add je volané pred _build() (napr. _check_models_first_run)
        w = log_widget or getattr(self, "_log", None)
        if w is None:
            print(line, flush=True)
            return
        # Color-coded lines
        lu = line.upper()
        if any(tag in lu for tag in ("[CHYBA]","[ERROR]","ERROR:","FAILED","TRACEBACK")):
            color = "#f87171"  # red
        elif any(tag in lu for tag in ("[OK]","DONE","HOTOVO","✓","DOWNLOADED","FINISHED")):
            color = ACCENTL    # teal
        elif any(tag in lu for tag in ("[INFO]","[STEP]","[WARNING]","WARNING")):
            color = "#8888a8"  # muted gray
        else:
            color = FG
        escaped = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        cursor = w.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(f'<span style="color:{color}; font-family:monospace; font-size:11px;">{escaped}</span><br>')
        w.setTextCursor(cursor)
        w.ensureCursorVisible()
        # Update step bar based on pipeline log patterns (main log only)
        if w is (getattr(self, "_log", None)) and hasattr(self, "_step_bar"):
            ll = line.lower()
            if any(x in ll for x in ("whisperx","transcrib","whisper","stt","speech-to-text","rozpoznáv")):
                self._step_bar.set_step(0)
            elif any(x in ll for x in ("translat","preklad","google translate","gemma","nllb","madlad")):
                self._step_bar.set_step(1)
            elif any(x in ll for x in ("tts","chatterbox","synthesiz","generujem hlas","speak")):
                self._step_bar.set_step(2)
            elif any(x in ll for x in ("assembl","ffmpeg mix","finalizáci","finalize","audio mix","mastering")):
                self._step_bar.set_step(3)

    def _set_running(self, running: bool, is_yt=False):
        if is_yt:
            self._yt_start.setEnabled(not running)
            self._yt_stop.setEnabled(running)
        else:
            self._start_btn.setEnabled(not running)
            self._stop_btn.setEnabled(running)
        if running:
            self._badge.setText(T.get("status_running","● Beží…"))
            self._badge.setStyleSheet(f"color:{ACCENTL}; font-weight:bold; background:transparent;")
            self._pulse_state = False
            if hasattr(self, "_step_bar"): self._step_bar.reset()
            if not self._proc_timer.isActive():
                self._start_time = time.time()
                self._last_progress = 0.0
                self._proc_timer.start()
        else:
            self._badge.setText(T.get("status_ready","● Pripravené"))
            self._badge.setStyleSheet(f"color:{GREEN}; font-weight:bold; background:transparent;")
            self._proc_timer.stop()
            self._time_label.setText("")

    def _on_timer_tick(self):
        elapsed = time.time() - self._start_time
        e_str = _fmt_duration(elapsed)
        pct = self._last_progress
        if pct > 1.0:
            eta = elapsed * (100.0 - pct) / pct
            self._time_label.setText(f"{e_str}  ETA {_fmt_duration(eta)}")
        else:
            self._time_label.setText(e_str)
        # Pulse badge
        self._pulse_state = not self._pulse_state
        badge_color = ACCENTL if self._pulse_state else ACCENTH
        self._badge.setStyleSheet(f"color:{badge_color}; font-weight:bold; background:transparent;")

    def _content_type_val(self) -> str:
        for btn in self._content_grp.buttons():
            if btn.isChecked(): return btn.property("val") or "general"
        return "general"

    def _selected_tts_engine(self) -> str:
        if self._rb_omnivoice.isChecked():
            return "omnivoice"
        return "chatterbox"

    def _tts_is_chatterbox(self) -> bool:
        return self._rb_chatterbox.isChecked()

    def _tts_is_omnivoice(self) -> bool:
        return self._rb_omnivoice.isChecked()

    def _set_selected_tts_engine(self, engine: str) -> None:
        if engine == "omnivoice":
            self._rb_omnivoice.setChecked(True)
        else:
            self._rb_chatterbox.setChecked(True)

    def _set_combo_value(self, combo: QComboBox, value: Optional[str], fallback: Optional[str] = None) -> None:
        # Voice sentinels are stored canonically (SK) but displayed translated;
        # try the translated form first when looking up by text.
        for cand in (value, _voice_disp(value) if value else None,
                     fallback, _voice_disp(fallback) if fallback else None):
            if cand and combo.findText(cand) >= 0:
                combo.setCurrentText(cand)
                return

    def _checkpoint_settings_snapshot(self) -> Dict[str, Any]:
        return {
            "src_lang": self._src_lang.currentText(),
            "lang": self._tgt_lang.currentText(),
            "tts": self._selected_tts_engine(),
            "trans_engine": self._trans_engine.currentText(),
            "out_dir": self._out_dir.text(),
            "content_type": self._content_type_val(),
            "keep_music": self._chk_keep_music.isChecked(),
            "refine": self._chk_refine.isChecked(),
            "voice_eq": self._chk_voice_eq.isChecked(),
            "multi_voice": self._chk_multi_voice.isChecked(),
            "multi_voice_n": self._multi_voice_n.value(),
            "srt_align": self._chk_srt_align.isChecked(),
            "eq_profile": self._eq_profile.currentText(),
            "denoise": self._chk_denoise.isChecked(),
            "denoise_preset": self._denoise_preset.currentText(),
            "pitch_shift": self._pitch_shift.value(),
            "cb_model": self._cb_model.text().strip(),
            "cb_voice": self._cb_voice.currentText(),
            "cb_default_voice": self._cb_default_voice.currentText() if hasattr(self, "_cb_default_voice") else "",
            "cb_male_voice": self._cb_male_voice.currentText() if hasattr(self, "_cb_male_voice") else "",
            "cb_female_voice": self._cb_female_voice.currentText() if hasattr(self, "_cb_female_voice") else "",
            "cb_clone_auto": self._cb_clone_auto.isChecked(),
            "cb_clone_start": self._cb_clone_start.value(),
            "cb_clone_duration": self._cb_clone_dur.value(),
            "cb_emo_clone": self._cb_emo_clone.isChecked(),
            "text_adapt_enabled": self._text_adapt_enabled.isChecked() if hasattr(self, "_text_adapt_enabled") else self.settings.text_adapt_enabled,
            "cb_exaggeration": self._cb_exag.value() if hasattr(self,"_cb_exag") else self.settings.cb_exaggeration,
            "cb_cfg": self._cb_cfg.value() if hasattr(self,"_cb_cfg") else self.settings.cb_cfg,
            "cb_temp": self._cb_temp.value() if hasattr(self,"_cb_temp") else self.settings.cb_temp,
            "ov_num_step": self._ov_num_step.value() if hasattr(self,"_ov_num_step") else getattr(self.settings, "ov_num_step", 64),
            "ov_guidance": self._ov_guidance.value() if hasattr(self,"_ov_guidance") else getattr(self.settings, "ov_guidance", 2.5),
            "ov_language": self._tgt_lang.currentText() if hasattr(self,"_tgt_lang") else getattr(self.settings, "ov_language", "sk"),
            "ov_instruct": self._ov_instruct.currentText() if hasattr(self,"_ov_instruct") else getattr(self.settings, "ov_instruct", ""),
            "ov_speed": self._ov_speed.value() if hasattr(self,"_ov_speed") else getattr(self.settings, "ov_speed", 1.0),
            "ov_expressive": self._ov_expressive.isChecked() if hasattr(self,"_ov_expressive") else getattr(self.settings, "ov_expressive", False),
            "ov_multi_voice": self._ov_multi_voice.isChecked() if hasattr(self,"_ov_multi_voice") else getattr(self.settings, "ov_multi_voice", False),
            "srt_input": self._srt_in.text().strip() if hasattr(self, "_srt_in") else "",
        }

    def _apply_checkpoint_settings(self, settings: Optional[Dict[str, Any]]) -> None:
        if not settings:
            return
        self._refresh_voices()
        if "src_lang" in settings:
            self._src_lang.setCurrentText(settings["src_lang"])
        if "lang" in settings:
            self._tgt_lang.setCurrentText(settings["lang"])
        if "tts" in settings:
            self._set_selected_tts_engine(settings["tts"])
        if "trans_engine" in settings:
            self._trans_engine.setCurrentText(settings["trans_engine"])
        if settings.get("out_dir") is not None:
            self._out_dir.setText(settings.get("out_dir") or "")
        if "content_type" in settings:
            for btn in self._content_grp.buttons():
                if btn.property("val") == settings["content_type"]:
                    btn.setChecked(True)
                    break
        if "keep_music" in settings:
            self._chk_keep_music.setChecked(bool(settings["keep_music"]))
        if "refine" in settings:
            self._chk_refine.setChecked(bool(settings["refine"]))
        if "voice_eq" in settings:
            self._chk_voice_eq.setChecked(bool(settings["voice_eq"]))
        if "multi_voice" in settings:
            self._chk_multi_voice.setChecked(bool(settings["multi_voice"]))
        if "multi_voice_n" in settings:
            self._multi_voice_n.setValue(max(2, int(settings["multi_voice_n"])))
        if "srt_align" in settings:
            self._chk_srt_align.setChecked(bool(settings["srt_align"]))
        if "eq_profile" in settings:
            self._set_combo_value(self._eq_profile, settings["eq_profile"])
        if "denoise" in settings:
            self._chk_denoise.setChecked(bool(settings["denoise"]))
        if "denoise_preset" in settings:
            self._set_combo_value(self._denoise_preset, settings["denoise_preset"])
        if "pitch_shift" in settings:
            self._pitch_shift.setValue(float(settings["pitch_shift"]))
        if "cb_model" in settings:
            self._cb_model.setText(settings["cb_model"] or str(_DEFAULT_CHATTERBOX_MODEL))
        self._set_combo_value(self._cb_voice, settings.get("cb_voice"), "(predvolený)")
        if hasattr(self, "_cb_default_voice"):
            self._set_combo_value(self._cb_default_voice, settings.get("cb_default_voice"), "tomas_sk.wav")
        if hasattr(self, "_cb_male_voice"):
            self._set_combo_value(self._cb_male_voice, settings.get("cb_male_voice"), "(žiadny)")
        if hasattr(self, "_cb_female_voice"):
            self._set_combo_value(self._cb_female_voice, settings.get("cb_female_voice"), "(žiadny)")
        if "cb_clone_auto" in settings:
            self._cb_clone_auto.setChecked(bool(settings["cb_clone_auto"]))
        if "cb_clone_start" in settings:
            self._cb_clone_start.setValue(float(settings["cb_clone_start"]))
        if "cb_clone_duration" in settings:
            self._cb_clone_dur.setValue(float(settings["cb_clone_duration"]))
        if "cb_emo_clone" in settings:
            self._cb_emo_clone.setChecked(bool(settings["cb_emo_clone"]))
        if "text_adapt_enabled" in settings and hasattr(self, "_text_adapt_enabled"):
            self._text_adapt_enabled.setChecked(bool(settings["text_adapt_enabled"]))
        elif "cb_text_adapt" in settings and hasattr(self, "_text_adapt_enabled"):
            self._text_adapt_enabled.setChecked(bool(settings["cb_text_adapt"]))
        if "cb_exaggeration" in settings:
            self._cb_exag.setValue(float(settings["cb_exaggeration"]))
        if "cb_cfg" in settings:
            self._cb_cfg.setValue(float(settings["cb_cfg"]))
        if "cb_temp" in settings:
            self._cb_temp.setValue(float(settings["cb_temp"]))
        if "ov_num_step" in settings and hasattr(self, "_ov_num_step"):
            self._ov_num_step.setValue(int(settings["ov_num_step"]))
        if "ov_guidance" in settings and hasattr(self, "_ov_guidance"):
            self._ov_guidance.setValue(float(settings["ov_guidance"]))
        # ov_language merged into tgt_lang (single source) — preskočené
        # ov_instruct sa NEOBNOVUJE zo settings — vždy začína prázdne pri novom behu GUI
        # (voice design je ad-hoc per video, nie persistent voľba)
        if "ov_speed" in settings and hasattr(self, "_ov_speed"):
            self._ov_speed.setValue(float(settings["ov_speed"]))
        if "ov_expressive" in settings and hasattr(self, "_ov_expressive"):
            self._ov_expressive.setChecked(bool(settings["ov_expressive"]))
        if "ov_multi_voice" in settings and hasattr(self, "_ov_multi_voice"):
            self._ov_multi_voice.setChecked(bool(settings["ov_multi_voice"]))
        if "srt_input" in settings and hasattr(self, "_srt_in"):
            # SRT je per-video voľba — neukladáme ju trvalo, len ak user explicitne nastavil pre rerun
            # Necháme prázdne pri reštarte (uživateľ ho znovu vyberie ak treba)
            pass
        self._sync_clone_controls()
        self._save_chk_state()

    def _get_api_key(self, engine: str) -> str:
        m = {"chatgpt": self._oa_key, "grok": self._grok_key, "gemini": self._gem_key}
        w = m.get(engine)
        direct = w.text().strip() if w else ""
        if direct:
            return direct
        env_map = {
            "chatgpt": ("API_TRANS_KEY", "OPENAI_API_KEY"),
            "grok": ("API_TRANS_KEY", "GROK_API_KEY", "XAI_API_KEY"),
            "gemini": ("API_TRANS_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
        }
        for env_name in env_map.get(engine, ()):
            value = os.environ.get(env_name, "").strip()
            if value:
                return value
        return ""

    def _get_hf_api_key(self) -> str:
        direct = self._hf_key.text().strip() if hasattr(self, "_hf_key") else ""
        if direct:
            return direct
        return (
            os.environ.get("HF_TOKEN", "").strip()
            or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Batch management
    # ══════════════════════════════════════════════════════════════════════════
    def _batch_add(self):
        paths,_ = QFileDialog.getOpenFileNames(self,"Vybrať videá","","Video/Audio (*.mp4 *.mkv *.avi *.mov *.webm *.mp3 *.wav);;All (*)")
        existing = {i.path for i in self.batch_items}
        for p in paths:
            if p and p not in existing:
                self.batch_items.append(BatchItem(path=p))
        self._refresh_batch()

    def _batch_remove(self):
        rows = sorted([i.row() for i in self._batch_list.selectedItems()], reverse=True)
        for r in rows:
            if r < len(self.batch_items): del self.batch_items[r]
        self._refresh_batch()

    def _batch_clear(self):
        self.batch_items.clear(); self._refresh_batch()

    def _on_files_dropped(self, paths: list):
        existing = {i.path for i in self.batch_items}
        for p in paths:
            if p not in existing:
                self.batch_items.append(BatchItem(path=p))
        self._refresh_batch()

    def _refresh_batch(self):
        self._batch_list.clear()
        has_items = bool(self.batch_items)
        self._drop_zone.setVisible(not has_items)
        self._batch_list.setVisible(has_items)
        status_icons = {"pending":"○","processing":"▶","done":"✓","failed":"✗"}
        status_colors = {"done":QColor(GREEN),"failed":QColor(RED),"processing":QColor(YELLOW)}
        for item in self.batch_items:
            icon = status_icons.get(item.status,"?")
            p = Path(item.path)
            # Try duration
            dur = _probe_media_duration(p)
            dur_str = f"  {_fmt_duration(dur)}" if dur else ""
            li = QListWidgetItem(f" {icon}  {p.name}{dur_str}")
            li.setSizeHint(QSize(0, 46))
            if item.status in status_colors:
                li.setForeground(status_colors[item.status])
            # Thumbnail (only for video files)
            if p.suffix.lower() in {'.mp4','.mkv','.avi','.mov','.webm'}:
                px = _extract_thumbnail(str(p), 72, 40)
                if px:
                    li.setIcon(QIcon(px))
            self._batch_list.addItem(li)

    def _load_segments_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Otvoriť WhisperX JSON", "", "JSON (*.json)")
        if not path: return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            segs = data.get("segments", [])
            if not segs:
                QMessageBox.warning(self, "Prázdny JSON", "Súbor neobsahuje segmenty."); return
            src_lines, tgt_lines = [], []
            for i, seg in enumerate(segs):
                t0 = seg.get("start", 0.0); t1 = seg.get("end", 0.0)
                ts = f"[{t0:.2f} → {t1:.2f}]"
                src_lines.append(f"{i+1:03d} {ts}\n{seg.get('text','').strip()}\n")
                tgt = seg.get("translated_text") or seg.get("translation","")
                tgt_lines.append(f"{i+1:03d} {ts}\n{tgt.strip()}\n")
            self._seg_src.setPlainText("\n".join(src_lines))
            self._seg_tgt.setPlainText("\n".join(tgt_lines))
            self._seg_section.expand()
        except Exception as e:
            QMessageBox.critical(self, "Chyba", str(e))

    def _batch_save_list(self):
        if not self.batch_items:
            QMessageBox.information(self,"Prázdny zoznam","Nemáte žiadne videá na uloženie."); return
        name,ok = QInputDialog.getText(self,"Uložiť zoznam","Názov zoznamu:")
        if not ok or not name.strip(): return
        name = name.strip()
        lists = _load_video_lists()
        if name in lists:
            r = QMessageBox.question(self,"Prepísať?",f"Zoznam '{name}' už existuje. Prepísať?")
            if r != QMessageBox.StandardButton.Yes: return
        lists[name] = [i.path for i in self.batch_items]
        _save_video_lists(lists)
        QMessageBox.information(self,"Uložené",f"Zoznam '{name}' uložený ({len(lists[name])} videí).")

    def _batch_load_list(self):
        lists = _load_video_lists()
        if not lists: QMessageBox.information(self,"Žiadne zoznamy","Žiadne uložené zoznamy."); return
        name,ok = QInputDialog.getItem(self,"Načítať zoznam","Vyberte zoznam:",sorted(lists.keys()),editable=False)
        if not ok: return
        self.batch_items = [BatchItem(path=p) for p in lists[name]]
        self._refresh_batch()

    def _checkpoint_menu(self):
        ckpt = _load_checkpoint()
        items = [
            ("Uložiť checkpoint", self._checkpoint_save),
            ("Načítať (auto)", lambda: self._checkpoint_load(ckpt)),
            ("Správca checkpointov", self._checkpoint_manager),
        ]
        # Simple dialog
        choice, ok = QInputDialog.getItem(self,"Checkpoint","Akcia:",
            [t for t,_ in items], editable=False)
        if not ok: return
        for t,fn in items:
            if t == choice: fn(); break

    def _checkpoint_save(self):
        name,ok = QInputDialog.getText(self,"Uložiť checkpoint","Názov:")
        if not ok or not name.strip(): return
        snap = self._checkpoint_settings_snapshot()
        _save_checkpoint(self.batch_items, 0, snap)
        ckpts = _load_saved_checkpoints()
        done = sum(1 for i in self.batch_items if i.status=="done")
        ckpts[name.strip()] = {"items":[{"path":i.path,"status":i.status,"error":i.error} for i in self.batch_items],
                                "settings":snap,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
                                "done":done,"total":len(self.batch_items)}
        _save_saved_checkpoints(ckpts)
        QMessageBox.information(self,"Uložené",f"Checkpoint '{name.strip()}' uložený.")

    def _checkpoint_load(self, ckpt):
        if not ckpt: QMessageBox.information(self,"Checkpoint","Žiadny auto-checkpoint."); return
        items,_,settings = ckpt
        for i in items:
            if i.status=="processing": i.status="pending"
        self.batch_items = items; self._refresh_batch()
        self._apply_checkpoint_settings(settings)
        self._log_add(f"Checkpoint načítaný ({sum(1 for i in items if i.status=='done')}/{len(items)} hotových)")

    def _checkpoint_manager(self):
        ckpts = _load_saved_checkpoints()
        if not ckpts: QMessageBox.information(self,"Správca","Žiadne uložené checkpointy."); return
        entries = sorted(ckpts.keys())
        name,ok = QInputDialog.getItem(self,"Správca checkpointov","Vyberte checkpoint:",entries,editable=False)
        if not ok: return
        ckpt = ckpts[name]
        items = [BatchItem(**{k:v for k,v in i.items() if k in ("path","status","error")}) for i in ckpt.get("items",[])]
        for i in items:
            if i.status=="processing": i.status="pending"
        self.batch_items = items; self._refresh_batch()
        self._apply_checkpoint_settings(ckpt.get("settings"))
        self._log_add(f"Checkpoint '{name}' načítaný.")


    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 9 — Narrátor (TTS narration via OmniVoice)
    # ══════════════════════════════════════════════════════════════════════════
    # Narrátor cesty — env override má prednosť, fallback na auto-detect cez paths.PATHS.
    # User môže nastaviť: VTS_NARRATOR_SCRIPT, VTS_NARRATOR_PYTHON, VTS_NARRATOR_VOICES_DIR, VTS_NARRATOR_OUT_DIR
    _NARRATOR_SCRIPT = os.environ.get(
        "VTS_NARRATOR_SCRIPT",
        str(Path(__file__).resolve().parent / "scripts" / "notes_to_audio_omnivoice.py"),
    )
    # OmniVoice je nainštalovaný v f5tts_env (cez PATHS.python_omnivoice).
    # python_fish (fish_env) NIE JE správny — fish_env nemá OmniVoice.
    _NARRATOR_PYTHON = os.environ.get(
        "VTS_NARRATOR_PYTHON",
        str(PATHS.python_omnivoice) if Path(PATHS.python_omnivoice).exists()
        else (str(PATHS.python_fish) if Path(PATHS.python_fish).exists() else "python3"),
    )
    _NARRATOR_VOICES_DIR = os.environ.get(
        "VTS_NARRATOR_VOICES_DIR",
        str(Path(__file__).resolve().parent / "voices"),
    )
    _NARRATOR_DEFAULT_OUT = os.environ.get(
        "VTS_NARRATOR_OUT_DIR",
        str(Path.home() / "VideoTranslator_output" / "narration"),
    )

    def _page_narrator(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(16, 14, 16, 14); lay.setSpacing(10)
        lay.addWidget(self._tr_reg(self._sect(T.get("sect_narrator", "NARRÁTOR — TEXT NA REČ")),
                                    "sect_narrator", "NARRÁTOR — TEXT NA REČ"))

        # ── Vstup: text alebo súbor ──────────────────────────────────────────
        grp_in = self._tr_grp("grp_narr_input", "Vstup")
        il = QGridLayout(grp_in); il.setSpacing(6)

        # Tabbed: textarea | súbor
        self._narr_tabs = QTabWidget()
        # Tab 1 — textarea
        ta_w = QWidget(); ta_l = QVBoxLayout(ta_w); ta_l.setContentsMargins(4, 6, 4, 4)
        self._narr_text = QTextEdit()
        self._narr_text.setPlaceholderText(T.get(
            "ph_narr_text",
            "Vlož sem text, ktorý sa má nahovoriť…\n\nPodporuje markdown — nadpisy ##, zoznamy -, atď."))
        self._narr_text.setMinimumHeight(180)
        ta_l.addWidget(self._narr_text)
        self._narr_tabs.addTab(ta_w, T.get("tab_narr_paste", "Vložiť text"))

        # Tab 2 — file
        fp_w = QWidget(); fp_l = QGridLayout(fp_w); fp_l.setContentsMargins(4, 6, 4, 4); fp_l.setSpacing(6)
        fp_l.addWidget(self._tr_lbl("lbl_narr_file", "Súbor (.md / .txt / .docx / .pdf):"), 0, 0)
        self._narr_file = QLineEdit(); self._narr_file.setFixedHeight(26)
        b_pick = QPushButton("…"); b_pick.setFixedSize(32, 32); b_pick.setStyleSheet(_pill_sm())
        b_pick.clicked.connect(lambda: self._narr_file.setText(
            QFileDialog.getOpenFileName(self, T.get("dlg_narr_pick", "Vybrať textový súbor"), "",
                                         "Text/Markdown/DOCX/PDF (*.md *.txt *.docx *.pdf);;All (*)")[0]
            or self._narr_file.text()))
        fp_l.addWidget(self._narr_file, 0, 1); fp_l.addWidget(b_pick, 0, 2)
        fp_l.setColumnStretch(1, 1)
        self._narr_tabs.addTab(fp_w, T.get("tab_narr_file", "Načítať súbor"))

        il.addWidget(self._narr_tabs, 0, 0, 1, 4)
        lay.addWidget(grp_in)

        # ── Hlas a parametre ─────────────────────────────────────────────────
        grp_voice = self._tr_grp("grp_narr_voice", "Hlas a tempo")
        vl = QGridLayout(grp_voice); vl.setSpacing(6)

        vl.addWidget(self._tr_lbl("lbl_narr_voice", "Referenčný hlas (.wav):"), 0, 0)
        self._narr_voice = QComboBox(); self._narr_voice.setFixedHeight(26)
        self._narr_voice.setEditable(True)
        self._narr_refresh_voices()
        vl.addWidget(self._narr_voice, 0, 1, 1, 2)
        b_refresh = QPushButton("↺"); b_refresh.setFixedSize(28, 28); b_refresh.setStyleSheet(_pill_sm())
        b_refresh.clicked.connect(self._narr_refresh_voices)
        vl.addWidget(b_refresh, 0, 3)

        vl.addWidget(self._tr_lbl("lbl_narr_speed", "Tempo:"), 1, 0)
        self._narr_speed = QDoubleSpinBox()
        self._narr_speed.setRange(0.7, 1.5); self._narr_speed.setSingleStep(0.05)
        self._narr_speed.setValue(1.00); self._narr_speed.setFixedHeight(26)
        vl.addWidget(self._narr_speed, 1, 1)

        vl.addWidget(self._tr_lbl("lbl_narr_instruct", "Štýl (instruct):"), 1, 2)
        self._narr_instruct = QLineEdit("male, young adult, moderate pitch")
        self._narr_instruct.setFixedHeight(26)
        self._narr_instruct.setToolTip(
            "OmniVoice instruct — povolené pojmy: male/female, young adult/middle-aged/elderly, "
            "high pitch/moderate pitch/low pitch, whisper, *_accent")
        vl.addWidget(self._narr_instruct, 1, 3)

        vl.setColumnStretch(1, 1); vl.setColumnStretch(3, 1)
        lay.addWidget(grp_voice)

        # ── Výstup ───────────────────────────────────────────────────────────
        grp_out = self._tr_grp("grp_narr_output", "Výstup")
        ol = QGridLayout(grp_out); ol.setSpacing(6)
        ol.addWidget(self._tr_lbl("lbl_narr_outfile", "Výstupný .wav:"), 0, 0)
        self._narr_out = QLineEdit(self._NARRATOR_DEFAULT_OUT + "/narration.wav")
        self._narr_out.setFixedHeight(26)
        b_pick_out = QPushButton("…"); b_pick_out.setFixedSize(32, 32); b_pick_out.setStyleSheet(_pill_sm())
        b_pick_out.clicked.connect(lambda: self._narr_out.setText(
            QFileDialog.getSaveFileName(self, T.get("dlg_narr_save", "Uložiť audio"),
                                         self._narr_out.text(), "WAV (*.wav)")[0] or self._narr_out.text()))
        ol.addWidget(self._narr_out, 0, 1); ol.addWidget(b_pick_out, 0, 2)
        ol.setColumnStretch(1, 1)
        lay.addWidget(grp_out)

        # ── Tlačidlá ─────────────────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        self._narr_btn_run = self._tr_btn("btn_narr_generate", "▶  Vygenerovať audio")
        self._narr_btn_run.setFixedHeight(34); self._narr_btn_run.setStyleSheet(_pill(r=8))
        self._narr_btn_run.clicked.connect(self._narr_run)
        btn_row.addWidget(self._narr_btn_run, 1)

        self._narr_btn_stop = self._tr_btn("btn_narr_stop", "■  Zastaviť")
        self._narr_btn_stop.setFixedHeight(34); self._narr_btn_stop.setEnabled(False)
        self._narr_btn_stop.setStyleSheet(_pill(bg="#3d1a1a", hover="#5a2020", pressed="#7a2020", text=RED, r=8))
        self._narr_btn_stop.clicked.connect(self._narr_stop)
        btn_row.addWidget(self._narr_btn_stop)

        # Po-úspešné akcie: prehrať / otvoriť priečinok
        self._narr_btn_play = self._tr_btn("btn_narr_play", "▶  Prehrať")
        self._narr_btn_play.setFixedHeight(34); self._narr_btn_play.setEnabled(False)
        self._narr_btn_play.setStyleSheet(_pill(r=8))
        self._narr_btn_play.clicked.connect(self._narr_play)
        btn_row.addWidget(self._narr_btn_play)

        self._narr_btn_open = self._tr_btn("btn_narr_open", "📂  Otvoriť priečinok")
        self._narr_btn_open.setFixedHeight(34); self._narr_btn_open.setEnabled(False)
        self._narr_btn_open.setStyleSheet(_pill(r=8))
        self._narr_btn_open.clicked.connect(self._narr_open_dir)
        btn_row.addWidget(self._narr_btn_open)
        lay.addLayout(btn_row)

        # ── Log ──────────────────────────────────────────────────────────────
        lay.addWidget(self._sect("LOG"))
        self._narr_log = QTextEdit(); self._narr_log.setReadOnly(True)
        self._narr_log.setMinimumHeight(160)
        self._narr_log.document().setMaximumBlockCount(2000)
        lay.addWidget(self._narr_log)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        scroll.setWidget(inner)
        return scroll

    def _narr_refresh_voices(self):
        d = Path(self._NARRATOR_VOICES_DIR)
        cur = self._narr_voice.currentText() if hasattr(self, "_narr_voice") else ""
        wavs = sorted(d.glob("*.wav")) if d.exists() else []
        self._narr_voice.clear()
        for w in wavs:
            self._narr_voice.addItem(str(w))
        # Default voice priority (first existing): SK natives produkujú stabilnejšie
        # output než video-clone WAV-y (clone-y majú trailing silence ktoré OmniVoice
        # napodobí → "too_short" QC warning).
        defaults = ["lubo_sk_studio.wav", "lubo_sk.wav", "juraj_sk_studio.wav",
                    "juraj_sk.wav", "chatterbox_sk_narrative_studio.wav",
                    "katka_sk.wav"]
        for fav in defaults:
            for i in range(self._narr_voice.count()):
                t = self._narr_voice.itemText(i)
                if t.endswith("/" + fav):
                    self._narr_voice.setCurrentIndex(i)
                    break
            else:
                continue
            break
        if cur:
            idx = self._narr_voice.findText(cur)
            if idx >= 0: self._narr_voice.setCurrentIndex(idx)

    def _narr_log_add(self, msg: str):
        # _narr_log je QTextEdit (nie QPlainTextEdit) — používame append() namiesto appendPlainText()
        self._narr_log.append(msg.rstrip())
        self._narr_log.verticalScrollBar().setValue(
            self._narr_log.verticalScrollBar().maximum())

    def _narr_input_to_md(self) -> Optional[str]:
        """Vráti cestu k .md súboru (z textarea uloží do tmp, alebo konvertuje docx/pdf)."""
        if self._narr_tabs.currentIndex() == 0:
            txt = self._narr_text.toPlainText().strip()
            if not txt:
                self._narr_log_add("[CHYBA] Prázdny text.")
                return None
            tmp = Path("/tmp/narrator_input.md")
            tmp.write_text(txt, encoding="utf-8")
            return str(tmp)
        # Súbor
        src = self._narr_file.text().strip()
        if not src or not Path(src).exists():
            self._narr_log_add(f"[CHYBA] Súbor neexistuje: {src}")
            return None
        ext = Path(src).suffix.lower()
        if ext in (".md", ".txt"):
            return src
        if ext == ".docx":
            self._narr_log_add(f"[INFO] Konvertujem DOCX → TXT cez libreoffice…")
            tmpdir = "/tmp"
            try:
                subprocess.run(["libreoffice", "--headless", "--convert-to", "txt",
                                src, "--outdir", tmpdir], check=True, timeout=60,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                out = Path(tmpdir) / (Path(src).stem + ".txt")
                if out.exists():
                    return str(out)
            except Exception as e:
                self._narr_log_add(f"[CHYBA] DOCX konverzia: {e}")
            return None
        if ext == ".pdf":
            self._narr_log_add(f"[INFO] Konvertujem PDF → TXT cez pdftotext…")
            out = Path("/tmp") / (Path(src).stem + ".txt")
            try:
                subprocess.run(["pdftotext", "-layout", src, str(out)],
                                check=True, timeout=60,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if out.exists():
                    return str(out)
            except Exception as e:
                self._narr_log_add(f"[CHYBA] PDF konverzia: {e}")
            return None
        self._narr_log_add(f"[CHYBA] Nepodporovaný formát: {ext}")
        return None

    def _narr_run(self):
        if hasattr(self, "_narr_proc") and self._narr_proc is not None:
            try:
                if self._narr_proc.poll() is None:
                    self._narr_log_add("[INFO] Beží — počkaj alebo zastav.")
                    return
            except Exception:
                pass

        md_path = self._narr_input_to_md()
        if not md_path:
            return

        out_wav = self._narr_out.text().strip() or (self._NARRATOR_DEFAULT_OUT + "/narration.wav")
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

        ref_wav = self._narr_voice.currentText().strip()
        speed = float(self._narr_speed.value())
        instruct = self._narr_instruct.text().strip()

        # Auto-load ref_text z cache alebo Whisper transcribe (kritické — bez správneho
        # ref_text generuje OmniVoice nezmysly / fragmenty DEFAULT_REF_TEXT-u)
        ref_text = self._narr_get_ref_text(ref_wav)

        cmd = [self._NARRATOR_PYTHON, self._NARRATOR_SCRIPT,
                md_path, out_wav,
                "--ref-wav", ref_wav,
                "--speed", str(speed)]
        if instruct:
            cmd += ["--instruct", instruct]
        if ref_text:
            cmd += ["--ref-text", ref_text]

        self._narr_log.clear()
        self._narr_log_add(f"[CMD] {' '.join(cmd)}")
        self._narr_log_add(f"[INFO] Štartujem OmniVoice TTS… (môže trvať pár minút)")

        self._narr_btn_run.setEnabled(False)
        self._narr_btn_stop.setEnabled(True)

        try:
            import subprocess as _sp
            self._narr_proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT,
                                         text=True, bufsize=1)
        except Exception as e:
            self._narr_log_add(f"[CHYBA] Nedá sa spustiť: {e}")
            self._narr_btn_run.setEnabled(True); self._narr_btn_stop.setEnabled(False)
            return

        # Čítanie výstupu cez QTimer (non-blocking)
        if not hasattr(self, "_narr_timer"):
            self._narr_timer = QTimer(self); self._narr_timer.setInterval(200)
            self._narr_timer.timeout.connect(self._narr_poll_proc)
        self._narr_timer.start()

    def _narr_poll_proc(self):
        proc = getattr(self, "_narr_proc", None)
        if proc is None:
            self._narr_timer.stop(); return
        # Read available output
        try:
            import select
            if proc.stdout:
                while True:
                    ready, _, _ = select.select([proc.stdout], [], [], 0)
                    if not ready:
                        break
                    line = proc.stdout.readline()
                    if not line:
                        break
                    self._narr_log_add(line.rstrip())
        except Exception:
            pass

        if proc.poll() is not None:
            # Drain rest
            if proc.stdout:
                rest = proc.stdout.read()
                if rest:
                    for ln in rest.splitlines():
                        self._narr_log_add(ln)
            self._narr_timer.stop()
            self._narr_log_add(f"[INFO] Hotovo (exit code {proc.returncode}).")
            self._narr_btn_run.setEnabled(True); self._narr_btn_stop.setEnabled(False)
            # Po-úspešná akcia: zobraz output path + povol Play / Open
            out_path = Path(self._narr_out.text().strip())
            if proc.returncode == 0 and out_path.exists():
                self._narr_log_add(f"[OUT] {out_path}")
                self._narr_log_add(f"[OUT] {out_path.with_suffix('.mp3')}")
                self._narr_btn_play.setEnabled(True)
                self._narr_btn_open.setEnabled(True)
            self._narr_proc = None

    def _narr_get_ref_text(self, ref_wav: str) -> str:
        """Vráti správny ref_text pre dané ref_wav. Postupnosť:
        1) cache: <ref_dir>/.ref_text_cache/<stem>.txt
        2) cache: voices/.ref_text_cache/<stem>.txt (legacy)
        3) Whisper transcribe (musetalk_env, lang=sk) + ulož do cache
        """
        if not ref_wav:
            return ""
        ref_path = Path(ref_wav).expanduser().resolve()
        if not ref_path.exists():
            return ""
        # 1+2) cache
        for cache_dir in (ref_path.parent / ".ref_text_cache",
                          Path(self._NARRATOR_VOICES_DIR) / ".ref_text_cache"):
            cache_file = cache_dir / f"{ref_path.stem}.txt"
            if cache_file.exists():
                txt = cache_file.read_text(encoding="utf-8").strip()
                if txt:
                    self._narr_log_add(f"[REF] ref_text cached: {txt[:80]}")
                    return txt
        # 3) Whisper transcribe — beží v musetalk_env (má faster-whisper)
        self._narr_log_add(f"[REF] Whisper transcribe ref WAV: {ref_path.name}...")
        try:
            import subprocess as _sp
            py_main = self._effective_tts_python() if hasattr(self, "_effective_tts_python") \
                else "/mnt/tts_data/miniforge3/envs/musetalk_env/bin/python"
            code = (
                "import os\n"
                "os.environ['LD_LIBRARY_PATH'] = "
                "'/mnt/tts_data/miniforge3/envs/musetalk_env/lib/python3.10/site-packages/nvidia/cublas/lib:'"
                " + '/mnt/tts_data/miniforge3/envs/musetalk_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:'"
                " + os.environ.get('LD_LIBRARY_PATH', '')\n"
                "from faster_whisper import WhisperModel\n"
                "import torch\n"
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
                "ct = 'int8' if device == 'cpu' else 'int8'\n"
                f"m = WhisperModel('base', device=device, compute_type=ct)\n"
                f"segs, _ = m.transcribe({str(ref_path)!r}, language='sk', beam_size=1)\n"
                "print(' '.join(s.text.strip() for s in segs).strip())\n"
            )
            r = _sp.run([py_main, "-c", code], capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                txt = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
                if txt:
                    # Save to cache
                    cache_dir = ref_path.parent / ".ref_text_cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    (cache_dir / f"{ref_path.stem}.txt").write_text(txt, encoding="utf-8")
                    self._narr_log_add(f"[REF] ref_text Whisper: {txt[:80]}")
                    return txt
            else:
                self._narr_log_add(f"[REF] Whisper zlyhal: {(r.stderr or '')[-200:]}")
        except Exception as e:
            self._narr_log_add(f"[REF] Whisper exception: {e}")
        return ""

    def _narr_play(self):
        """Prehrať vygenerované audio cez systémový default player (xdg-open)."""
        out_path = Path(self._narr_out.text().strip())
        if not out_path.exists():
            self._narr_log_add(f"[CHYBA] Súbor nenájdený: {out_path}")
            return
        try:
            import subprocess as _sp
            _sp.Popen(["xdg-open", str(out_path)],
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, start_new_session=True)
            self._narr_log_add(f"[PLAY] {out_path.name}")
        except Exception as e:
            self._narr_log_add(f"[PLAY] zlyhalo ({e}) — použi: xdg-open {out_path}")

    def _narr_open_dir(self):
        """Otvor output priečinok v file manageri."""
        out_path = Path(self._narr_out.text().strip())
        target_dir = out_path.parent if out_path.parent.exists() else out_path
        try:
            import subprocess as _sp
            _sp.Popen(["xdg-open", str(target_dir)],
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, start_new_session=True)
            self._narr_log_add(f"[OPEN] {target_dir}")
        except Exception as e:
            self._narr_log_add(f"[OPEN] zlyhalo ({e})")

    def _narr_stop(self):
        proc = getattr(self, "_narr_proc", None)
        if proc is None:
            return
        try:
            proc.terminate()
            self._narr_log_add("[INFO] Posielam terminate signál…")
        except Exception as e:
            self._narr_log_add(f"[CHYBA] terminate: {e}")


    # ══════════════════════════════════════════════════════════════════════════
    # Command building
    # ══════════════════════════════════════════════════════════════════════════
    def _apply_settings_visibility(self, *_) -> None:
        """Auto-hide neaktívne polia v Settings tabe podľa stavu prepínačov."""
        # Denoise preset — len keď je denoise zapnuté
        if hasattr(self, "_chk_denoise") and hasattr(self, "_denoise_preset"):
            on = self._chk_denoise.isChecked()
            for w in (self._denoise_preset, getattr(self, "_denoise_preset_lbl", None)):
                if w is not None: w.setVisible(on)
        # Text adaptation backend / ollama polia
        if hasattr(self, "_text_adapt_enabled") and hasattr(self, "_adapt_provider"):
            adapt_on = self._text_adapt_enabled.isChecked()
            is_ollama = self._adapt_provider.currentText() == "ollama"
            # Backend dropdown — viditeľný len keď text adapt ON
            for w in (self._adapt_provider, getattr(self, "_adapt_provider_lbl", None)):
                if w is not None: w.setVisible(adapt_on)
            # Ollama model + URL — len keď text adapt ON aj backend=ollama
            ollama_visible = adapt_on and is_ollama
            for w in (
                self._adapt_ollama_model, getattr(self, "_adapt_ollama_model_lbl", None),
                self._adapt_ollama_url,   getattr(self, "_adapt_ollama_url_lbl",   None),
            ):
                if w is not None: w.setVisible(ollama_visible)

    def _on_models_skip_toggled(self, state):
        """Settings checkbox → uloží preferenciu (auto-save)."""
        from PyQt6.QtCore import Qt as _Qt
        new_val = (state == _Qt.CheckState.Checked.value)
        self.settings.models_skip_dialog = new_val
        self.settings.save()
        self._log_add(f"[SETTINGS] Models dialog at startup: {'OFF (skip)' if new_val else 'ON'}")

    def _on_refine_engine_changed(self, txt: str):
        """Settings refine engine dropdown → save."""
        self.settings.refine_engine_choice = txt
        self.settings.save()
        self._log_add(f"[SETTINGS] Refine LLM: {txt}")

    def _on_default_tts_changed(self, txt: str):
        """Settings dropdown → save + sync Local page radio buttons live."""
        self.settings.def_tts_engine = txt
        self.settings.save()
        # Sync Local page radio buttons
        if txt == "omnivoice" and hasattr(self, "_rb_omnivoice"):
            self._rb_omnivoice.setChecked(True)
        elif txt == "chatterbox" and hasattr(self, "_rb_chatterbox"):
            self._rb_chatterbox.setChecked(True)
        self._update_engine_options_visibility()
        self._log_add(f"[SETTINGS] Default TTS engine: {txt} (uložené, aktívne aj teraz)")

    def _on_auto_detect_speaker(self):
        """Spustí auto_detect_speakers.py pre prvé video v batch + auto-set voice design."""
        # Najprv nájdi cieľové video
        if not self.batch_items:
            self._log_add("[DETECT] Žiadne video v zozname — pridaj video najprv.")
            return
        video_path = self.batch_items[0].path
        if not Path(video_path).exists():
            self._log_add(f"[DETECT] Video nenájdené: {video_path}")
            return

        # Disable Start button + auto-detect button počas detekcie
        self._start_btn.setEnabled(False)
        self._auto_detect_btn.setEnabled(False)
        self._auto_detect_btn.setText("⏳  Analyzujem...")
        self._log_add(f"[DETECT] Spúšťam pitch analýzu: {Path(video_path).name}")

        # Run v threade aby GUI nezamrzol
        from PyQt6.QtCore import QThread, pyqtSignal

        class _DetectThread(QThread):
            finished_signal = pyqtSignal(dict, str)

            def __init__(self, video, lang, py_exec):
                super().__init__()
                self.video = video
                self.lang = lang
                self.py_exec = py_exec

            def run(self):
                import subprocess as _sp
                script = BASE_DIR / "auto_detect_speakers.py"
                if not script.exists():
                    self.finished_signal.emit({}, f"Skript chýba: {script}")
                    return
                cmd = [self.py_exec, str(script), self.video, "--lang", self.lang, "--json"]
                try:
                    proc = _sp.run(cmd, capture_output=True, text=True, timeout=600)
                    if proc.returncode != 0:
                        self.finished_signal.emit({}, f"Detection zlyhala: {proc.stderr[-300:]}")
                        return
                    import json as _j
                    # Output má JSON ako poslednú časť (pred ním môžu byť log lines)
                    out = proc.stdout.strip()
                    # Find first { and last }
                    start = out.find('{')
                    end = out.rfind('}')
                    if start >= 0 and end > start:
                        result = _j.loads(out[start:end+1])
                        self.finished_signal.emit(result, "")
                    else:
                        self.finished_signal.emit({}, f"Žiadny JSON output: {out[-200:]}")
                except Exception as e:
                    self.finished_signal.emit({}, f"Exception: {e}")

        self._detect_thread = _DetectThread(
            video=video_path,
            lang=self._tgt_lang.currentText() or "sk",
            py_exec=self._effective_tts_python(),
        )
        self._detect_thread.finished_signal.connect(self._on_auto_detect_finished)
        self._detect_thread.start()

    def _on_auto_detect_finished(self, result: dict, error: str):
        """Callback po dokončení detection thread-u."""
        # Re-enable buttons
        self._start_btn.setEnabled(True)
        self._auto_detect_btn.setEnabled(True)
        self._auto_detect_btn.setText("🔍  Detekuj hovoriaceho")

        if error:
            self._log_add(f"[DETECT] CHYBA: {error}")
            return

        # Log výsledok
        self._log_add(
            f"[DETECT] Median F0={result.get('median_f0_hz',0)} Hz, "
            f"gender={result.get('gender','?').upper()}, "
            f"confidence={result.get('confidence','?').upper()}, "
            f"multispeaker={'YES' if result.get('multispeaker') else 'NO'}"
        )
        if result.get("alt_gender"):
            self._log_add(f"[DETECT]   Borderline — možno chceš {result['alt_gender']} namiesto {result['gender']}")

        # Auto-set Hlas WAV → "🎬 Cloning z videa" (lebo analyzujeme zdrojové video,
        # má zmysel klonovať jeho hlas + voice design hint pre prozódiu).
        # Nastavujeme OBA pickery (Chatterbox _cb_voice aj OmniVoice _ov_voice),
        # lebo user môže prepnúť engine kedykoľvek po auto-detect.
        cloning_label = _voice_disp("🎬 Cloning z videa")
        for picker_attr in ("_cb_voice", "_ov_voice"):
            picker = getattr(self, picker_attr, None)
            if picker is None:
                continue
            applied = False
            for i in range(picker.count()):
                if picker.itemText(i) == cloning_label:
                    picker.setCurrentText(cloning_label)
                    applied = True
                    break
            if applied:
                engine_name = "OmniVoice" if picker_attr == "_ov_voice" else "Chatterbox"
                self._log_add(f"[DETECT] ✓ Hlas WAV ({engine_name}): {cloning_label}")

        # TTS engine sa nemení — OmniVoice ostáva (vyššia kvalita).
        # Chatterbox sa použije IBA ako "ref-generator" v auto_clone_ref.py:
        # Chatterbox vyrobí SK-flavored ref WAV (nahradí raw video clone) a OmniVoice
        # potom robí finálnu syntézu na tento SK-flavored ref → SK akcent + OmniVoice quality.
        # Logika je v auto_clone_ref.py: ak je Chatterbox dostupný, použije sa automaticky.

        # Auto-set voice design v dropdowne
        suggestion = result.get("suggested_voice_design", "")
        if suggestion and hasattr(self, "_ov_instruct"):
            for i in range(self._ov_instruct.count()):
                preset = self._ov_instruct.itemText(i)
                if preset == suggestion or (preset and suggestion in preset):
                    self._ov_instruct.setCurrentText(preset)
                    self._log_add(f"[DETECT] ✓ Voice design nastavený: {preset}")
                    break
            else:
                self._log_add(f"[DETECT] Suggestion: {suggestion} (manuálne vyber v Voice design)")

        # Auto-set OmniVoice diffusion steps + guidance scale podľa detekovaných
        # podmienok (tech content / multispeaker / nízka confidence → vyššie hodnoty).
        sd_steps = result.get("suggested_diffusion_steps")
        sd_guid = result.get("suggested_guidance_scale")
        if sd_steps and hasattr(self, "_ov_num_step"):
            try:
                self._ov_num_step.setValue(int(sd_steps))
            except Exception:
                pass
        if sd_guid and hasattr(self, "_ov_guidance"):
            try:
                self._ov_guidance.setValue(float(sd_guid))
            except Exception:
                pass
        if sd_steps or sd_guid:
            self._log_add(f"[DETECT] ✓ OmniVoice: diffusion_steps={sd_steps}, guidance={sd_guid}")

        # Accent ref/strength auto-detection odstránená — Chatterbox-as-ref-generator
        # v auto_clone_ref.py rieši SK akcent automaticky (Chatterbox SK 2.2 model).

        # Multi-voice info: vždy zalogujeme detekciu, ale NEZAPNEME checkbox
        # automaticky — užívateľ rozhodne či chce multi-voice použiť.
        multi = bool(result.get("multispeaker"))
        n_est = int(result.get("speaker_count_estimated") or 1)
        peaks = result.get("peaks_hz", [])
        if multi and n_est >= 2:
            checkbox_status = ""
            if hasattr(self, "_ov_multi_voice"):
                checkbox_status = (" (multi-voice ZAPNUTÉ — bude použité)"
                                   if self._ov_multi_voice.isChecked()
                                   else " (multi-voice VYPNUTÉ — zaškrtni '🎭 Multi-voice' pre per-speaker clone)")
            peaks_str = f", peaks: {peaks} Hz" if peaks else ""
            self._log_add(f"[DETECT] Multispeaker: ~{n_est} speakerov{peaks_str}{checkbox_status}")
        else:
            self._log_add(f"[DETECT] Single speaker (multi-voice nie je potrebný)")

        # Auto-set Typ obsahu podľa Whisper sample + keyword analýzy
        ct_val = result.get("suggested_content_type", "")
        ct_conf = result.get("content_confidence", "low")
        if ct_val and hasattr(self, "_content_grp"):
            applied = False
            for btn in self._content_grp.buttons():
                if btn.property("val") == ct_val:
                    btn.setChecked(True)
                    applied = True
                    break
            scores = result.get("content_scores", {})
            top_hits = ""
            if isinstance(scores, dict) and scores:
                top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
                top_hits = " (" + ", ".join(f"{k}={v}" for k, v in top3 if v > 0) + ")"
            if applied:
                self._log_add(f"[DETECT] ✓ Typ obsahu: {ct_val.upper()} [{ct_conf}]{top_hits}")
            else:
                self._log_add(f"[DETECT] Suggested content_type: {ct_val} (radio nenájdené)")

    def _refresh_voice_design_presets(self, *_):
        """Rebuild voice design dropdown podľa aktuálneho cieľového jazyka.
        SK → SK presety + GB britský; CZ → CZ presety + GB britský."""
        if not hasattr(self, "_ov_instruct") or not hasattr(self, "_tgt_lang"):
            return
        cur = self._ov_instruct.currentText()
        lang = self._tgt_lang.currentText() if self._tgt_lang else "sk"
        new_items = _voice_design_presets_for(lang)
        self._ov_instruct.blockSignals(True)
        self._ov_instruct.clear()
        for preset in new_items:
            self._ov_instruct.addItem(preset)
        # Try preserve current selection ak je v novom liste
        if cur in new_items:
            self._ov_instruct.setCurrentText(cur)
        else:
            self._ov_instruct.setCurrentIndex(0)
        self._ov_instruct.blockSignals(False)

    def _update_engine_options_visibility(self):
        """Zobrazí/skryje checkboxy v Možnosti podľa zvoleného TTS engine.
        OmniVoice-only: stop_after_translate
        Chatterbox-only: multi_voice, refine
        Common (visible vždy): keep_music, voice_eq, srt_align, mini_level11, mixed_technical_terms
        """
        is_ov = self._tts_is_omnivoice() if hasattr(self, '_tts_is_omnivoice') else False
        for attr in getattr(self, '_opt_omnivoice_only', []):
            cb = getattr(self, attr, None)
            if cb is not None: cb.setVisible(is_ov)
        for attr in getattr(self, '_opt_chatterbox_only', []):
            cb = getattr(self, attr, None)
            if cb is not None: cb.setVisible(not is_ov)
        # Multi-voice n spinner — viditeľný iba ak Chatterbox + multi-voice checked
        if hasattr(self, '_multi_voice_n_lbl') and hasattr(self, '_chk_multi_voice'):
            show_mv_n = (not is_ov) and self._chk_multi_voice.isChecked()
            self._multi_voice_n_lbl.setVisible(show_mv_n)
            self._multi_voice_n.setVisible(show_mv_n)

    def _build_menu_bar(self) -> None:
        """Top menu: Súbor / Nástroje / Pomoc."""
        mb = self.menuBar()
        # ── Súbor ─────────────────────────────────────────────────────────
        m_file = mb.addMenu("&Súbor")
        a_quit = QAction("Ukončiť", self)
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)
        # ── Nástroje ──────────────────────────────────────────────────────
        m_tools = mb.addMenu("&Nástroje")
        a_models = QAction("Spravovať modely…", self)
        a_models.setStatusTip("Stiahnuť / aktualizovať modely z HuggingFace")
        a_models.triggered.connect(self._on_manage_models_clicked)
        m_tools.addAction(a_models)
        a_vram = QAction("Uvoľniť VRAM", self)
        a_vram.setStatusTip("Unload Ollama + LM Studio modely + zabiť orphan procesy")
        a_vram.triggered.connect(self._on_release_vram_clicked)
        m_tools.addAction(a_vram)
        m_tools.addSeparator()
        a_paths = QAction("Cesty (paths.json)…", self)
        a_paths.setStatusTip("Otvor settings/paths.json v editore")
        a_paths.triggered.connect(self._open_paths_json)
        m_tools.addAction(a_paths)
        # ── Pomoc ─────────────────────────────────────────────────────────
        m_help = mb.addMenu("&Pomoc")
        a_about = QAction("O programe…", self)
        a_about.triggered.connect(self._show_about)
        m_help.addAction(a_about)
        a_github = QAction("GitHub repo", self)
        a_github.triggered.connect(lambda: __import__("webbrowser").open("https://github.com/Vojtech-Pet/videotranslator-studio"))
        m_help.addAction(a_github)

    def _open_paths_json(self):
        from scripts.paths import CONFIG_FILE
        if not CONFIG_FILE.exists():
            from scripts.paths import save_paths, load_paths
            save_paths(load_paths())
        import subprocess as _sp
        for editor in ("xdg-open", "gnome-open", "nano"):
            try:
                _sp.Popen([editor, str(CONFIG_FILE)])
                return
            except FileNotFoundError:
                continue
        QMessageBox.information(self, "Cesta k config", str(CONFIG_FILE))

    def _show_about(self):
        QMessageBox.about(
            self, "O programe",
            "<h2>VideoTranslator Studio v2.0</h2>"
            "<p>Automatický EN→SK video-dubbing pipeline.</p>"
            "<p>STT (Whisper) → preklad (Gemma) → TTS (OmniVoice) → finalizácia.</p>"
            "<hr>"
            "<p><b>Web:</b> <a href='https://vojtech-pet.github.io/videotranslator-studio'>"
            "vojtech-pet.github.io/videotranslator-studio</a><br>"
            "<b>GitHub:</b> <a href='https://github.com/Vojtech-Pet/videotranslator-studio'>"
            "Vojtech-Pet/videotranslator-studio</a><br>"
            "<b>HuggingFace:</b> <a href='https://huggingface.co/pekiskol'>huggingface.co/pekiskol</a></p>"
            "<p>© 2026 Vojtech Petrik · MIT License</p>"
        )

    def _on_manage_models_clicked(self):
        """Manual trigger pre model manager dialog (Tools tab)."""
        self._show_models_dialog(force=True)

    def _check_models_first_run(self) -> None:
        """Wrapper — automaticky show dialog len ak essential modely chýbajú."""
        self._show_models_dialog(force=False)

    def _show_models_dialog(self, force: bool = False) -> None:
        """Show model bootstrap dialog.

        force=False: pri štarte — ukáž LEN keď essential modely chýbajú
        force=True:  z Tools → Spravovať modely — vždy ukáž celý zoznam
        """
        try:
            from scripts.model_bootstrap import (
                missing_models, is_model_present, models_disk_path, REQUIRED_MODELS,
                total_download_size_gb, disk_space_gb, bootstrap_models,
            )
        except ImportError:
            if force:
                QMessageBox.warning(self, "Chyba",
                    "huggingface_hub modul chýba.\nNainštaluj: pip install huggingface_hub")
            return
        models_dir = models_disk_path()
        missing_essential = missing_models(models_dir, only_essential=True)
        # Auto-mode (force=False): zobrazi sa len ak essential chýbajú a user nezakliknul "Nezobrazovať"
        if not force:
            if not missing_essential:
                return
            if getattr(self.settings, "models_skip_dialog", False):
                # User predtým zaškrtnol "nezobrazovať" — len log warn, dialog vynechaj
                self._log_add(f"[MODELS] {len(missing_essential)} essential modelov chýba, "
                              f"ale dialog je v Settings vypnutý. Otvor cez Nástroje → Spravovať modely.")
                return
        # Build dialog
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QProgressBar, QLabel
        dlg = QDialog(self)
        dlg.setWindowTitle("Spravovať modely" if force else "Prvé spustenie — modely")
        dlg.setMinimumWidth(580)
        v = QVBoxLayout(dlg)
        free_gb = disk_space_gb(models_dir)
        if force:
            header = "<b>Modely VideoTranslator Studio</b><br>"
        else:
            header = f"<b>Treba stiahnuť {len(missing_essential)} povinný/é modely</b> z HuggingFace.<br>"
        info = QLabel(
            f"{header}"
            f"Cieľ: <code>{models_dir}</code><br>"
            f"Voľné miesto: {free_gb:.1f} GB"
        )
        info.setWordWrap(True)
        v.addWidget(info)
        # V force mode ukáž všetky modely; v auto mode iba chýbajúce
        if force:
            display_specs = list(REQUIRED_MODELS)
        else:
            missing_optional = [m for m in missing_models(models_dir) if not m.essential]
            display_specs = missing_essential + missing_optional
        checkboxes = {}
        for spec in display_specs:
            present = is_model_present(spec, models_dir)
            label = f"{spec.description}  —  {spec.size_gb:.1f} GB"
            if present:
                label += "  ✓ stiahnuté"
            box = QCheckBox(label)
            # Auto-mode: essential ktoré chýbajú = checked + locked
            # Force mode: nič nepredvolené (user vyberie)
            if not force and spec.essential and not present:
                box.setChecked(True)
                box.setEnabled(False)
            else:
                box.setChecked(False)
            box.setToolTip(f"Repo: {spec.repo_id}\n" +
                           ("Už stiahnuté — zaškrtni iba ak chceš re-download" if present
                            else "Stiahnuť teraz"))
            checkboxes[spec.id] = (box, spec)
            v.addWidget(box)
        size_lbl = QLabel()
        v.addWidget(size_lbl)
        def _update_size():
            total = sum(spec.size_gb for box, spec in checkboxes.values() if box.isChecked())
            size_lbl.setText(f"<b>Spolu:</b> {total:.1f} GB")
        for box, _ in checkboxes.values():
            box.stateChanged.connect(lambda *_: _update_size())
        _update_size()
        # "Nezobrazovať pri ďalšom spustení" checkbox (iba v auto-mode, force=True ho nepotrebuje)
        skip_box = None
        if not force:
            skip_box = QCheckBox("Pri ďalšom spustení nezobrazovať tento dialog (môžeš ho vyvolať z Nástroje → Spravovať modely)")
            skip_box.setChecked(getattr(self.settings, "models_skip_dialog", False))
            v.addWidget(skip_box)
        # Buttons
        btn_row = QHBoxLayout()
        b_skip = QPushButton("Preskočiť (manuálne)")
        b_download = QPushButton("Stiahnuť")
        b_download.setDefault(True)
        btn_row.addStretch(1); btn_row.addWidget(b_skip); btn_row.addWidget(b_download)
        v.addLayout(btn_row)
        # Progress (skrytý dokým download nezačne)
        progress_lbl = QLabel(""); progress_lbl.setVisible(False)
        progress_bar = QProgressBar(); progress_bar.setVisible(False); progress_bar.setRange(0, 100)
        v.addWidget(progress_lbl); v.addWidget(progress_bar)
        # Helper — uloží skip preferenciu pri zatvorení (oboma cestami)
        def _save_skip_pref():
            if skip_box is not None and skip_box.isChecked() != getattr(self.settings, "models_skip_dialog", False):
                self.settings.models_skip_dialog = skip_box.isChecked()
                self.settings.save()
        # Handlers
        b_skip.clicked.connect(lambda: (_save_skip_pref(), dlg.reject()))
        def _on_download():
            selected = [spec for box, spec in checkboxes.values() if box.isChecked()]
            if not selected:
                dlg.reject(); return
            for box, _ in checkboxes.values():
                box.setEnabled(False)
            b_download.setEnabled(False); b_skip.setEnabled(False)
            progress_lbl.setVisible(True); progress_bar.setVisible(True)
            from PyQt6.QtCore import QCoreApplication
            def cb(mid, pct):
                progress_lbl.setText(f"Sťahujem: {mid}")
                progress_bar.setValue(max(0, pct))
                QCoreApplication.processEvents()
            results = bootstrap_models(models_dir, only=selected, progress_callback=cb)
            failed = [m for m, s in results.items() if not s.startswith("ok")]
            if failed:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(dlg, "Niektoré modely zlyhali",
                    "\n".join(f"{m}: {results[m]}" for m in failed))
            _save_skip_pref()
            dlg.accept()
        b_download.clicked.connect(_on_download)
        # Aj pri zatvorení X (rejected) ulož preferenciu
        dlg.finished.connect(lambda _: _save_skip_pref())
        dlg.exec()

    def _check_bootstrap_paths(self) -> None:
        """Pri prvom štarte over esenciálne cesty. Ak chýbajú, zobraz dialog."""
        validation = PATHS.validate()
        essential = ("miniforge_root", "models_root", "knihy_root")
        missing = [k for k in essential if not validation.get(k, False)]
        if not missing:
            return
        from scripts.paths import CONFIG_FILE
        msg = (
            "<b>Niektoré esenciálne cesty neexistujú:</b><br><br>"
            + "<br>".join(f"  • <b>{k}</b>: <code>{getattr(PATHS, k)}</code>" for k in missing)
            + "<br><br>Aplikácia bude pokračovať, ale niektoré funkcie nemusia fungovať.<br>"
            + "Cesty môžeš opraviť v <b>Nastavenia → Modely a koreňové cesty</b>.<br>"
            + f"Konfigurácia sa uloží do:<br><code>{CONFIG_FILE}</code>"
        )
        QMessageBox.warning(self, "Bootstrap — chýbajúce cesty", msg)

    def _save_paths_config(self) -> None:
        """Uloží paths.py config do ~/.config/videotranslator/paths.json."""
        from scripts.paths import Paths, save_paths, load_paths
        new_paths = Paths(
            miniforge_root=self._path_miniforge.text().strip() or PATHS.miniforge_root,
            models_root=self._path_models.text().strip() or PATHS.models_root,
            knihy_root=self._path_knihy.text().strip() or PATHS.knihy_root,
            ollama_bin=self._path_ollama.text().strip() or PATHS.ollama_bin,
            lmstudio_bin=PATHS.lmstudio_bin,
            fish_legacy_root=PATHS.fish_legacy_root,
            dataset_root=PATHS.dataset_root,
        )
        save_paths(new_paths)
        # Validácia
        problems = [k for k, v in new_paths.validate().items()
                    if not v and k in ("miniforge_root", "models_root", "knihy_root")]
        if problems:
            QMessageBox.warning(
                self, "Cesty uložené",
                f"Cesty boli uložené do ~/.config/videotranslator/paths.json,\n"
                f"ale tieto neexistujú:\n  - " + "\n  - ".join(problems) +
                "\n\nReštartuj aplikáciu po oprave."
            )
        else:
            QMessageBox.information(
                self, "Cesty uložené",
                "Cesty boli úspešne uložené.\n"
                "Reštartuj aplikáciu aby sa zmeny aplikovali."
            )

    def _ensure_ollama_running(self, timeout: int = 30) -> bool:
        """Skontroluj či ollama beží na :11434; ak nie, spusť ju a počkaj."""
        import urllib.request, urllib.error
        url = "http://localhost:11434/api/tags"
        def _alive() -> bool:
            try:
                urllib.request.urlopen(url, timeout=1.5)
                return True
            except (urllib.error.URLError, OSError, TimeoutError):
                return False
        if _alive():
            return True
        self._log_add("[OLLAMA] Server nebeží — spúšťam...")
        ollama_bin = str(PATHS.ollama_bin)
        if not Path(ollama_bin).exists():
            ollama_bin = shutil.which("ollama") or "ollama"
        try:
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as e:
            self._log_add(f"[OLLAMA] Spustenie zlyhalo: {e}")
            return False
        t0 = time.time()
        while time.time() - t0 < timeout:
            QApplication.processEvents()
            if _alive():
                self._log_add(f"[OLLAMA] Server pripravený ({time.time()-t0:.1f}s)")
                return True
            time.sleep(0.5)
        self._log_add(f"[OLLAMA] Timeout {timeout}s — server neodpovedal")
        return False

    def _on_ui_mode_changed(self, _txt: str = "") -> None:
        mode = "extended" if self._ui_mode_cb.currentIndex() == 1 else "normal"
        self.settings.ui_mode = mode
        self.settings.save()
        self._apply_ui_mode(mode)

    def _apply_ui_mode(self, mode: str) -> None:
        is_normal = (mode == "normal")
        # Skupiny / sekcie (Normal = skryť) — _grp_ct (Typ obsahu) je vždy viditeľné,
        # lebo sa týka prekladu pre VŠETKY engines (LMStudio, Gemma, OmniVoice path).
        # User požiadal aby zostalo viditeľné aj v normal móde.
        # Možnosti — checkboxy ktoré sa v Normal móde skryjú (po public-cleanup 2026-05-06)
        normal_hidden_chks = (
            "_chk_keep_music", "_chk_refine",
            "_chk_voice_eq", "_chk_srt_align",
            "_chk_mini_level11", "_chk_mixed_technical_terms",
            "_cb_emo_clone",
            "_vocal_sep", "_eq_profile", "_eq_profile_lbl",
        )
        for w_name in normal_hidden_chks:
            w = getattr(self, w_name, None)
            if w is not None: w.setVisible(not is_normal)

    def _on_engine_changed(self, engine: str):
        # Model dropdowny už nie sú v hlavnom layoute — len refresh contentu pre Settings sync.
        if hasattr(self, "_madlad_model"):
            self._refresh_tgemma_items(prefer_engine=engine)
        # Sync items do Settings dropdownu (ak existuje, t.j. _page_settings už beží)
        if hasattr(self, "_set_gemma_model") and hasattr(self, "_madlad_model"):
            items = [self._madlad_model.itemText(i) for i in range(self._madlad_model.count())]
            cur = self._set_gemma_model.currentText()
            self._set_gemma_model.blockSignals(True)
            self._set_gemma_model.clear()
            self._set_gemma_model.addItems(items)
            if cur in items:
                self._set_gemma_model.setCurrentText(cur)
            elif items:
                self._set_gemma_model.setCurrentIndex(0)
            self._set_gemma_model.blockSignals(False)

    def _sync_gemma_model_from_settings(self, txt: str):
        """Settings dropdown → main page widget (zachová single source of truth)."""
        if not hasattr(self, "_madlad_model"):
            return
        if self._madlad_model.currentText() == txt:
            return
        idx = self._madlad_model.findText(txt)
        if idx >= 0:
            self._madlad_model.setCurrentIndex(idx)
        else:
            self._madlad_model.setEditText(txt)

    def _current_tgemma_list(self, engine: Optional[str] = None) -> List[str]:
        eng = engine if engine is not None else self._trans_engine.currentText()
        return self.settings.tgemma_models_26b if eng == "gemma" else self.settings.tgemma_models_e4b

    def _set_current_tgemma_list(self, items: List[str], engine: Optional[str] = None) -> None:
        eng = engine if engine is not None else self._trans_engine.currentText()
        if eng == "gemma":
            self.settings.tgemma_models_26b = items
        else:
            self.settings.tgemma_models_e4b = items
        self.settings.save()

    def _refresh_tgemma_items(self, prefer_engine: Optional[str] = None) -> None:
        target_items = self._current_tgemma_list(prefer_engine)
        current = self._madlad_model.currentText()
        existing_items = [self._madlad_model.itemText(i) for i in range(self._madlad_model.count())]
        if existing_items != target_items:
            self._madlad_model.blockSignals(True)
            self._madlad_model.clear()
            self._madlad_model.addItems(target_items)
            if current in target_items:
                self._madlad_model.setCurrentText(current)
            elif target_items:
                self._madlad_model.setCurrentIndex(0)
            self._madlad_model.blockSignals(False)

    def _on_tgemma_add(self) -> None:
        eng = self._trans_engine.currentText()
        if eng == "gemma":
            # Ollama model name — textový vstup
            text, ok = QInputDialog.getText(
                self, "Pridať Ollama model",
                "Názov ollama modelu (napr. moj-model:latest):"
            )
            if not ok or not text.strip(): return
            path = text.strip()
        else:
            # GGUF cesta — file dialog
            cur = self._madlad_model.currentText().strip()
            start_dir = ""
            if cur:
                p = Path(cur)
                if p.parent.exists(): start_dir = str(p.parent)
            if not start_dir:
                for cand in (PATHS.knihy_root, PATHS.models_root):
                    if Path(cand).exists(): start_dir = cand; break
            path, _ = QFileDialog.getOpenFileName(self, "Vyber GGUF model", start_dir, "GGUF (*.gguf);;Všetky súbory (*)")
            if not path: return
        items = list(self._current_tgemma_list())
        if path not in items:
            items.append(path)
            self._set_current_tgemma_list(items)
        self._refresh_tgemma_items()
        self._madlad_model.setCurrentText(path)

    def _on_tgemma_del(self) -> None:
        cur = self._madlad_model.currentText().strip()
        if not cur: return
        items = list(self._current_tgemma_list())
        if cur not in items: return
        items.remove(cur)
        self._set_current_tgemma_list(items)
        self._refresh_tgemma_items()

    def _on_tgemma_text_edited(self) -> None:
        # Po Enter / focus-out ulož pridanú/upravenú položku do zoznamu
        txt = self._madlad_model.currentText().strip()
        if not txt: return
        items = list(self._current_tgemma_list())
        if txt in items: return
        items.append(txt)
        self._set_current_tgemma_list(items)
        self._refresh_tgemma_items()
        self._madlad_model.setCurrentText(txt)

    # ══════════════════════════════════════════════════════════════════════════
    # Profile management
    # ══════════════════════════════════════════════════════════════════════════
    def _profile_path(self, name: str) -> Path:
        """JSON cesta pre pomenovaný profil."""
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("._")
        return PROFILES_DIR / f"{safe or 'unnamed'}.json"

    def _profile_list_names(self) -> list[str]:
        """Zoznam dostupných profilov (bez .json)."""
        if not PROFILES_DIR.exists():
            return []
        return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))

    def _refresh_profile_list(self) -> None:
        """Refresh dropdown z PROFILES_DIR."""
        if not hasattr(self, "_profile_combo"):
            return
        names = self._profile_list_names()
        current = self._profile_combo.currentText()
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        self._profile_combo.addItems(names if names else ["(žiadne profily)"])
        if current in names:
            self._profile_combo.setCurrentText(current)
        self._profile_combo.blockSignals(False)

    def _profile_save(self, name: str) -> bool:
        """Uloží aktuálne self.settings ako profil pod názvom name."""
        try:
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
            self._save_chk_state()  # sync widgets → settings
            data = self.settings._secret_free_dict()
            self._profile_path(name).write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            return True
        except Exception as e:
            self._log_add(f"[PROFILE] Save zlyhal: {e}")
            return False

    def _profile_load(self, name: str) -> bool:
        """Načíta profil → self.settings + uloží + signalizuje restart."""
        try:
            p = self._profile_path(name)
            if not p.exists():
                self._log_add(f"[PROFILE] Profil neexistuje: {name}")
                return False
            d = json.loads(p.read_text(encoding="utf-8"))
            for k in d:
                if hasattr(self.settings, k):
                    setattr(self.settings, k, d[k])
            self.settings.save()
            return True
        except Exception as e:
            self._log_add(f"[PROFILE] Load zlyhal: {e}")
            return False

    def _on_profile_save_as(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Uložiť profil",
            "Názov profilu (a-z, 0-9, _, -):")
        if not ok or not name.strip(): return
        if self._profile_save(name.strip()):
            self._refresh_profile_list()
            self._profile_combo.setCurrentText(name.strip())
            self._log_add(f"[PROFILE] Uložené: {name}")

    def _on_profile_load(self) -> None:
        name = self._profile_combo.currentText()
        if not name or name.startswith("(žiadne"):
            self._log_add("[PROFILE] Žiadny profil vybraný")
            return
        if self._profile_load(name):
            QMessageBox.information(self, "Profil",
                f"Profil '{name}' načítaný.\n\nReštartuj GUI pre úplné aplikovanie zmien.")
            self._log_add(f"[PROFILE] Načítaný: {name} (reštartuj GUI)")

    def _on_profile_delete(self) -> None:
        name = self._profile_combo.currentText()
        if not name or name.startswith("(žiadne"): return
        r = QMessageBox.question(self, "Vymazať profil",
            f"Naozaj vymazať profil '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes: return
        try:
            self._profile_path(name).unlink(missing_ok=True)
            self._refresh_profile_list()
            self._log_add(f"[PROFILE] Vymazané: {name}")
        except Exception as e:
            self._log_add(f"[PROFILE] Delete zlyhal: {e}")

    def _on_set_normal_default(self) -> None:
        """Uloží aktuálne nastavenia ako Normal mode default."""
        if self._profile_save(NORMAL_DEFAULT_PROFILE):
            self._refresh_profile_list()
            QMessageBox.information(self, "Normal default",
                "Aktuálne nastavenia uložené ako Normal default.\n"
                "Pri každom prepnutí UI módu na Normal sa obnovia.")
            self._log_add(f"[PROFILE] Normal default nastavený")

    def _on_factory_reset(self) -> None:
        r = QMessageBox.question(self, "Factory reset",
            "Naozaj obnoviť VŠETKY nastavenia na pôvodné továrenské?\n"
            "Profily zostanú zachované.\n\nVyžaduje reštart GUI.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes: return
        try:
            CONFIG_FILE.unlink(missing_ok=True)
            QMessageBox.information(self, "Factory reset",
                "Nastavenia resetované. Reštartuj GUI.")
            self._log_add("[PROFILE] Factory reset — reštartuj GUI")
        except Exception as e:
            self._log_add(f"[PROFILE] Factory reset zlyhal: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Custom TTS engines (user-managed in Settings)
    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_custom_tts_combo(self) -> None:
        """Aktualizuje dropdown v Settings."""
        if not hasattr(self, "_custom_tts_combo"): return
        names = [e.get("name","") for e in self.settings.custom_tts_engines if e.get("name")]
        current = self._custom_tts_combo.currentText()
        self._custom_tts_combo.blockSignals(True)
        self._custom_tts_combo.clear()
        self._custom_tts_combo.addItems(names if names else ["(žiadne)"])
        if current in names:
            self._custom_tts_combo.setCurrentText(current)
        self._custom_tts_combo.blockSignals(False)
        self._on_custom_tts_selected(self._custom_tts_combo.currentText())

    def _on_custom_tts_selected(self, name: str) -> None:
        """Po výbere ukáž aktuálny typ + cestu."""
        if not hasattr(self, "_custom_tts_type"): return
        if name and not name.startswith("(žiadne"):
            entry = next((e for e in self.settings.custom_tts_engines if e.get("name") == name), None)
            if entry:
                t = entry.get("type", "chatterbox")
                p = entry.get("path", "")
                self._custom_tts_type.blockSignals(True); self._custom_tts_type.setCurrentText(t); self._custom_tts_type.blockSignals(False)
                self._custom_tts_path.blockSignals(True); self._custom_tts_path.setText(p); self._custom_tts_path.blockSignals(False)
                return
        self._custom_tts_path.setText("")

    def _on_custom_tts_add(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Pridať TTS engine",
            "Názov nového custom TTS enginu:")
        if not ok or not name.strip(): return
        name = name.strip()
        if any(e.get("name") == name for e in self.settings.custom_tts_engines):
            QMessageBox.warning(self, "Pridať", f"Engine s názvom '{name}' už existuje."); return
        # Default values
        self.settings.custom_tts_engines.append(
            {"name": name, "type": "chatterbox", "path": ""})
        self.settings.save()
        self._refresh_custom_tts_combo()
        self._refresh_main_tts_combo()
        self._custom_tts_combo.setCurrentText(name)

    def _on_custom_tts_del(self) -> None:
        name = self._custom_tts_combo.currentText()
        if not name or name.startswith("(žiadne"): return
        r = QMessageBox.question(self, "Odstrániť", f"Vymazať custom TTS engine '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes: return
        self.settings.custom_tts_engines = [
            e for e in self.settings.custom_tts_engines if e.get("name") != name]
        self.settings.save()
        self._refresh_custom_tts_combo()
        self._refresh_main_tts_combo()

    def _on_custom_tts_browse(self) -> None:
        cur = self._custom_tts_path.text().strip()
        path, _ = QFileDialog.getOpenFileName(self, "Cesta k modelu",
            cur or "", "Modely (*.pt *.pth *.safetensors *.gguf);;Všetko (*)")
        if path:
            self._custom_tts_path.setText(path)
            self._on_custom_tts_field_edit()

    def _on_custom_tts_field_edit(self) -> None:
        """Po zmene typu/cesty ulož do entry."""
        name = self._custom_tts_combo.currentText()
        if not name or name.startswith("(žiadne"): return
        for e in self.settings.custom_tts_engines:
            if e.get("name") == name:
                e["type"] = self._custom_tts_type.currentText()
                e["path"] = self._custom_tts_path.text().strip()
                break
        self.settings.save()

    def _refresh_main_tts_combo(self) -> None:
        """Refresh hlavného TTS engine dropdown — pridaj custom enginy."""
        if not hasattr(self, "_set_tts"): return
        builtins = ["chatterbox", "omnivoice"]
        custom = [e.get("name","") for e in self.settings.custom_tts_engines if e.get("name")]
        all_engines = builtins + custom
        current = self._set_tts.currentText()
        self._set_tts.blockSignals(True)
        self._set_tts.clear()
        self._set_tts.addItems(all_engines)
        if current in all_engines:
            self._set_tts.setCurrentText(current)
        self._set_tts.blockSignals(False)

    def _resolve_tts_engine(self, engine_name: str) -> tuple[str, str]:
        """Vráti (built_in_type, override_path) pre engine name z dropdownu."""
        builtins = {"chatterbox", "omnivoice"}
        if engine_name in builtins:
            return engine_name, ""
        entry = next((e for e in self.settings.custom_tts_engines if e.get("name") == engine_name), None)
        if entry:
            return entry.get("type", "chatterbox"), entry.get("path", "")
        return "chatterbox", ""  # fallback

    def _on_whisper_model_edited(self) -> None:
        """Po Enter / focus-out ulož pridaný/upravený whisper model do zoznamu."""
        txt = self._set_whisper.currentText().strip()
        if not txt: return
        if txt not in self.settings.whisper_models:
            self.settings.whisper_models.append(txt)
            # Aktualizuj dropdown bez straty current
            current_items = [self._set_whisper.itemText(i) for i in range(self._set_whisper.count())]
            if txt not in current_items:
                self._set_whisper.addItem(txt)
        self.settings.whisper_model = txt
        self.settings.save()

    def _on_refresh_ollama_models(self) -> None:
        """Stiahne zoznam nainštalovaných ollama modelov a doplní ich do dropdownu.

        Vlákna prevažne v engine=gemma (ollama backend). Filtruje na translator-relevantné
        názvy (translator/gemma/translate/26b/g3-12b/sk-).
        """
        engine = self._trans_engine.currentText()
        if engine != "gemma":
            self._log_add("[OLLAMA] Refresh sa použije len pre engine=gemma — prepni najprv engine.")
            return
        try:
            import urllib.request as _ur, json as _json
            with _ur.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
                data = _json.loads(r.read().decode("utf-8"))
            all_names = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception as e:
            self._log_add(f"[OLLAMA] Refresh zlyhal — beží ollama? ({e})")
            return
        if not all_names:
            self._log_add("[OLLAMA] Žiadne modely nájdené (ollama list je prázdny)")
            return
        # Filter na translator-relevant
        keywords = ("translator","gemma","translate","26b","g3-12b","sk-","mistral","qwen","llama")
        relevant = [n for n in all_names if any(k in n.lower() for k in keywords)]
        if not relevant:
            relevant = all_names  # fallback: ukáž všetky
        # Zaradiť g3-12b-translator-v1 na vrchol ak je tam
        relevant = sorted(set(relevant), key=lambda n: (0 if "g3-12b-translator" in n else 1, n))
        # Update settings + dropdown
        self.settings.tgemma_models_26b = relevant
        current = self._madlad_model.currentText()
        self._madlad_model.blockSignals(True)
        self._madlad_model.clear()
        self._madlad_model.addItems(relevant)
        if current in relevant:
            self._madlad_model.setCurrentText(current)
        else:
            self._madlad_model.setCurrentIndex(0)
        self._madlad_model.blockSignals(False)
        self.settings.save()
        self._log_add(f"[OLLAMA] Načítaných {len(relevant)} modelov: {', '.join(relevant[:5])}{'...' if len(relevant)>5 else ''}")

    def _build_trans_args(self, engine: str, tgemma_model: str) -> tuple[List[str], Optional[Dict]]:
        env = None
        if engine == "google":
            # --no_adapt_llm: zakáže Gemma-27B pre text adaptation (skracuje text na nezmysel)
            return ["--use_google_translate", "--no_adapt_llm"], env
        if engine == "madlad":
            # Text adaptation nechávame aktívnu: Gemma 12B sa pridá neskôr v _build_tts_args().
            return ["--use_madlad", "--madlad_model", "google/madlad400-7b-mt"], env
        if engine == "multislav5lang":
            # Text adaptation nechávame aktívnu: Gemma 12B sa pridá neskôr v _build_tts_args().
            return ["--use_multislav5lang"], env
        if engine == "translategemma":
            return ["--use_translategemma_fulldoc","--translategemma_model",tgemma_model,"--translategemma_gpu_layers","-1","--no_adapt_llm"], env
        if engine == "llama":
            return ["--llama_model",str(PATHS.gemma3_12b_q8),"--llama_gpu_layers","-1"], env
        if engine == "gemma":
            # Gemma engine — dispatch podľa konkrétneho modelu (12B v1 alebo 26B legacy)
            model_name = (tgemma_model or "").strip()
            if not model_name or model_name.endswith(".gguf"):
                model_name = "g3-12b-translator-v1:latest"  # default = nový lepší v1
            # G3 27B multitask v1 — single-step (translate + length-aware), nahradí 12B
            if "g3-27b-multitask" in model_name:
                bare = model_name.replace(":latest", "")
                return ["--use_g3_27b_translator",
                        "--g3_27b_translator_model", bare,
                        "--no_adapt_llm"], env
            # G3 12B v1 — QLoRA fine-tuned, dataset 55 810 párov, 5× lepšia length presnosť
            # Length adjuster auto-on (kompatibilný s translator-om, 26B compress má HTML leak)
            if "g3-12b-translator" in model_name:
                bare = model_name.replace(":latest", "")
                return ["--use_g3_v1_translator",
                        "--g3_v1_translator_model", bare,
                        "--use_g3_v1_length_adjuster",
                        "--no_adapt_llm"], env
            # 26B legacy fine-tune — môže obsahovať HTML leak (OPUS-100 kontaminácia)
            # Fallback na llama-cpp-python ak ollama nebeží/zlyhala (iba pre 26B varianty)
            if not getattr(self, "_ollama_ok", True):
                gguf_path = PATHS.ollama_to_gguf(model_name) or PATHS._26b_q5km_ft
                gguf = str(gguf_path)
                ctx = "4096" if "q6" in Path(gguf).name.lower() else "8192"
                return ["--llama_model", gguf, "--llama_gpu_layers", "-1", "--llama_ctx", ctx], env
            return [
                "--use_ollama_translate",
                "--ollama_translate_model", model_name,
                "--ollama_translate_url", "http://localhost:11434/api/generate",
                "--ollama_translate_timeout", "180",
                "--no_adapt_llm",
            ], env
        if engine == "lmstudio":
            lms_url   = self._lmstudio_url.text().strip()   if hasattr(self, "_lmstudio_url")   else "http://localhost:1234/v1"
            lms_model = self._lmstudio_model.text().strip() if hasattr(self, "_lmstudio_model") else ""
            args = ["--use_lmstudio_translate",
                    "--lmstudio_translate_url",   lms_url   or "http://localhost:1234/v1",
                    "--lmstudio_translate_timeout", "180",
                    "--no_adapt_llm"]
            if lms_model:
                args += ["--lmstudio_translate_model", lms_model]
            # Built-in llama-cpp-server alternative (open source, súčasť VTS)
            if getattr(self.settings, "use_llamacpp_server", True):
                args.append("--use_llamacpp_server")
            return args, env
        if engine == "hybrid":
            # Gemma 4 E4B (HF) preloží → Gemma 4 26B opraví SQL terminológiu → SQL guard
            # Benchmark 2026-04-11: avg_overall_score 0.8328 na 16 kritických SQL segmentoch
            # (vs 26B single-pass 0.7915, vs E4B single-pass 0.7899)
            # --no_adapt_llm: text adaptation cez stretch; LLM adapt prebieha cez refine pass
            # --llama_ctx 8192: dosť kontextu pre refine batch
            return ["--use_gemma4_hf",
                    "--gemma4_hf_model",str(PATHS.gemma4_e4b_hf),
                    "--no_adapt_llm",
                    "--refine_translation",
                    "--refine_model",str(PATHS.gemma4_26b_base_q4km),
                    "--refine_gpu_layers","-1",
                    "--llama_ctx","8192"], env
        if engine == "grammar_fix":
            # Preskočí preklad — transkribuje SK audio a opraví len gramatiku cez Ollama (sk-gemma4-e4b-v6)
            return ["--grammar_fix_only"], env
        if engine == "nllb":
            nllb_m = self._nllb_model_cb.currentText() if hasattr(self,"_nllb_model_cb") else "facebook/nllb-200-1.3B"
            return ["--use_nllb","--nllb_model",nllb_m], env
        if engine in ("chatgpt","grok","gemini"):
            key = self._get_api_key(engine)
            if not key: return [], None
            models = {
                "chatgpt": self._oa_translate_model.text(),
                "grok": self._grok_translate_model.text(),
                "gemini": self._gem_translate_model.text(),
            }
            urls   = {"chatgpt":self._oa_url.text(),"grok":self._grok_url.text(),"gemini":self._gem_url.text()}
            args = ["--use_api_translate","--api_translate_provider",engine,
                    "--api_translate_model",models[engine],"--api_translate_base_url",urls[engine]]
            env = os.environ.copy(); env["API_TRANS_KEY"] = key
            hf_key = self._get_hf_api_key()
            if hf_key: env["HF_TOKEN"] = hf_key; env["HUGGINGFACE_HUB_TOKEN"] = hf_key
            return args, env
        return ["--use_translategemma_fulldoc","--translategemma_model",tgemma_model,"--translategemma_gpu_layers","-1"], env

    def _effective_tts_python(self) -> str:
        chosen = self._set_py.text().strip() or self.settings.py
        chosen_path = Path(chosen).expanduser() if chosen else Path()
        if self._tts_is_chatterbox() and _DEFAULT_PY.exists():
            if (not chosen_path.exists()) or ("musetalk_env" in str(chosen_path)):
                return str(_DEFAULT_PY)
        return chosen

    def _build_tts_args(self) -> List[str]:
        def _adapt_args(enabled: bool) -> List[str]:
            adapt_provider = self._adapt_provider.currentText() if hasattr(self, "_adapt_provider") else self.settings.adapt_provider
            adapt_url = self._adapt_ollama_url.text().strip() if hasattr(self, "_adapt_ollama_url") else self.settings.adapt_ollama_url
            adapt_model = self._adapt_ollama_model.text().strip() if hasattr(self, "_adapt_ollama_model") else self.settings.adapt_ollama_model
            common = [
                "--adapt_provider", adapt_provider or "llama_cpp",
                "--adapt_ollama_url", adapt_url or "http://localhost:11434/api/generate",
                "--adapt_ollama_model", adapt_model or "llama3.1:8b",
            ]
            if not enabled:
                return common + ["--no-use_text_adaptation"]
            if adapt_provider == "ollama":
                return common + [
                    "--use_text_adaptation",
                    "--use_ollama_adaptation",
                ]
            return common + [
                "--use_text_adaptation",
                "--llama_model", _DEFAULT_ADAPT_MODEL,
                "--llama_gpu_layers", "-1",
            ]

        if self._tts_is_omnivoice():
            # OmniVoice je multilingual zero-shot — EN slová prečíta s anglickou výslovnosťou
            # natívne. Phonetic respelling (shader→šejder) je škodlivé pre OmniVoice → vypneme.
            # POZNÁMKA: BooleanOptionalAction → negácia má pomlčku: --no-phonetic_respelling
            args = ["--use_omnivoice", "--no-phonetic_respelling"]
            if hasattr(self, "_ov_num_step"):
                args += ["--omnivoice_num_step", str(self._ov_num_step.value())]
            if hasattr(self, "_ov_guidance"):
                args += ["--omnivoice_guidance", f"{self._ov_guidance.value():.2f}"]
            # OmniVoice language = cieľový jazyk z hlavnej voľby (single source of truth)
            _lang = self._tgt_lang.currentText() if hasattr(self, "_tgt_lang") else (
                getattr(self.settings, "tgt_lang", "sk") or "sk")
            args += ["--omnivoice_language", _lang]
            if hasattr(self, "_ov_instruct"):
                _inst = self._ov_instruct.currentText().strip()
                if _inst:
                    _inst_en = _voice_design_canon(_inst)
                    args += ["--omnivoice_instruct", _inst_en]
            if hasattr(self, "_ov_speed"):
                _sp = self._ov_speed.value()
                if abs(_sp - 1.0) > 0.01:
                    args += ["--omnivoice_speed", f"{_sp:.2f}"]
            if hasattr(self, "_ov_expressive") and self._ov_expressive.isChecked():
                args += ["--omnivoice_expressive"]
            voice = _voice_canon(self._cb_voice.currentText()) if hasattr(self, "_cb_voice") else ""
            if voice and voice not in ("(predvolený)", "(originál video)", "🎬 Cloning z videa"):
                voice_path = str(VOICES_DIR / voice)
                args += ["--omnivoice_ref_audio", voice_path]
                # Whisper auto-transcribe ref_audio — path setup pre import 'tts' modulu.
                # GUI bežal v rôznych working dirs, sys.path nemusí mať scripts/.
                try:
                    import sys as _sys_w
                    _scripts_dir = str(BASE_DIR / "scripts")
                    if _scripts_dir not in _sys_w.path:
                        _sys_w.path.insert(0, _scripts_dir)
                    from tts import whisper_transcribe_ref_audio
                    lang = self._tgt_lang.currentText() if hasattr(self, "_tgt_lang") else "sk"
                    ref_text = whisper_transcribe_ref_audio(voice_path, language=lang)
                    args += ["--omnivoice_ref_text", ref_text]
                    self._log_add(f"[OMNIVOICE] Auto ref_text z Whisper: {ref_text[:80]}...")
                except Exception as _e:
                    # Pipeline subprocess to spraví aj tak v omnivoice_tts() — tu len graceful skip
                    self._log_add(f"[OMNIVOICE] Whisper pre-transcribe v GUI zlyhal ({_e}). Pipeline subprocess to skúsi znova.")
        else:
            # Chatterbox path — pre uložiteľnosť overíme dostupnosť modelu + source dir.
            # Ak chýba (užívateľ nemá Chatterbox nainštalovaný), auto-fallback na OmniVoice.
            cb_model_path = Path(self._cb_model.text().strip() or str(_DEFAULT_CHATTERBOX_MODEL))
            cb_src_dirs = [BASE_DIR.parent / "Ai" / "chatterbox_git" / "src",
                            BASE_DIR.parent / "chatterbox_git" / "src"]
            if not cb_model_path.exists() or not any(d.exists() for d in cb_src_dirs):
                missing = "model" if not cb_model_path.exists() else "source"
                self._log_add(f"[CHATTERBOX] {missing} chýba ({cb_model_path if missing=='model' else cb_src_dirs[0]}) "
                              f"— fallback na OmniVoice")
                # Re-execute as OmniVoice — preskočí celý Chatterbox blok
                args = ["--use_omnivoice"]
                if hasattr(self, "_ov_num_step"):
                    args += ["--omnivoice_num_step", str(self._ov_num_step.value())]
                if hasattr(self, "_ov_guidance"):
                    args += ["--omnivoice_guidance", f"{self._ov_guidance.value():.2f}"]
                if hasattr(self, "_ov_speed"):
                    args += ["--omnivoice_speed", f"{self._ov_speed.value():.2f}"]
                if hasattr(self, "_ov_expressive") and self._ov_expressive.isChecked():
                    args.append("--omnivoice_expressive")
                args += ["--omnivoice_language", self._tgt_lang.currentText() or "sk"]
                # Voice ref handling — ak má user vybraté custom WAV, použiť, inak auto-clone z videa.
                voice = _voice_canon(self._cb_voice.currentText())
                if voice and voice not in ("(predvolený)", "(originál video)", "🎬 Cloning z videa"):
                    args += ["--omnivoice_ref_audio", str(VOICES_DIR / voice)]
                return args

            args = ["--use_chatterbox","--chatterbox_model",str(cb_model_path)]
            voice = _voice_canon(self._cb_voice.currentText())
            cb_has_custom_wav = bool(voice and voice not in ("(predvolený)", "(originál video)", "🎬 Cloning z videa"))
            if voice and voice not in ("(predvolený)", "(originál video)", "🎬 Cloning z videa"):
                args += ["--xtts_speaker_wav", str(VOICES_DIR / voice)]
            elif voice == "(predvolený)":
                # Predvolený hlas — použij nakonfigurovaný fallback WAV namiesto auto-extrakcie z videa
                _def = _voice_canon(self._cb_default_voice.currentText()) if hasattr(self, "_cb_default_voice") else ""
                if _def and _def != "(žiadny)" and (VOICES_DIR / _def).exists():
                    args += ["--xtts_speaker_wav", str(VOICES_DIR / _def)]
            elif voice == "(originál video)" and hasattr(self, "_cb_clone_auto") and not self._cb_clone_auto.isChecked():
                args += ["--clone_voice",
                         "--voice_sample_start", f"{self._cb_clone_start.value():.2f}",
                         "--voice_sample_duration", f"{self._cb_clone_dur.value():.2f}"]
            _exag = self._cb_exag.value()
            _cfg  = self._cb_cfg.value()
            _temp = self._cb_temp.value()
            args += ["--chatterbox_exaggeration", f"{_exag:.2f}",
                     "--chatterbox_cfg_weight",   f"{_cfg:.2f}",
                     "--chatterbox_temperature",  f"{_temp:.2f}"]
            if self._cb_emo_clone.isChecked(): args.append("--emotion_clone")
            if hasattr(self, "_cb_male_voice"):
                _male = _voice_canon(self._cb_male_voice.currentText())
                if _male and _male != "(žiadny)" and (VOICES_DIR / _male).exists():
                    args += ["--multi_voice_male_wav", str(VOICES_DIR / _male)]
            if hasattr(self, "_cb_female_voice"):
                _female = _voice_canon(self._cb_female_voice.currentText())
                if _female and _female != "(žiadny)" and (VOICES_DIR / _female).exists():
                    args += ["--multi_voice_female_wav", str(VOICES_DIR / _female)]
            args += _adapt_args(self._text_adapt_enabled.isChecked() if hasattr(self, "_text_adapt_enabled") else self.settings.text_adapt_enabled)
        return args

    def _build_local_cmd(self, input_file: str) -> tuple[List[str], Optional[Dict]]:
        lang = self._tgt_lang.currentText()
        src = self._src_lang.currentText()
        engine = self._trans_engine.currentText()
        tgemma = self._madlad_model.currentText()
        trans_args, env = self._build_trans_args(engine, tgemma)
        if not trans_args and engine in ("chatgpt","grok","gemini"):
            self._log_add(f"[CHYBA] Chýba API key pre {engine.upper()} — nastav v karte API.")
            return [], None
        tts_args = self._build_tts_args()
        content = self._content_type_val()
        settings = self.settings
        raw_py = self._set_py.text().strip() or settings.py
        py_exec = self._effective_tts_python()
        effective_speech_gain = settings.def_speech_gain
        cmd = [
            py_exec, str(SCRIPT_DIR / "main.py"),
            "--input",input_file,"--tgt_lang",lang,
            "--whisper_root","models/faster-whisper",
            "--whisper_model", (self.settings.whisper_model or "large-v3-turbo"),
            "--max_chars","200",
            "--timed","--vad","--timeline_mode","--no_torch_compile",
            "--stretch_min",settings.def_stretch_min,"--stretch_max",settings.def_stretch_max,
            "--base_tempo",settings.def_base_tempo,"--max_pause_gap_s",settings.def_pause_gap_s,
            "--speech_gain",effective_speech_gain,
            "--vad_min_speech_ms",settings.def_vad_min_speech_ms,
            "--vad_min_silence_ms",settings.def_vad_min_silence_ms,
            "--vad_max_speech_s",settings.def_vad_max_speech_s,
            *tts_args, *trans_args,
        ]
        if py_exec != raw_py:
            self._log_add(f"[INFO] Pre Chatterbox používam {py_exec} namiesto {raw_py}.")
        if content != "general": cmd += ["--content_type",content]
        _backaudio_vol = float(settings.def_backaudio_volume or "0.0")
        if _backaudio_vol > 0.0: cmd += ["--backaudio_volume", str(_backaudio_vol)]
        if self._chk_keep_music.isChecked():
            cmd += ["--keep_music", "--music_volume", settings.def_music_volume]
            _vsep = self._vocal_sep.currentText() if hasattr(self, "_vocal_sep") else "demucs"
            # map display name → CLI value
            _vsep_cli = "mdx_net" if _vsep == "BS-Roformer" else _vsep
            if _vsep_cli != "demucs":
                cmd += ["--vocal_separator", _vsep_cli]
        # Opraviť gramatiku: 2-LLM refine pass. Backend podľa Settings → "Refine LLM" dropdown.
        if self._chk_refine.isChecked() and engine not in ("chatgpt","grok","gemini","hybrid"):
            cmd += ["--refine_translation"]
            _refine_choice = getattr(self.settings, "refine_engine_choice",
                                     "Gemma 3 12B v1 (Ollama, fine-tune, default)")
            if "Gemma 3 12B v1 (Ollama" in _refine_choice:
                # Default: stabilný fine-tune pre tech (verified 2026-05-06)
                cmd += ["--adapt_provider", "ollama",
                        "--adapt_ollama_model", "g3-12b-translator-v1",
                        "--use_ollama_adaptation"]
            elif "LM Studio" in _refine_choice:
                # Refine cez ten istý LM Studio model (single GPU constraint)
                cmd += ["--refine_model", "",
                        "--refine_gpu_layers", "-1"]
            elif "Custom Ollama" in _refine_choice:
                # User si nastaví adapt_ollama_model v Settings (adapt_provider=ollama)
                cmd += ["--adapt_provider", "ollama", "--use_ollama_adaptation"]
            else:
                # llama.cpp Gemma 3 12B Q8 GGUF
                cmd += ["--refine_model", str(PATHS.gemma3_12b_q8),
                        "--refine_gpu_layers", "-1"]
        if self._chk_voice_eq.isChecked():
            cmd.append("--voice_eq")
            eq_p = self._eq_profile.currentText()
            if eq_p: cmd += ["--tts_eq_profile",eq_p]
        _cb_voice_text = _voice_canon(self._cb_voice.currentText()) if hasattr(self, "_cb_voice") else "(predvolený)"
        _cb_has_custom_wav = _cb_voice_text not in ("(predvolený)", "(originál video)")
        if self._chk_multi_voice.isChecked() and not (self._tts_is_chatterbox() and _cb_has_custom_wav):
            cmd += ["--multi_voice", "--multi_voice_n", str(self._multi_voice_n.value())]
        # "(predvolený)" means: use source-video voice automatically.
        # "(originál video)" keeps the same source-voice behavior, but also exposes
        # manual clone sample controls when the user disables auto mode.
        _source_voice_mode = (
            self._tts_is_chatterbox() and _voice_canon(self._cb_voice.currentText()) in ("(predvolený)", "(originál video)")
        )
        if not _source_voice_mode:
            cmd.append("--no_voice_clone")
        if self._chk_denoise.isChecked(): cmd += ["--denoise_preset",self._denoise_preset.currentText()]
        else: cmd.append("--no_denoise")
        ps = self._pitch_shift.value()
        if ps != 0.0: cmd += ["--pitch_shift",str(ps)]
        if self._chk_srt_align.isChecked(): cmd.append("--srt_align")
        if hasattr(self, "_chk_mini_level11") and self._chk_mini_level11.isChecked():
            cmd.append("--use_mini_level11")
        if hasattr(self, "_chk_mixed_technical_terms") and self._chk_mixed_technical_terms.isChecked():
            cmd.append("--mixed_technical_terms")
        if hasattr(self, "_chk_stop_after_translate") and self._chk_stop_after_translate.isChecked():
            cmd.append("--stop_after_translate")
            self._log_add("[STOP] Pipeline urobí iba STT+translate. Audio si vyrob cez ./test_omnivoice_batch.py")

        # Auto speaker gender — odvodené z voice design (žena/female)
        if hasattr(self, "_ov_instruct"):
            _vd = self._ov_instruct.currentText().strip().lower()
            _gender = ""
            if any(w in _vd for w in ["žena", "ženská", "ženský hlas", "female", "slovenská žena", "česká žena"]):
                _gender = "feminine"
            elif any(w in _vd for w in ["muž", "mužský", "male", "slovenský muž", "český muž"]):
                _gender = "masculine"
            if _gender:
                cmd += ["--speaker_gender", _gender]
                self._log_add(f"[GENDER] Voice design → preklad v {('ženskom' if _gender=='feminine' else 'mužskom')} rode")
        # --lipsync nie je v main.py — preskočiť
        temp_dir = BASE_DIR / "temp" / Path(input_file).stem
        cmd += ["--out_dir", str(temp_dir)]
        return cmd, env

    # ══════════════════════════════════════════════════════════════════════════
    # Actions — Local
    # ══════════════════════════════════════════════════════════════════════════
    def _cleanup_pre_run(self, input_file: str):
        """Vymaže residuálne súbory z predchádzajúceho behu pre tento input.

        - temp/{stem}/ adresár (sidecar JSONy, segments, intermediate)
        - work/tts_chunks pre tento stem
        - existujúci dabovaný .mp4 výstup (donutí re-mux)
        - tts_*.meta.json (semantic hash cache → donúti regenerovať)
        """
        import shutil
        stem = Path(input_file).stem
        targets = [
            BASE_DIR / "temp" / stem,
            BASE_DIR / "work" / "tts_chunks" / stem,
            BASE_DIR / "input" / "sk" / f"{stem}_sk_tts.mp4",
            BASE_DIR / "input" / "cs" / f"{stem}_cs_tts.mp4",
        ]
        cleaned = 0
        for t in targets:
            if t.exists():
                try:
                    if t.is_dir():
                        shutil.rmtree(t)
                    else:
                        t.unlink()
                    cleaned += 1
                    self._log_add(f"[CLEANUP] Odstránené: {t.name}")
                except Exception as e:
                    self._log_add(f"[CLEANUP] WARN nemohol som zmazať {t}: {e}")
        if cleaned == 0:
            self._log_add(f"[CLEANUP] Žiadne staré súbory pre {stem} — clean start.")

    def _on_start(self):
        if not self.batch_items:
            self._log_add("[CHYBA] Žiadne súbory v zozname. Pridajte súbory pomocou 'Pridať'."); return
        # Find first pending
        pending = [i for i in self.batch_items if i.status in ("pending","failed")]
        if not pending:
            r = QMessageBox.question(
                self, "Znovu spustiť?",
                "Všetky súbory sú už hotové.\n\nChceš spustiť preklad znovu?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return
            for i in self.batch_items:
                i.status = "pending"
            self._refresh_batch()
            pending = list(self.batch_items)
        item = pending[0]
        # Pre-flight: ollama check + auto-start ak je potrebná pre engine/adapt
        needs_ollama = (
            self._trans_engine.currentText() == "gemma"
            or (hasattr(self, "_adapt_provider") and self._adapt_provider.currentText() == "ollama")
        )
        self._ollama_ok = True
        if needs_ollama:
            self._ollama_ok = self._ensure_ollama_running(timeout=30)
            if not self._ollama_ok:
                self._log_add("[OLLAMA] Nebeží — fallback na llama.cpp (lokálny GGUF)")
        # Cleanup residuálnych súborov z predchádzajúceho behu (re-run safety)
        self._cleanup_pre_run(item.path)

        # SRT shortcut: ak je v GUI nastavené existujúce SRT, preskočíme pipeline úplne
        # (žiadny Whisper STT, žiadny preklad — iba TTS + master z SRT-derived segmentov)
        srt_input = ""
        if hasattr(self, "_srt_in"):
            srt_input = (self._srt_in.text() or "").strip()
        if srt_input:
            srt_path = Path(srt_input)
            if not srt_path.exists():
                self._log_add(f"[SRT] Súbor neexistuje: {srt_path} — fallback na štandardný flow")
                srt_input = ""
            elif self._tts_is_omnivoice():
                # SK SRT → priamo OmniVoice standalone bez pipeline
                self._log_add(f"[SRT] Použijem existujúce titulky: {srt_path.name} (preskakujem STT + preklad)")
                src = Path(item.path)
                stem = src.stem
                tgt = self._tgt_lang.currentText() or "sk"
                seg_json = BASE_DIR / "temp" / stem / f"{stem}_{tgt}_segments.json"
                seg_json.parent.mkdir(parents=True, exist_ok=True)
                # Convert SRT → segments JSON
                import subprocess as _sp
                py = self._effective_tts_python()
                conv_cmd = [py, str(BASE_DIR / "srt_to_segments.py"),
                            str(srt_path),
                            "--lang", tgt,
                            "--video-stem", stem,
                            "--out", str(seg_json)]
                r = _sp.run(conv_cmd, capture_output=True, text=True, timeout=60)
                if r.returncode != 0 or not seg_json.exists():
                    self._log_add(f"[SRT] Konverzia zlyhala: {r.stderr[-300:]}")
                    return
                # Print log z konverzie (dôležité info: count, span)
                for line in (r.stdout or "").splitlines():
                    if line.strip() and ("parsed" in line or "total span" in line or "merged" in line):
                        self._log_add(f"[SRT] {line.strip()}")
                item.status = "processing"; self._refresh_batch()
                self._current_batch_item = item
                self._set_running(True)
                # Spustiť priamo OmniVoice standalone (pipeline už nie je potrebný)
                self._log_add(f"[SRT] ✓ Segments JSON: {seg_json.name} — spúšťam OmniVoice TTS")
                self._run_omnivoice_standalone(seg_json, stem, tgt, src)
                return
            else:
                # Chatterbox path s SRT → fallback na pipeline s --use_existing_segments
                self._log_add(f"[SRT] Použijem existujúce titulky pre Chatterbox engine — pipeline s --use_existing_segments")
                # Pre-konvertuj SRT do očakávaného JSON umiestnenia, pipeline ho použije
                src = Path(item.path)
                stem = src.stem
                tgt = self._tgt_lang.currentText() or "sk"
                seg_json = BASE_DIR / "temp" / stem / f"{stem}_{tgt}_segments.json"
                seg_json.parent.mkdir(parents=True, exist_ok=True)
                import subprocess as _sp
                py = self._effective_tts_python()
                _sp.run([py, str(BASE_DIR / "srt_to_segments.py"), str(srt_path),
                         "--lang", tgt, "--video-stem", stem, "--out", str(seg_json)],
                        capture_output=True, timeout=60)

        cmd, env = self._build_local_cmd(item.path)
        if not cmd: return
        # SRT mode: pridaj --use_existing_segments aby pipeline preskočil STT + translate
        if srt_input and "--use_existing_segments" not in cmd:
            cmd.append("--use_existing_segments")
        item.status = "processing"; self._refresh_batch()
        self._current_batch_item = item
        self._runner = Runner(cmd, env)
        self._runner.log_line.connect(self._log_add)
        self._runner.progress.connect(lambda v: (self._progress.setValue(int(v)), setattr(self, '_last_progress', float(v))))
        self._runner.finished.connect(self._on_finished)
        self._runner.start()
        self._set_running(True)
        self._log_add(f"[INFO] Spracovávam: {Path(item.path).name}")
        self._log_add(f"[CMD] {' '.join(cmd[:6])} ...")

    def _on_resume(self):
        ckpt = _load_checkpoint()
        if not ckpt:
            QMessageBox.information(self,"Checkpoint","Žiadny checkpoint na obnovenie.")
            return
        items, _, settings = ckpt
        for i in items:
            if i.status == "processing": i.status = "pending"
        pending = sum(1 for i in items if i.status in ("pending","failed"))
        done    = sum(1 for i in items if i.status == "done")
        r = QMessageBox.question(self,"Obnoviť checkpoint",
            f"Checkpoint nájdený:\n• Hotové: {done}\n• Zostáva: {pending}\n\nPokračovať?")
        if r != QMessageBox.StandardButton.Yes: return
        self.batch_items = items
        self._apply_checkpoint_settings(settings)
        self._refresh_batch()
        self._log_add(f"[INFO] Checkpoint načítaný — pokračujem ({pending} zostáva)")
        self._on_start()

    def _on_stop(self):
        if self._runner and self._runner.isRunning():
            reply = QMessageBox.question(
                self,
                T.get("dlg_stop_title", "Zastaviť preklad?"),
                T.get("dlg_stop_msg", "Preklad stále beží.\nNaozaj ho chceš ukončiť?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        if self._runner: self._runner.stop()
        self._set_running(False)
        self._set_running(False, is_yt=True)
        self._release_resources()

    def _release_resources(self) -> None:
        """Po STOP uvoľní VRAM/RAM — unload ollama modely, GC."""
        import urllib.request as _ur, urllib.error as _ue
        import json as _json
        import gc as _gc
        # 1. Zber všetkých ollama modelov ktoré mohli byť nahrané
        models_to_unload: List[str] = []
        if hasattr(self, "_madlad_model") and hasattr(self, "_trans_engine"):
            if self._trans_engine.currentText() == "gemma":
                m = self._madlad_model.currentText().strip()
                if m and not m.endswith(".gguf"):
                    models_to_unload.append(m)
        if hasattr(self, "_adapt_provider") and self._adapt_provider.currentText() == "ollama":
            if hasattr(self, "_adapt_ollama_model"):
                m = self._adapt_ollama_model.text().strip()
                if m: models_to_unload.append(m)
        # 2. Unload každý cez ollama API (keep_alive=0)
        url = "http://localhost:11434/api/generate"
        for model in set(models_to_unload):
            try:
                body = _json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
                req = _ur.Request(url, data=body, headers={"Content-Type": "application/json"})
                _ur.urlopen(req, timeout=5)
                self._log_add(f"[STOP] Ollama unload: {model}")
            except (_ue.URLError, OSError, TimeoutError) as e:
                self._log_add(f"[STOP] Ollama unload {model} skipped: {e}")
        # 3. Python GC pre lokálne objekty
        _gc.collect()
        if not models_to_unload:
            self._log_add("[STOP] Žiadne ollama modely na uvoľnenie")

    def _on_finished(self, ok: bool):
        item = getattr(self,"_current_batch_item",None)
        if item:
            item.status = "done" if ok else "failed"
            self._refresh_batch()
        self._set_running(False)
        self._progress.setValue(100 if ok else 0)
        msg = f"[{'OK' if ok else 'CHYBA'}] {'Hotovo ✓' if ok else 'Zlyhal!'} — {Path(item.path).name if item else ''}"
        self._log_add(msg)
        self._persist_batch_state()
        # CLEANUP po každom behu (úspešnom aj neúspešnom) — Ollama, LM Studio
        # a python objekty držia VRAM aj keď je pipeline subprocess hotový.
        # Bez cleanup-u sa modely akumulujú v pamäti pri opakovaných behoch.
        try:
            results = self._release_gpu_memory()
            cleaned = [k for k, v in results.items() if v == "unloaded" or v == "terminated"]
            if cleaned:
                self._log_add(f"[CLEANUP] Uvoľnené: {', '.join(cleaned)}")
        except Exception as e:
            self._log_add(f"[CLEANUP] WARN: {e}")
        # Python GC pre lokálne objekty (segments, audio buffers, atď.)
        import gc as _gc_fin
        _gc_fin.collect()
        try:
            import torch as _t_fin
            if _t_fin.cuda.is_available():
                _t_fin.cuda.empty_cache()
                _t_fin.cuda.ipc_collect()
        except Exception:
            pass
        if ok and item:
            self._mux_video(item.path)
        else:
            self._advance_batch(ok)

    def _run_omnivoice_standalone(self, segments_json: Path, stem: str, tgt: str, src_video: Path):
        """Plný OmniVoice flow:
          1. auto_clone_ref.py — extract clone ref z input videa (cached)
          2. test_omnivoice_batch.py — generate per-segment audio s --keep-chunks
          3. match_audio_dynamic.py — per-segment timeline align (atempo, no cuts)
          4. master_pro.py — voice + bg music + heavy compress + EBU R128 -20 LUFS
          5. Output MP4 cez master_pro priamo do input/sk/<stem>_sk_tts.mp4
        """
        import subprocess
        py = self._effective_tts_python()
        scripts = {
            "clone": BASE_DIR / "auto_clone_ref.py",
            "batch": BASE_DIR / "test_omnivoice_batch.py",
            "align": BASE_DIR / "match_audio_dynamic.py",
            "master": BASE_DIR / "master_pro.py",
            "diarize": BASE_DIR / "auto_diarize.py",
            "multi_clone": BASE_DIR / "auto_clone_multi.py",
        }
        for name, p in scripts.items():
            if not p.exists():
                # diarize/multi_clone sú voliteľné — chýbajú len ak multi-voice nie je k dispozícii
                if name in ("diarize", "multi_clone"):
                    continue
                self._log_add(f"[OMNIVOICE] CHYBA: skript chýba {p}")
                self._advance_batch(False)
                return

        # === Step 0 (optional): Multi-voice diarization + per-speaker refs ===
        multi_voice_on = (hasattr(self, "_ov_multi_voice")
                          and self._ov_multi_voice.isChecked()
                          and scripts["diarize"].exists()
                          and scripts["multi_clone"].exists())
        speaker_refs_json = None
        diarized_segments_json = segments_json  # default = original
        if multi_voice_on:
            self._log_add(f"[MULTI-VOICE] Diarization (Resemblyzer + KMeans)...")
            diarized_path = segments_json.with_suffix(".diarized.json")
            r = subprocess.run(
                [py, str(scripts["diarize"]), str(src_video),
                 "--segments", str(segments_json),
                 "--out", str(diarized_path)],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode == 0 and diarized_path.exists():
                # Skontroluj koľko speakers diarizácia zistila
                try:
                    import json as _j
                    _diar = _j.loads(diarized_path.read_text(encoding="utf-8"))
                    _segs = _diar["segments"] if isinstance(_diar, dict) else _diar
                    _spks = sorted(set(s.get("speaker_id", "SPEAKER_00") for s in _segs))
                    self._log_add(f"[MULTI-VOICE] Detected {len(_spks)} speakers: {', '.join(_spks)}")
                    if len(_spks) >= 2:
                        # Per-speaker SK ref-gen
                        self._log_add(f"[MULTI-VOICE] Generujem per-speaker SK refs (Chatterbox PRO config)...")
                        refs_out_dir = VOICES_DIR / src_video.stem
                        r2 = subprocess.run(
                            [py, str(scripts["multi_clone"]), str(src_video),
                             "--diarized", str(diarized_path),
                             "--out-dir", str(refs_out_dir)],
                            capture_output=True, text=True, timeout=1800,
                        )
                        speaker_refs_json = refs_out_dir / "_speaker_refs.json"
                        if r2.returncode == 0 and speaker_refs_json.exists():
                            self._log_add(f"[MULTI-VOICE] ✓ Per-speaker refs: {speaker_refs_json}")
                            diarized_segments_json = diarized_path
                        else:
                            self._log_add(f"[MULTI-VOICE] multi-clone zlyhal: {(r2.stderr or '')[-300:]}")
                            speaker_refs_json = None
                            multi_voice_on = False
                    else:
                        self._log_add(f"[MULTI-VOICE] Single speaker detegovaný — fallback na single-ref flow")
                        multi_voice_on = False
                except Exception as e:
                    self._log_add(f"[MULTI-VOICE] Diarized JSON parse zlyhal: {e}")
                    multi_voice_on = False
            else:
                self._log_add(f"[MULTI-VOICE] Diarization zlyhala: {(r.stderr or '')[-300:]}")
                multi_voice_on = False

        # === Step 1: Auto-clone ref z videa ===
        # Ak user explicitne vybral voice WAV v GUI, použi ten.
        # Inak (predvolený / Cloning z videa / originál video) auto-clone z input videa.
        voice = _voice_canon(self._cb_voice.currentText()) if hasattr(self, "_cb_voice") else "(predvolený)"
        clone_triggers = ("(predvolený)", "🎬 Cloning z videa", "(originál video)")
        if voice and voice not in clone_triggers:
            ref_audio = str(VOICES_DIR / voice)
            self._log_add(f"[OMNIVOICE] Použijem custom voice: {voice}")
        else:
            self._log_add(f"[OMNIVOICE] Auto-clone ref z videa: {src_video.name}")
            clone_ref = VOICES_DIR / f"{src_video.stem}_clone.wav"
            r = subprocess.run(
                [py, str(scripts["clone"]), str(src_video),
                 "--out", str(clone_ref), "--ref-lang", "en"],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0 or not clone_ref.exists():
                self._log_add(f"[OMNIVOICE] auto-clone zlyhal: {r.stderr[-300:]}")
                # Fallback: predvolený greeting
                clone_ref = VOICES_DIR / "chatterbox_sk_greeting_studio.wav"
            ref_audio = str(clone_ref)

        # === Step 2: Batch OmniVoice TTS (single-ref alebo multi-voice) ===
        chunks_dir = BASE_DIR / "work" / "tts_chunks_standalone" / stem
        chunks_dir.mkdir(parents=True, exist_ok=True)
        timeline_in = BASE_DIR / "temp" / stem / f"{stem}_{tgt}_raw.wav"
        timeline_in.parent.mkdir(parents=True, exist_ok=True)
        cmd = [py, str(scripts["batch"]),
               "--segments", str(diarized_segments_json),  # uses diarized if multi-voice on
               "--out", str(timeline_in),
               "--ref", ref_audio,
               "--chunks-dir", str(chunks_dir),
               "--keep-chunks"]
        if multi_voice_on and speaker_refs_json and speaker_refs_json.exists():
            cmd += ["--speaker-refs", str(speaker_refs_json)]
            self._log_add(f"[OMNIVOICE] Batch TTS (MULTI-VOICE): per-speaker refs aktivované")
        else:
            self._log_add(f"[OMNIVOICE] Batch TTS: {len(list(segments_json.read_text().split('text')))-1} segments...")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not timeline_in.exists():
            self._log_add(f"[OMNIVOICE] Batch zlyhal: {r.stderr[-300:]}")
            self._advance_batch(False)
            return

        # === Step 3: Timeline alignment podľa zvoleného módu ===
        timeline_out = BASE_DIR / "temp" / stem / f"{stem}_{tgt}_timeline.wav"
        tl_mode = (self._ov_timeline_mode.currentText() if hasattr(self, "_ov_timeline_mode")
                   else "Dynamický (per-segment)")
        if tl_mode.startswith("Dynamický"):
            dyn_max = float(getattr(self, "_ov_dyn_max", None).value()) if hasattr(self, "_ov_dyn_max") else 1.40
            dyn_min = float(getattr(self, "_ov_dyn_min", None).value()) if hasattr(self, "_ov_dyn_min") else 0.95
            cmd = [py, str(scripts["align"]),
                   "--segments", str(segments_json),
                   "--chunks-dir", str(chunks_dir),
                   "--video", str(src_video),
                   "--out", str(timeline_out),
                   "--max-factor", f"{dyn_max:.2f}",
                   "--min-factor", f"{dyn_min:.2f}"]
            self._log_add(f"[OMNIVOICE] Sync: Dynamický (per-segment atempo, min={dyn_min} max={dyn_max})...")
        elif tl_mode.startswith("Manuálny"):
            # Use match_audio_to_video.py s manual speed factor
            uniform_script = BASE_DIR / "match_audio_to_video.py"
            manual_speed = float(getattr(self.settings, "ov_manual_speed", 1.18))
            if hasattr(self, "_ov_manual_speed"):
                manual_speed = float(self._ov_manual_speed.value())
            cmd = [py, str(uniform_script), str(src_video), str(timeline_in),
                   "--out", str(timeline_out),
                   "--max-factor", str(max(1.0, manual_speed))]
            self._log_add(f"[OMNIVOICE] Sync: Manuálny (uniform speed {manual_speed:.2f}×)...")
        else:
            # "Žiadny" — iba copy raw audio do timeline_out
            import shutil as _sh
            _sh.copy(timeline_in, timeline_out)
            cmd = None
            self._log_add(f"[OMNIVOICE] Sync: Žiadny (raw audio bez stretchu)")
        if cmd is not None:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0 or not timeline_out.exists():
                self._log_add(f"[OMNIVOICE] Sync zlyhal: {r.stderr[-300:]}")
                self._advance_batch(False)
                return

        # Accent transfer Step 3.5 odstránený — SK akcent rieši Chatterbox-ref-gen
        # v auto_clone_ref.py (Chatterbox vyrobí SK-flavored ref, OmniVoice klonuje z neho).

        # === Step 4: PRO master (voice + bg + compress + LUFS) ===
        out_mp4 = BASE_DIR / "input" / tgt / f"{stem}_{tgt}_tts.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [py, str(scripts["master"]),
               str(src_video), str(timeline_out),
               "--out", str(out_mp4),
               "--bg-volume", "-15", "--lufs", "-20"]
        self._log_add(f"[OMNIVOICE] PRO master (voice + bg + LUFS -20)...")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0 or not out_mp4.exists():
            self._log_add(f"[OMNIVOICE] PRO master zlyhal: {r.stderr[-300:]}")
            self._advance_batch(False)
            return

        self._log_add(f"[OMNIVOICE] ✓ Hotové: {out_mp4}")
        # Cleanup po standalone chain (demucs, OmniVoice subprocess, master)
        try:
            self._release_gpu_memory()
        except Exception:
            pass
        # Disk cleanup — chunks accumulujú stovky MB (94 chunks × ~300KB = ~30MB per beh)
        try:
            import shutil as _sh_cl
            if chunks_dir.exists():
                _sh_cl.rmtree(chunks_dir, ignore_errors=True)
                self._log_add(f"[CLEANUP] Disk: zmazané chunks {chunks_dir.name}")
            # Aj intermediate timeline_in (raw concat pred align)
            if timeline_in.exists():
                timeline_in.unlink(missing_ok=True)
        except Exception as _ce:
            self._log_add(f"[CLEANUP] Disk warning: {_ce}")
        import gc as _gc_ov
        _gc_ov.collect()
        try:
            import torch as _t_ov
            if _t_ov.cuda.is_available():
                _t_ov.cuda.empty_cache()
                _t_ov.cuda.ipc_collect()
        except Exception:
            pass
        # Mark batch item as done — final MP4 už existuje, mux preskočiť
        item = getattr(self, "_current_batch_item", None)
        if item:
            item.status = "done"
            self._refresh_batch()
        self._persist_batch_state()
        # Unfreeze GUI — Spustiť/Stop tlačidlá späť do normal mode
        self._set_running(False)
        self._progress.setValue(100)
        self._advance_batch(True)

    def _mux_video(self, src_path: str):
        """Merge translated WAV into original video → MP4, then continue batch."""
        src   = Path(src_path)
        tgt   = self._tgt_lang.currentText()
        stem  = src.stem
        # Strip any existing _{tgt}_tts suffix to avoid double-suffix (_sk_tts_sk_tts)
        _suffix = f"_{tgt}_tts"
        if stem.endswith(_suffix):
            stem = stem[:-len(_suffix)]
        # WAV may be in temp/<stem>/ (new) or next to video (legacy fallback)
        temp_audio = BASE_DIR / "temp" / stem / f"{stem}_{tgt}_tts.wav"
        legacy_audio = src.with_name(f"{stem}_{tgt}_tts.wav")
        audio = temp_audio if temp_audio.exists() else legacy_audio
        if not audio.exists():
            # OMNIVOICE auto-stop scenár: pipeline preskočil TTS (pre OmniVoice je
            # zakázané sa dotknúť audia). Audio sa vyrába samostatne. Spusti standalone
            # batch script automaticky.
            if self._tts_is_omnivoice():
                seg_json = BASE_DIR / "temp" / stem / f"{stem}_{tgt}_segments.json"
                if seg_json.exists():
                    self._log_add(f"[OMNIVOICE] Pipeline urobil translate. Spúšťam standalone OmniVoice TTS pre audio...")
                    self._run_omnivoice_standalone(seg_json, stem, tgt, src)
                    return
            self._log_add(f"[MUX] Audio nenájdené, preskakujem mux: {audio}")
            item = getattr(self, "_current_batch_item", None)
            if item:
                item.status = "failed"
                self._refresh_batch()
            self._persist_batch_state()
            self._advance_batch(False)
            return
        out_dir_str = self._out_dir.text().strip()
        out_dir = Path(out_dir_str) if out_dir_str else src.parent / tgt
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_dir / f"{stem}_{tgt}_tts.mp4"
        ffmpeg  = self.settings.ffmpeg or "ffmpeg"
        # Use soundfile for WAV duration (fast, no subprocess) and ffprobe only for video
        # to avoid blocking the main thread with two slow subprocess.run() calls.
        try:
            import soundfile as _sf
            audio_dur = _sf.info(str(audio)).duration
        except Exception:
            audio_dur = None
        video_dur = _probe_media_duration(src)
        tail_gap = (video_dur - audio_dur) if video_dur and audio_dur else 0.0
        max_tail_gap = max(30.0, min(60.0, (video_dur or 0.0) * 0.05)) if video_dur else 30.0

        # If the dubbed audio already reaches the final spoken segment and only the
        # original video has a longer outro tail, allow controlled padding instead
        # of failing the whole batch item.
        _tail_is_only_outro = False
        _last_seg_end = None
        try:
            import json as _json
            _temp_dir = BASE_DIR / "temp" / stem
            _tts_json = _temp_dir / f"{stem}_{tgt}_tts_segments.json"
            if not _tts_json.exists():
                _tts_json = _temp_dir / f"{stem}_{tgt}_segments.json"
            if _tts_json.exists():
                _payload = _json.loads(_tts_json.read_text(encoding="utf-8"))
                if isinstance(_payload, list) and _payload:
                    _last_seg_end = max(float(seg.get("end", 0.0)) for seg in _payload)
                    # Audio should already cover the spoken material, with a small
                    # grace margin for fades/mastering. The remaining video tail can
                    # then be padded safely.
                    if (
                        audio_dur is not None
                        and video_dur is not None
                        and audio_dur >= (_last_seg_end - 2.0)
                        and (video_dur - _last_seg_end) <= 35.0
                    ):
                        _tail_is_only_outro = True
        except Exception:
            _tail_is_only_outro = False

        if tail_gap > max_tail_gap and not _tail_is_only_outro:
            self._log_add(
                f"[MUX] ✗ Audio je kratšie než video o {_fmt_duration(tail_gap)}. "
                f"To by vytvorilo dlhý tichý chvost, mux ruším.",
            )
            item = getattr(self, "_current_batch_item", None)
            if item:
                item.status = "failed"
                self._refresh_batch()
            self._persist_batch_state()
            self._advance_batch(False)
            return
        if tail_gap > max_tail_gap and _tail_is_only_outro:
            self._log_add(
                f"[MUX] Audio je kratšie než video o {_fmt_duration(tail_gap)}, "
                f"ale dub pokrýva posledný hovorený segment"
                + (f" (do {_fmt_duration(_last_seg_end)})." if _last_seg_end else ".")
                + " Doplňujem kontrolovaný tichý/outro chvost."
            )
        _mux_res = self.settings.mux_resolution.strip() if self.settings.mux_resolution else ""
        # Subtitle burn-in: look for translated SRT in temp dir
        _burn_subs = getattr(self.settings, "chk_burn_subtitles", False)
        _srt_path: Path | None = None
        if _burn_subs:
            _temp_dir = BASE_DIR / "temp" / stem
            for _srt_candidate in [
                _temp_dir / f"{stem}_{tgt}_tts_input.srt",
                _temp_dir / f"{stem}_{tgt}.srt",
            ]:
                if _srt_candidate.exists():
                    _srt_path = _srt_candidate
                    break
            if _srt_path:
                self._log_add(f"[MUX] Titulky: {_srt_path.name}")
            else:
                self._log_add("[MUX] Titulky: SRT nenájdené, pokračujem bez titulkov")
        # Build video filter: subtitles and/or scale
        _vf_parts = []
        if _srt_path:
            # Escape backslashes and colons for ffmpeg filter syntax
            _srt_escaped = str(_srt_path).replace("\\", "/").replace(":", "\\:")
            _vf_parts.append(
                f"subtitles='{_srt_escaped}':force_style='FontName=Arial,FontSize=22,"
                f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0'"
            )
        if _mux_res:
            _vf_parts.append(f"scale={_mux_res}")
        _scale_args = ["-vf", ",".join(_vf_parts)] if _vf_parts else []
        # Must re-encode when burning subtitles or rescaling
        _needs_reencode = bool(_mux_res or _srt_path)
        _vcodec_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "18"] if _needs_reencode else ["-c:v", "copy"]
        if _mux_res:
            self._log_add(f"[MUX] Zmena rozlíšenia → {_mux_res} (re-encode libx264)")
        if tail_gap > 0.25:
            pad_to = max(video_dur or 0.0, audio_dur or 0.0)
            cmd = [
                ffmpeg, "-y",
                "-i", str(src),
                "-i", str(audio),
                "-filter_complex", f"[1:a]apad=whole_dur={pad_to:.3f}[a]",
                "-map", "0:v:0", "-map", "[a]",
                *_scale_args, *_vcodec_args,
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
                str(out_mp4),
            ]
        else:
            cmd = [
                ffmpeg, "-y",
                "-i", str(src),
                "-i", str(audio),
                "-map", "0:v:0", "-map", "1:a:0",
                *_scale_args, *_vcodec_args,
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
                str(out_mp4),
            ]
        self._log_add(f"[MUX] {src.name} + {audio.name} → {out_mp4.name}")
        self._mux_runner = Runner(cmd)   # must be stored — local var gets GC'd
        self._mux_runner.log_line.connect(self._log_add)
        self._mux_runner.finished.connect(lambda ok, p=out_mp4: self._on_mux_done(ok, p))
        self._mux_runner.start()

    def _on_mux_done(self, ok: bool, out_mp4: Path):
        if ok:
            self._log_add(f"[MUX] ✓ Video vytvorené: {out_mp4}")
        else:
            item = getattr(self, "_current_batch_item", None)
            if item:
                item.status = "failed"
                self._refresh_batch()
            self._log_add(f"[MUX] ✗ ffmpeg zlyhal — {out_mp4}")
        self._persist_batch_state()
        self._advance_batch(ok)

    def _advance_batch(self, ok: bool):
        if ok:
            self._badge.setText("● Hotovo")
            self._badge.setStyleSheet(f"color:{GREEN}; font-weight:bold; background:transparent;")
            if hasattr(self, "_step_bar"): self._step_bar.set_step(4)  # all done
        pending = [i for i in self.batch_items if i.status == "pending"]
        if pending:
            self._log_add(f"[INFO] Pokračujem ďalším za 8 s (uvoľnenie VRAM): {Path(pending[0].path).name}")
            QTimer.singleShot(8000, self._on_start)
        else:
            # Žiadne ďalšie pending → batch hotový. Unfreeze GUI (kritické pre SRT path,
            # kde _run_omnivoice_standalone ide priamo bez Runner subprocess-u).
            self._set_running(False)
            failed = [i for i in self.batch_items if i.status == "failed"]
            if failed:
                self._log_add(f"[INFO] Batch dokončený s chybami: {len(failed)} zlyhaní")
                self._persist_batch_state()
            else:
                self._log_add("[INFO] Batch dokončený ✓")
                CHECKPOINT_FILE.unlink(missing_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Actions — Stiahnutie videa
    # ══════════════════════════════════════════════════════════════════════════
    def _on_yt_start(self):
        url = self._yt_url.text().strip()
        if not url: self._log_add("[CHYBA] Zadajte URL adresu videa.", self._yt_log); return
        quality   = self._yt_quality.currentText()
        out_dir   = self._yt_outdir.text().strip() or str(Path.home() / "Stiahnuté")
        is_audio  = self._yt_fmt.currentIndex() == 1
        playlist  = self._yt_playlist.isChecked()
        cookies   = self._yt_cookies.currentText()

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if playlist:
            out_tmpl = str(Path(out_dir) / "%(playlist_title)s" / "%(playlist_index)s - %(title)s.%(ext)s")
        else:
            out_tmpl = str(Path(out_dir) / "%(title)s.%(ext)s")

        playlist_args = [] if playlist else ["--no-playlist"]
        cookies_args  = ["--cookies-from-browser", cookies] if cookies != "(žiadne)" else []

        if is_audio:
            cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", out_tmpl, *playlist_args, *cookies_args, url]
        else:
            cmd = ["yt-dlp", "-f", f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]",
                   "--merge-output-format", "mp4", "-o", out_tmpl, *playlist_args, *cookies_args, url]

        self._log_add(f"[INFO] Sťahujem: {url}", self._yt_log)
        self._runner = Runner(cmd)
        self._runner.log_line.connect(lambda l: self._log_add(l, self._yt_log))
        self._runner.finished.connect(lambda ok: (
            self._log_add("[OK] Stiahnuté ✓  →  " + out_dir, self._yt_log) if ok
            else self._log_add("[CHYBA] yt-dlp zlyhal.", self._yt_log),
            self._set_running(False, is_yt=True),
        ))
        self._runner.start()
        self._set_running(True, is_yt=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Actions — API save
    # ══════════════════════════════════════════════════════════════════════════
    def _save_api(self):
        self.settings.openai_api_key  = ""
        self.settings.openai_translate_model = self._oa_translate_model.text().strip() or "gpt-5-mini-2025-08-07"
        self.settings.openai_tts_model = self._oa_tts_model.text().strip() or "gpt-4o-mini-tts"
        self.settings.openai_tts_voice = self._oa_voice.text().strip()
        self.settings.openai_tts_base_url = self._oa_url.text().strip()
        self.settings.grok_api_key    = ""
        self.settings.grok_translate_model = self._grok_translate_model.text().strip() or "grok-3-mini"
        self.settings.grok_tts_model  = self._grok_tts_model.text().strip() or "grok-3-mini"
        self.settings.grok_tts_voice  = self._grok_voice.text().strip()
        self.settings.grok_tts_base_url = self._grok_url.text().strip()
        self.settings.gemini_api_key  = ""
        self.settings.gemini_translate_model = self._gem_translate_model.text().strip() or "gemini-2.5-flash"
        self.settings.gemini_tts_model = self._gem_tts_model.text().strip() or "gemini-2.5-flash"
        self.settings.gemini_tts_voice = self._gem_voice.text().strip()
        self.settings.gemini_tts_base_url = self._gem_url.text().strip()
        self.settings.hf_api_key = self._get_hf_api_key()
        self.settings.save()
        QMessageBox.information(
            self,
            "API nastavenia",
            "Uložené ✓",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Actions — Tools
    # ══════════════════════════════════════════════════════════════════════════
    def _run_tool(self, cmd: List[str]):
        r = Runner(cmd)
        r.log_line.connect(lambda l: self._log_add(l, self._tools_log))
        r.finished.connect(lambda ok: self._log_add("[OK] Hotovo ✓" if ok else "[CHYBA] Zlyhalo!", self._tools_log))
        r.start(); self._runner = r

    def _tool_fps(self):
        src = self._fps_in.text().strip()
        if not src: QMessageBox.warning(self,"Chyba","Vyberte vstupný súbor."); return
        dst = self._fps_out.text().strip() or str(Path(src).with_stem(Path(src).stem+"_25fps"))
        cmd = [self.settings.ffmpeg,"-i",src,"-r","25","-c:v","libx264","-preset","fast","-crf","18","-c:a","copy","-y",dst]
        self._run_tool(cmd)

    def _tool_split(self):
        src = self._split_in.text().strip()
        if not src: QMessageBox.warning(self,"Chyba","Vyberte vstupný súbor."); return
        mins = self._split_mins.value()
        out_dir = Path(src).with_suffix(""); out_dir.mkdir(exist_ok=True)
        cmd = [self.settings.ffmpeg,"-i",src,"-c","copy","-map","0","-segment_time",str(mins*60),
               "-f","segment","-reset_timestamps","1",str(out_dir/"%03d.mp4")]
        self._run_tool(cmd)

    def _tool_join(self):
        d = self._join_dir.text().strip()
        if not d: QMessageBox.warning(self,"Chyba","Vyberte priečinok s časťami."); return
        parts = sorted(Path(d).glob("*.mp4"))
        if not parts: QMessageBox.warning(self,"Chyba","Nenašli sa MP4 súbory."); return
        lst = Path(d)/"concat_list.txt"
        lst.write_text("\n".join(f"file '{p}'" for p in parts))
        dst = self._join_out.text().strip() or str(Path(d)/"joined.mp4")
        cmd = [self.settings.ffmpeg,"-f","concat","-safe","0","-i",str(lst),"-c","copy","-y",dst]
        self._run_tool(cmd)

    def _tool_diarize(self):
        src = self._diar_in.text().strip()
        if not src or not Path(src).exists():
            QMessageBox.warning(self, "Chyba", "Vyberte vstupný video/audio súbor."); return
        lang = self._diar_lang.text().strip() or "en"
        min_spk = self._diar_min_spk.value()
        max_spk = self._diar_max_spk.value()
        hf_token = self._get_hf_api_key()
        if not hf_token:
            QMessageBox.warning(self, "Chyba", "Chýba HuggingFace token — nastav ho v karte API."); return

        py = str(PATHS.python_chatterbox)
        stem = Path(src).stem
        out_l0  = str(BASE_DIR / "output" / f"{stem}_level0.json")
        out_l0b = str(BASE_DIR / "output" / f"{stem}_level0b.json")
        refs_dir = str(BASE_DIR / "output" / "speaker_refs")

        # Spustíme Level 0 a Level 0b ako jeden zreťazený príkaz
        env_prefix = "PYTHONWARNINGS=ignore::UserWarning TRANSFORMERS_VERBOSITY=error"
        cmd = [
            "bash", "-c",
            f'{env_prefix} {py} {BASE_DIR}/run_level0.py "{src}" '
            f'--language {lang} --device cuda '
            f'--min-speakers {min_spk} --max-speakers {max_spk} '
            f'--hf-token "{hf_token}" '
            f'--output "{out_l0}" --refs-dir "{refs_dir}" && '
            f'{env_prefix} {py} {BASE_DIR}/run_level0b.py "{src}" '
            f'--input "{out_l0}" --output "{out_l0b}" --device cuda'
        ]
        self._log_add(f"[DIAR] Spúšťam diarizáciu: {Path(src).name} (min={min_spk}, max={max_spk})", self._tools_log)
        self._log_add(f"[DIAR] Výstup: {out_l0b}", self._tools_log)
        self._run_tool(cmd)

    def _tool_onnx(self):
        trt = "TensorRT" in self._onnx_type.currentText()
        mod = self._onnx_models.currentText()
        args = [self.settings.py,"export_onnx.py"]
        if "All" in mod: args.append("--export_all")
        elif "UNet" in mod: args.append("--export_unet")
        elif "VAE" in mod: args.append("--export_vae")
        elif "BiSeNet" in mod: args.append("--export_bisenet")
        elif "Whisper" in mod: args.append("--export_whisper")
        if trt: args.append("--tensorrt")
        if self._onnx_fp16.isChecked(): args.append("--fp16")
        self._run_tool(args)

    # ══════════════════════════════════════════════════════════════════════════
    # Actions — EQ
    # ══════════════════════════════════════════════════════════════════════════
    def _eq_load_preset(self, name: str):
        filters = EQ_BUILTIN_PRESETS.get(name, [])
        self._eq_filters.setPlainText("\n".join(filters))

    def _eq_apply(self):
        src = self._eq_in.text().strip(); dst = self._eq_out.text().strip()
        if not src or not dst: QMessageBox.warning(self,"Chyba","Zadajte vstup aj výstup."); return
        filters = [l.strip() for l in self._eq_filters.toPlainText().splitlines() if l.strip()]
        if not filters: QMessageBox.warning(self,"Chyba","Žiadne filtre."); return
        af = ",".join(filters)
        cmd = [self.settings.ffmpeg,"-i",src,"-af",af,"-y",dst]
        r = Runner(cmd)
        r.log_line.connect(lambda l: self._log_add(l,self._eq_log))
        r.finished.connect(lambda ok: self._log_add("[OK] EQ aplikovaný ✓" if ok else "[CHYBA]",self._eq_log))
        r.start(); self._runner = r

    def _play_audio(self, path: str):
        if not path: return
        ffplay = shutil.which("ffplay") or str(Path(self.settings.ffmpeg).parent / "ffplay")
        subprocess.Popen([ffplay, "-nodisp", "-autoexit", path])

    # ══════════════════════════════════════════════════════════════════════════
    # Actions — Settings save
    # ══════════════════════════════════════════════════════════════════════════
    def _save_settings(self):
        self.settings.py           = self._set_py.text().strip()
        self.settings.ffmpeg       = self._set_ffmpeg.text().strip()
        self.settings.def_tts_engine   = self._set_tts.currentText()
        self.settings.def_trans_engine = self._set_trans.currentText()
        self.settings.def_speech_gain  = self._set_speech_gain.text().strip()
        self.settings.def_music_volume = self._set_music_vol.text().strip()
        self.settings.def_base_tempo   = self._set_tempo.text().strip()
        self.settings.def_stretch_min  = self._set_str_min.text().strip()
        self.settings.def_stretch_max  = self._set_str_max.text().strip()
        self.settings.def_pause_gap_s  = self._set_pause.text().strip()
        self.settings.chk_denoise      = self._chk_denoise.isChecked()
        self.settings.denoise_preset   = self._denoise_preset.currentText()
        if hasattr(self, "_mux_resolution"):
            _res = self._mux_resolution.currentText()
            self.settings.mux_resolution = "" if _res == T.get("mux_res_original","(pôvodné)") else _res
        if hasattr(self, "_adapt_provider"):
            self.settings.adapt_provider = self._adapt_provider.currentText()
        if hasattr(self, "_adapt_ollama_url"):
            self.settings.adapt_ollama_url = self._adapt_ollama_url.text().strip() or "http://localhost:11434/api/generate"
        if hasattr(self, "_adapt_ollama_model"):
            self.settings.adapt_ollama_model = self._adapt_ollama_model.text().strip() or "llama3.1:8b"
        self.settings.def_vad_min_speech_ms  = self._set_vad_speech.text().strip()
        self.settings.def_vad_min_silence_ms = self._set_vad_silence.text().strip()
        self.settings.def_vad_max_speech_s   = self._set_vad_max.text().strip()
        self.settings.cb_exaggeration = self._cb_exag.value()
        self.settings.cb_cfg          = self._cb_cfg.value()
        self.settings.cb_temp         = self._cb_temp.value()
        self.settings.window_width  = self._win_width.value()
        self.settings.window_height = self._win_height.value()
        new_lang = self._lang_cb.currentData()
        lang_changed = new_lang != self.settings.ui_language
        self.settings.ui_language = new_lang
        self.settings.save()
        QMessageBox.information(self, T.get("msg_restart_title","Nastavenia"), T.get("btn_save_settings","Uložiť nastavenia") + " ✓")
        if lang_changed:
            self._retranslate()

    def _save_chk_state(self, *_):
        s = self.settings
        _cb_voice_text = _voice_canon(self._cb_voice.currentText()) if hasattr(self, "_cb_voice") else "(predvolený)"
        _cb_has_custom_wav = _cb_voice_text not in ("(predvolený)", "(originál video)")
        s.chk_keep_music       = self._chk_keep_music.isChecked()
        if hasattr(self, "_vocal_sep"):
            s.vocal_separator = self._vocal_sep.currentText()
        s.chk_refine      = self._chk_refine.isChecked()
        s.chk_voice_eq    = self._chk_voice_eq.isChecked()
        s.chk_multi_voice = self._chk_multi_voice.isChecked() and not _cb_has_custom_wav
        s.multi_voice_n   = self._multi_voice_n.value()
        s.chk_srt_align   = self._chk_srt_align.isChecked()
        if hasattr(self, "_chk_mini_level11"):
            s.chk_mini_level11 = self._chk_mini_level11.isChecked()
        if hasattr(self, "_chk_mixed_technical_terms"):
            s.chk_mixed_technical_terms = self._chk_mixed_technical_terms.isChecked()
        s.eq_profile      = self._eq_profile.currentText()
        s.denoise_preset  = self._denoise_preset.currentText()
        s.cb_exaggeration  = self._cb_exag.value()
        s.cb_cfg           = self._cb_cfg.value()
        s.cb_temp          = self._cb_temp.value()
        s.cb_emo_clone     = self._cb_emo_clone.isChecked()
        if hasattr(self, "_cb_voice"):
            s.cb_voice = _voice_canon(self._cb_voice.currentText())
        if hasattr(self, "_cb_default_voice"):
            s.cb_default_voice = _voice_canon(self._cb_default_voice.currentText())
        if hasattr(self, "_cb_male_voice"):
            s.cb_male_voice = _voice_canon(self._cb_male_voice.currentText())
        if hasattr(self, "_cb_female_voice"):
            s.cb_female_voice = _voice_canon(self._cb_female_voice.currentText())
        if hasattr(self, "_cb_clone_auto"):
            s.cb_clone_auto = self._cb_clone_auto.isChecked()
            s.cb_clone_start = self._cb_clone_start.value()
            s.cb_clone_duration = self._cb_clone_dur.value()
        if hasattr(self, "_cb_model") and self._cb_model.text().strip():
            s.chatterbox_model = self._cb_model.text().strip()
        if hasattr(self, "_ov_num_step"):
            s.ov_num_step = int(self._ov_num_step.value())
        if hasattr(self, "_ov_guidance"):
            s.ov_guidance = float(self._ov_guidance.value())
        # ov_language sa odvodzuje z tgt_lang (single source)
        if hasattr(self, "_tgt_lang"):
            s.ov_language = self._tgt_lang.currentText() or "sk"
        # ov_instruct sa zámerne NEukladá — pri ďalšom štarte GUI začne prázdne.
        # Voice design je per-video voľba, nemá zmysel pamätať medzi videami.
        s.ov_instruct = ""
        if hasattr(self, "_ov_speed"):
            s.ov_speed = float(self._ov_speed.value())
        if hasattr(self, "_ov_expressive"):
            s.ov_expressive = self._ov_expressive.isChecked()
        s.src_lang         = self._src_lang.currentText()
        s.tgt_lang         = self._tgt_lang.currentText()
        s.trans_engine     = self._trans_engine.currentText()
        # sync paths from Settings tab so they never get overwritten with stale values
        if hasattr(self, "_set_py") and self._set_py.text().strip():
            s.py = self._set_py.text().strip()
        if hasattr(self, "_set_ffmpeg") and self._set_ffmpeg.text().strip():
            s.ffmpeg = self._set_ffmpeg.text().strip()
        if hasattr(self, "_adapt_provider"):
            s.adapt_provider = self._adapt_provider.currentText()
        if hasattr(self, "_adapt_ollama_url"):
            s.adapt_ollama_url = self._adapt_ollama_url.text().strip() or "http://localhost:11434/api/generate"
        if hasattr(self, "_adapt_ollama_model"):
            s.adapt_ollama_model = self._adapt_ollama_model.text().strip() or "llama3.1:8b"
        if hasattr(self, "_text_adapt_enabled"):
            s.text_adapt_enabled = self._text_adapt_enabled.isChecked()
        if hasattr(self, "_ui_mode_cb"):
            s.ui_mode = "extended" if self._ui_mode_cb.currentIndex() == 1 else "normal"
        s.save()

    def _on_release_vram_clicked(self):
        self._log_add("[GPU] Uvoľňujem VRAM...")
        results = self._release_gpu_memory()
        for k, v in results.items():
            self._log_add(f"[GPU] {k}: {v}")
        # Po cleanup show nvidia-smi výsledok
        try:
            import subprocess as _sp
            r = _sp.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                used, free = r.stdout.strip().split(", ")
                self._log_add(f"[GPU] Stav po cleanup: {used} MiB used / {free} MiB free")
        except Exception:
            pass
        QMessageBox.information(self, "GPU cleanup", f"Hotovo. Pozri log pre detaily.")

    def _release_gpu_memory(self) -> dict:
        """Pokus o kompletné uvoľnenie VRAM — Ollama unload + LM Studio unload + best effort.

        Volá sa pri:
        - manual Tools → "Uvoľniť VRAM"
        - close window keď bežal preklad
        - po crashe pipeline subprocess-u
        """
        results = {}
        # 1) Unload Ollama models (HTTP API)
        try:
            import requests as _rq
            r = _rq.get("http://localhost:11434/api/ps", timeout=3)
            if r.ok:
                loaded = r.json().get("models", [])
                for m in loaded:
                    name = m.get("name", "")
                    if name:
                        try:
                            _rq.post("http://localhost:11434/api/generate",
                                     json={"model": name, "keep_alive": 0, "prompt": ""},
                                     timeout=10)
                            results[f"ollama:{name}"] = "unloaded"
                        except Exception as e:
                            results[f"ollama:{name}"] = f"failed: {e}"
        except Exception as e:
            results["ollama"] = f"unreachable: {e}"
        # 2) Unload LM Studio model
        try:
            import subprocess as _sp
            _lms = Path(PATHS.lmstudio_bin) if PATHS.lmstudio_bin else None
            if _lms and _lms.exists():
                r = _sp.run([str(_lms), "unload", "--all"], capture_output=True, timeout=15)
                results["lmstudio"] = "unloaded" if r.returncode == 0 else f"failed: {r.stderr.decode()[-200:]}"
        except Exception as e:
            results["lmstudio"] = f"failed: {e}"
        # 3) Force kill any orphaned VTS subprocess (f5tts_env / chatterbox_env etc.)
        try:
            import subprocess as _sp
            r = _sp.run(["pgrep", "-f", "VideoTranslator_studio_v2|f5tts_env/bin/python|chatterbox_env/bin/python"],
                        capture_output=True, text=True, timeout=5)
            pids = [int(p) for p in r.stdout.split() if p.isdigit()]
            our_pid = os.getpid()
            for pid in pids:
                if pid == our_pid:
                    continue
                try:
                    os.kill(pid, 15)  # SIGTERM
                    results[f"pid_{pid}"] = "terminated"
                except ProcessLookupError:
                    pass
                except PermissionError:
                    results[f"pid_{pid}"] = "no_permission"
        except Exception as e:
            results["pid_cleanup"] = f"failed: {e}"
        return results

    def closeEvent(self, event):
        if self._runner and self._runner.isRunning():
            reply = QMessageBox.question(
                self,
                T.get("dlg_close_title", "Zavrieť program?"),
                T.get("dlg_close_msg", "Preklad stále beží.\nNaozaj chceš program zavrieť?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._runner.stop(); self._runner.wait(2000)
        # Pri zatváraní vždy uvoľni VRAM ak bol niekedy spustený preklad
        if getattr(self, "_runner", None) is not None:
            try:
                self._release_gpu_memory()
            except Exception:
                pass
        self._save_chk_state()
        super().closeEvent(event)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    global T
    # Load language from config before building UI
    _s = AppSettings(); _s.load()
    T = _load_language(_s.ui_language)
    if not T:  # fallback
        T = _load_language("sk")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = StudioWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
