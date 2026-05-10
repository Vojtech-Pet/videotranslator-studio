"""F5 SK smart wrapper - inferencia s adaptívnou duration heuristikou.

Rieši F5-TTS architektonický limit (math-based duration prediction)
pre slovenský fine-tune modelu. Smart duration aktivuje fix_duration
override pre známe problémové prípady (krátke texty, intro words,
skratky, čísla v slovách).

Test matrix (Whisper QA z 8 testov, 2026-05-05):
    F5 default:        363
    Smart heuristika:  378  (+15)

Použitie:
    from f5_sk_smart import F5SKSmart
    f5 = F5SKSmart(ckpt_path="model_60000.pt", vocab_path="vocab.txt",
                   ref_audio="juraj_sk_clean_3s.wav",
                   ref_text="Nahrávka vznikla na základe...")
    f5.infer("Dobrý deň.", out_path="hello.wav")
"""
import re
from pathlib import Path
from typing import Optional
import subprocess

INTRO_WORDS = {
    'hej', 'áno', 'nie', 'ach', 'oh', 'wow', 'fíha', 'aha', 'och', 'ó',
    'pozri', 'počuj', 'počúvaj',
}

NUMBER_WORDS = (
    r'\b(jeden|dva|tri|štyri|päť|šesť|sedem|osem|deväť|desať'
    r'|jedenásť|dvanásť|trinásť|štrnásť|pätnásť|šestnásť|sedemnásť|osemnásť|devätnásť'
    r'|dvadsať|tridsať|štyridsať|päťdesiat|šesťdesiat|sedemdesiat|osemdesiat|deväťdesiat'
    r'|sto|tisíc|dvetisíc|tisícdeväťsto|miliarda|milión)\w*'
)


def needs_smart_duration(text: str) -> bool:
    """Vráti True ak by text mal použiť smart duration override."""
    t = text.strip()
    chars = len(t)
    if chars < 30:
        return True
    first = re.split(r'[\s,.!?]', t, 1)[0].lower().strip()
    if first in INTRO_WORDS:
        return True
    # abbreviations like "sí-plus-plus" or "X+Y"
    if re.search(r'\b\w+[+\-]\w+', t):
        return True
    # number words spelled out
    if re.search(NUMBER_WORDS, t.lower()):
        return True
    return False


def smart_gen_duration(text: str) -> float:
    """Vypočíta odporúčanú gen audio duration (sec) z textu."""
    chars = len(text)
    if chars < 20:
        rate = 6.5  # extra slow for very short
    elif chars < 30:
        rate = 8.0
    elif chars < 60:
        rate = 11.0
    elif chars < 100:
        rate = 12.5
    else:
        rate = 13.5
    duration = chars / rate

    first = re.split(r'[\s,.!?]', text.strip(), 1)[0].lower().strip()
    if first in INTRO_WORDS:
        duration += 0.4

    pauses = text.count(',') * 0.15 + text.count(':') * 0.15
    duration += pauses
    return duration


def get_audio_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True
    )
    try:
        return float(r.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


class F5SKSmart:
    """F5 SK fine-tune wrapper s adaptívnou duration heuristikou."""

    def __init__(
        self,
        ckpt_path: str,
        vocab_path: str,
        ref_audio: str,
        ref_text: str,
        device: str = "cuda",
        use_ema: bool = False,
    ):
        from f5_tts.api import F5TTS
        self.f5 = F5TTS(
            model="F5TTS_v1_Base", ckpt_file=ckpt_path,
            vocab_file=vocab_path, device=device, use_ema=use_ema,
        )
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.ref_duration = get_audio_duration(ref_audio)

    def infer(
        self,
        text: str,
        out_path: str,
        seed: int = 42,
        cfg_strength: float = 2.0,
        nfe_step: int = 50,
        sway_sampling_coef: float = -1.0,
        force_default: bool = False,
        force_smart: bool = False,
    ) -> str:
        """Generate audio. Returns path."""
        kwargs = dict(
            gen_text=text, ref_file=self.ref_audio, ref_text=self.ref_text,
            cfg_strength=cfg_strength, nfe_step=nfe_step,
            sway_sampling_coef=sway_sampling_coef, seed=seed,
            file_wave=str(out_path),
        )
        use_smart = force_smart or (not force_default and needs_smart_duration(text))
        if use_smart:
            kwargs['fix_duration'] = self.ref_duration + smart_gen_duration(text)
        self.f5.infer(**kwargs)
        return str(out_path)
