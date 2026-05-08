"""
VideoTranslator — entry point.

Usage:
    python main.py --input video.mp4 --tgt_lang ces --use_chatterbox --timed --timeline_mode --vad
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from pipeline import main

if __name__ == "__main__":
    main()
