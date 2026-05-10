#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path


SPOOL_DIR = Path("/tmp/epson-l850-app-jobs")
APP = Path("/mnt/tts_data/VideoTranslator_studio/epson_l850_print.py")


def open_job(path: Path) -> None:
    claimed = path.with_suffix(path.suffix + ".opened")
    try:
        path.rename(claimed)
    except OSError:
        return
    subprocess.Popen(["python3", str(APP), str(claimed)])


def main() -> None:
    SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        SPOOL_DIR.chmod(0o777)
    except PermissionError:
        pass
    seen: set[Path] = set()
    while True:
        for path in sorted(SPOOL_DIR.glob("*.print")):
            if path not in seen:
                seen.add(path)
                open_job(path)
        time.sleep(1)


if __name__ == "__main__":
    main()
