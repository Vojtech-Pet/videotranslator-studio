"""
llama-cpp-server helper — open source OpenAI-compatible HTTP server.

Náhrada za LM Studio (closed-source proprietary). Používa `llama-cpp-python`
ktorý je už v requirements.txt. Spustí sa ako subprocess, exponuje rovnaký
OpenAI-compatible endpoint na `localhost:1234/v1/chat/completions` — pipeline
nemení nič, len namiesto LM Studio sa serve-uje cez built-in.

Použitie:
    from scripts.llamacpp_server import LlamaCppServer

    server = LlamaCppServer(model_path="/path/to/gemma-4-26b-q4km.gguf",
                            port=1234, n_gpu_layers=-1, n_ctx=8192)
    server.start()       # blokuje kým API nezačne odpovedať
    # ... pipeline robí translate cez localhost:1234/v1 ...
    server.stop()        # SIGTERM + wait

CLI:
    python scripts/llamacpp_server.py --model PATH [--port 1234] [--ctx 8192]
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class LlamaCppServer:
    """Manages a llama-cpp-python OpenAI-compatible server subprocess."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        port: int = 1234,
        host: str = "127.0.0.1",
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        chat_format: str = "gemma",
        python_exe: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model_path = str(model_path)
        self.port = port
        self.host = host
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.chat_format = chat_format
        self.python_exe = python_exe or sys.executable
        self.verbose = verbose
        self._proc: Optional[subprocess.Popen] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def is_running(self, timeout: float = 1.0) -> bool:
        """Check if the OpenAI API endpoint is alive."""
        try:
            import requests
            r = requests.get(f"{self.url}/models", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def start(self, ready_timeout: int = 120) -> None:
        """Spawn server subprocess. Block kým API neodpovie alebo timeout."""
        if self.is_running():
            print(f"[LLAMACPP] Server už beží na {self.url}", flush=True)
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"GGUF model nenájdený: {self.model_path}")

        cmd = [
            self.python_exe, "-m", "llama_cpp.server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--n_gpu_layers", str(self.n_gpu_layers),
            "--n_ctx", str(self.n_ctx),
            "--chat_format", self.chat_format,
        ]
        if not self.verbose:
            cmd += ["--verbose", "False"]

        print(f"[LLAMACPP] Spúšťam: {self.model_path}", flush=True)
        print(f"[LLAMACPP] Port: {self.port}, n_gpu_layers={self.n_gpu_layers}, n_ctx={self.n_ctx}", flush=True)

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL if not self.verbose else None,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        # Wait for API to come up (cold model load = slow)
        t0 = time.time()
        while time.time() - t0 < ready_timeout:
            if self._proc.poll() is not None:
                err = self._proc.stderr.read().decode("utf-8", errors="replace")[-1500:] if self._proc.stderr else ""
                raise RuntimeError(f"[LLAMACPP] Server padol pri štarte:\n{err}")
            if self.is_running(timeout=2.0):
                elapsed = time.time() - t0
                print(f"[LLAMACPP] Server ready on {self.url} ({elapsed:.1f}s)", flush=True)
                return
            time.sleep(2)
        raise TimeoutError(f"[LLAMACPP] Server sa neradzil za {ready_timeout}s")

    def stop(self, kill_timeout: float = 10.0) -> None:
        """Gracefully terminate the server subprocess."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc = None
            return
        print(f"[LLAMACPP] Stopping server (PID {self._proc.pid})...", flush=True)
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            else:
                self._proc.terminate()
            self._proc.wait(timeout=kill_timeout)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
        self._proc = None
        print(f"[LLAMACPP] Server stopped.", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def detect_default_model() -> Optional[Path]:
    """Auto-detect Gemma 4 26B GGUF — env override → paths → standard locations."""
    env_override = os.environ.get("VTS_LLAMACPP_MODEL", "")
    if env_override and Path(env_override).exists():
        return Path(env_override)
    try:
        from paths import PATHS
        candidate = Path(PATHS.gemma4_26b_base_q4km)
        if candidate.exists():
            return candidate
    except Exception:
        pass
    # Fallbacks — common locations
    for c in [
        Path.home() / "Ai/models/gemma-4-26B/gemma-4-26B-A4B-it-Q4_K_M.gguf",
        Path("/mnt/tts_data/VideoTranslator_studio/models/gemma-4-26B/gemma-4-26B-A4B-it-Q4_K_M.gguf"),
        Path.home() / ".local/share/videotranslator/models/gemma-4-26b/gemma-4-26B-A4B-it-Q4_K_M.gguf",
    ]:
        if c.exists():
            return c
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Start llama-cpp OpenAI-compatible server")
    ap.add_argument("--model", default="", help="GGUF model path (auto-detect ak prázdne)")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--n_gpu_layers", type=int, default=-1)
    ap.add_argument("--n_ctx", type=int, default=8192)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model = args.model or str(detect_default_model() or "")
    if not model:
        print("ERROR: GGUF model nenájdený. Set VTS_LLAMACPP_MODEL alebo --model", file=sys.stderr)
        sys.exit(1)

    server = LlamaCppServer(
        model_path=model, port=args.port, host=args.host,
        n_gpu_layers=args.n_gpu_layers, n_ctx=args.n_ctx, verbose=args.verbose,
    )
    try:
        server.start()
        print(f"[LLAMACPP] Press Ctrl+C to stop")
        if server._proc:
            server._proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
