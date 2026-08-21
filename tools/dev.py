#!/usr/bin/env python
"""Sobe o sistema inteiro em modo local -- sem Docker, sem Redis, sem MinIO.

Cada microsservico continua sendo um processo separado falando pelo
barramento; o que muda e so a implementacao por tras das interfaces (fila em
disco, storage em pasta, SQLite).

    python tools/dev.py            # tudo
    python tools/dev.py --only gateway editor
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVICES: dict[str, list[str]] = {
    "gateway": [
        sys.executable, "-m", "uvicorn", "app:app",
        "--app-dir", str(ROOT / "services" / "gateway"),
        "--host", "0.0.0.0", "--port", "8000",
    ],
    "preprocessor": [sys.executable, str(ROOT / "services" / "preprocessor" / "main.py")],
    "detector-kills": [sys.executable, str(ROOT / "services" / "detector_kills" / "main.py")],
    "detector-survival": [sys.executable, str(ROOT / "services" / "detector_survival" / "main.py")],
    "detector-ults": [sys.executable, str(ROOT / "services" / "detector_ults" / "main.py")],
    "detector-banner": [sys.executable, str(ROOT / "services" / "detector_banner" / "main.py")],
    "planner": [sys.executable, str(ROOT / "services" / "planner" / "main.py")],
    "thumbs": [sys.executable, str(ROOT / "services" / "thumbs" / "main.py")],
    "beats": [sys.executable, str(ROOT / "services" / "beats" / "main.py")],
    "editor": [sys.executable, str(ROOT / "services" / "editor" / "main.py")],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="sobe so estes servicos")
    args = ap.parse_args()

    chosen = args.only or list(SERVICES)
    unknown = [c for c in chosen if c not in SERVICES]
    if unknown:
        print(f"servico desconhecido: {', '.join(unknown)}", file=sys.stderr)
        print(f"disponiveis: {', '.join(SERVICES)}", file=sys.stderr)
        return 2

    env = {**os.environ, "OW_MODE": "local", "PYTHONUNBUFFERED": "1"}
    procs: list[tuple[str, subprocess.Popen]] = []

    print("modo local: fila em disco + storage em pasta + SQLite\n")
    for name in chosen:
        p = subprocess.Popen(SERVICES[name], cwd=str(ROOT), env=env)
        procs.append((name, p))
        print(f"  [{p.pid:>6}] {name}")

    print("\napi em http://localhost:8000/docs   (ctrl+c encerra tudo)\n")

    try:
        while True:
            for name, p in procs:
                code = p.poll()
                if code is not None:
                    print(f"\n!! '{name}' terminou com codigo {code}", file=sys.stderr)
                    raise KeyboardInterrupt
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nencerrando...")
    finally:
        for _name, p in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 8
        for name, p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
