#!/usr/bin/env python
"""Downloads the Overwatch 2 ability icons into `templates/abilities/`.

The system recognises **which** ultimate the player used and **which ability**
they killed with by comparing the drawing that shows up on the HUD with that
ability's official icon. Cropping 270 icons by hand from your own recording is
not reasonable -- so they come from the web, once.

The hero list and the icon addresses come from the OverFast API
(<https://overfast-api.tekrop.fr>), which mirrors Blizzard's official heroes
page; the files themselves come from Blizzard's CDN. Since the API follows the
patches, a new hero enters the bank by running this script again -- there is no
hero list written in the repository to go stale.

    python tools/fetch_ability_icons.py            # only what is missing
    python tools/fetch_ability_icons.py --force    # download everything again

The icons are **game assets**: they stay out of version control, like the
others in `templates/`. Without them the system keeps working -- it detects
that an ultimate was used, it just cannot say whose it was.

## Why the image is stored black on white

Blizzard's icon is white on a transparent background: the drawing lives in the
alpha channel. On the HUD it shows up either black on a white disc (ultimate)
or white on a dark box (ability in the killfeed). Storing `255 - alpha` --
black on white -- gives a file that opens visibly in any viewer and that
`IconBank` turns into a mask with a single threshold.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
API = "https://overfast-api.tekrop.fr"
UA = {"User-Agent": "ow-editor/1.0 (+https://github.com/)"}


def _get(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _save_icon(raw: bytes, dest: Path) -> bool:
    """Stores the icon as a black mark on a white background.

    Without an alpha channel there is no way to separate the drawing from the
    background, and a template with its background attached would match
    anything -- so the file is refused.
    """
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), 255 - img[:, :, 3])
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "templates" / "abilities",
                    help="destination (default: templates/abilities)")
    ap.add_argument("--force", action="store_true", help="download again what already exists")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    try:
        heroes = json.loads(_get(f"{API}/heroes", args.timeout))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"could not list the heroes: {exc}", file=sys.stderr)
        return 1

    print(f"{len(heroes)} heroes in the API list")
    downloaded = skipped = failures = 0
    for h in heroes:
        key = h["key"]
        try:
            hero = json.loads(_get(f"{API}/heroes/{key}", args.timeout))
        except Exception as exc:  # one hero being unavailable must not stop the rest
            print(f"  {key}: could not read the abilities ({exc})", file=sys.stderr)
            failures += 1
            continue

        for ability in hero.get("abilities", []):
            dest = args.out / key / f"{_slug(ability['name'])}.png"
            if dest.exists() and not args.force:
                skipped += 1
                continue
            try:
                raw = _get(ability["icon"], args.timeout)
            except Exception as exc:
                print(f"  {dest.name}: {exc}", file=sys.stderr)
                failures += 1
                continue
            if _save_icon(raw, dest):
                downloaded += 1
            else:
                print(f"  {dest.name}: icon without an alpha channel, skipped", file=sys.stderr)
                failures += 1
        print(f"  {key:15s} {len(list((args.out / key).glob('*.png')))} icon(s)")

    print(f"\n{downloaded} downloaded, {skipped} already there, {failures} failure(s)")
    print(f"destination: {args.out}")
    return 1 if downloaded == 0 and failures else 0


if __name__ == "__main__":
    sys.exit(main())
