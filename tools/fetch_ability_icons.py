#!/usr/bin/env python
"""Baixa os ícones de habilidade do Overwatch 2 para `templates/abilities/`.

O sistema reconhece **qual** ultimate o jogador usou e **com que habilidade**
ele matou comparando o desenho que aparece na HUD com o ícone oficial daquela
habilidade. Recortar 270 ícones à mão da própria gravação não é razoável --
então eles vêm da web, uma vez só.

A lista de heróis e os endereços dos ícones saem da OverFast API
(<https://overfast-api.tekrop.fr>), que espelha a página oficial de heróis da
Blizzard; os arquivos em si vêm do CDN da Blizzard. Como a API acompanha os
patches, um herói novo entra no banco rodando este script de novo -- não há
lista de heróis escrita no repositório para ficar desatualizada.

    python tools/fetch_ability_icons.py            # só o que falta
    python tools/fetch_ability_icons.py --force    # rebaixa tudo

Os ícones são **assets do jogo**: ficam fora do controle de versão, como os
outros em `templates/`. Sem eles o sistema continua funcionando -- ele detecta
que a ultimate foi usada, só não sabe dizer de quem era.

## Por que a imagem é gravada em preto sobre branco

O ícone da Blizzard é branco com fundo transparente: o desenho vive no canal
alfa. Na HUD ele aparece ora preto sobre disco branco (ultimate), ora branco
sobre caixa escura (habilidade no killfeed). Guardar `255 - alfa` -- preto
sobre branco -- dá um arquivo que abre visível em qualquer visualizador e que
o `IconBank` transforma em máscara com um único limiar.
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
    """Grava o ícone como marca preta sobre fundo branco.

    Sem canal alfa não há como separar o desenho do fundo, e um molde com fundo
    junto casaria com qualquer coisa -- então o arquivo é recusado.
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
                    help="destino (padrão: templates/abilities)")
    ap.add_argument("--force", action="store_true", help="rebaixa o que já existe")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    try:
        heroes = json.loads(_get(f"{API}/heroes", args.timeout))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"não consegui listar os heróis: {exc}", file=sys.stderr)
        return 1

    print(f"{len(heroes)} heróis na lista da API")
    baixados = pulados = falhas = 0
    for h in heroes:
        key = h["key"]
        try:
            hero = json.loads(_get(f"{API}/heroes/{key}", args.timeout))
        except Exception as exc:  # um herói fora do ar não pode derrubar o resto
            print(f"  {key}: não consegui ler as habilidades ({exc})", file=sys.stderr)
            falhas += 1
            continue

        for ability in hero.get("abilities", []):
            dest = args.out / key / f"{_slug(ability['name'])}.png"
            if dest.exists() and not args.force:
                pulados += 1
                continue
            try:
                raw = _get(ability["icon"], args.timeout)
            except Exception as exc:
                print(f"  {dest.name}: {exc}", file=sys.stderr)
                falhas += 1
                continue
            if _save_icon(raw, dest):
                baixados += 1
            else:
                print(f"  {dest.name}: ícone sem canal alfa, ignorado", file=sys.stderr)
                falhas += 1
        print(f"  {key:15s} {len(list((args.out / key).glob('*.png')))} ícone(s)")

    print(f"\n{baixados} baixado(s), {pulados} já estava(m) lá, {falhas} falha(s)")
    print(f"destino: {args.out}")
    return 1 if baixados == 0 and falhas else 0


if __name__ == "__main__":
    sys.exit(main())
