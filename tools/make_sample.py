#!/usr/bin/env python
"""Gera um video sintetico que imita a HUD do Overwatch 2, com um JSON de
gabarito (timestamps exatos de cada evento).

Serve para dois propositos:

1. testar o pipeline inteiro de ponta a ponta sem precisar de gameplay real;
2. dar ao usuario um jeito de ver o sistema funcionando antes de gravar.

Uso:
    python tools/make_sample.py --out data/sample/match.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30
SR = 44100

# gabarito (segundos)
KILLS = [
    5.0, 6.2, 7.4,            # rajada de 3  -> MULTIKILL
    20.0, 21.0, 22.2, 23.1,   # rajada de 4  -> SOLO_WIPE
    34.0,                     # avulsa       -> montagem
    41.5,                     # avulsa       -> montagem
    52.0,                     # avulsa       -> montagem
]
# Ao morrer no OW2 voce passa a espectar um companheiro: a vida zera por um
# instante e volta cheia. E essa a assinatura que o detector procura.
DEATHS = [28.0]
LOW_HP = [(12.0, 2.5), (15.5, 2.0), (45.0, 2.2)]  # episodios sobrevividos
LOW_HP_FRAC = 0.18
FULL_HP_FRAC = 0.95
ULTS = [40.0, 51.0]                               # ult inimiga (audio + icone)
# habilidades anunciadas no rodape, e avisos-isca com o mesmo formato
SLEEPS = [18.0, 30.0, 48.0]          # dardo da Ana, faixa ciano
STUNS = [22.0, 38.0]                 # pedrada do Sigma, faixa VERDE de proposito:
                                     # a cor do aviso muda de gravacao para
                                     # gravacao, e o pipeline tem de aguentar
AVISOS_ISCA = [25.0, 55.0]
BANNER_S = 2.5
SKULL_DURATION = 0.9

# ultimate DO JOGADOR: o instante em que ela e usada. O botao fica carregado
# nos segundos anteriores e apaga aqui -- e a borda de descida que vira evento.
SELF_ULTS = [24.0, 47.0]
SELF_ULT_CHARGE_S = 6.0
# armadilha: o botao aparece carregado por um instante e some. No jogo isso e a
# kill cam (um disco com o rosto de quem matou, no mesmo lugar) ou um clarao de
# explosao; em 27 min de gravacao real toda faixa curta assim era falsa, e
# ultimate de verdade fica carregada segundos antes de ser usada. Longe de
# `SELF_ULTS` o bastante para `min_after_s` nao ser quem a descarta: quem tem
# de recusa-la e `min_charged_s`.
ULT_FLASHES = [30.0]
ULT_FLASH_S = 0.4
# acerto critico: marcador em X vermelho na mira. Longe das eliminacoes de
# proposito -- a caveira cobre as mesmas diagonais, e o detector avisa que
# nesse encontro ela vence.
HEADSHOTS = [10.0, 33.0, 44.0]
HEADSHOT_S = 0.35
# eliminacao com habilidade anunciada no killfeed. A linha fica na tela por
# varios segundos: o evento e ela APARECER, nao ela estar la.
# Longe das janelas de LOW_HP: a vinheta de dano do video sintetico e bem
# mais forte que a do jogo e cobre o canto do killfeed, apagando a placa
# vermelha. Isso e limitacao do desenho de teste, nao do detector.
ABILITY_KILLS = [26.0, 36.0]
ABILITY_ROW_S = 6.0
# a mesma linha de killfeed, mas com OUTRO nome na placa de quem matou: um
# colega de time eliminando com habilidade. O killfeed anuncia as dez pessoas
# da partida, e so as do jogador viram material da montagem -- estas existem
# para o detector ter de recusa-las.
#
# `PATRICK` tem as mesmas 7 letras de `PLAYER_NAME` de proposito: com nomes de
# comprimentos diferentes a comparacao acertaria pela contagem, sem nunca olhar
# o desenho das letras, e o teste passaria mesmo com o desenho quebrado.
TEAMMATE_KILLS = [16.0, 51.0]
PLAYER_NAME = "JOGADOR"
TEAMMATE_NAME = "PATRICK"

DURATION = 60.0


# ------------------------------- desenho ------------------------------------


def draw_skull(img: np.ndarray, cx: int, cy: int, r: int, alpha: float) -> None:
    """Caveira de eliminacao, na mira.

    Cor e posicao medidas em gameplay real: magenta bem saturado (HSV ~167,
    230, 235) centrado em (0.50, 0.485) da tela -- ligeiramente *acima* do meio.
    """
    layer = img.copy()
    red = (115, 23, 235)  # BGR do magenta da HUD
    cv2.circle(layer, (cx, cy - r // 5), r, red, -1)
    cv2.rectangle(layer, (cx - r // 2, cy + r // 2), (cx + r // 2, cy + r), red, -1)
    dark = (10, 10, 40)
    eye = max(2, r // 4)
    cv2.circle(layer, (cx - r // 2, cy - r // 4), eye, dark, -1)
    cv2.circle(layer, (cx + r // 2, cy - r // 4), eye, dark, -1)
    cv2.rectangle(layer, (cx - eye // 2, cy + r // 3), (cx + eye // 2, cy + r), dark, -1)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


def draw_ult_icon(img: np.ndarray, x: int, y: int) -> None:
    """Icone no killfeed usado como ultimate: losango laranja com um anel."""
    pts = np.array([[x, y - 18], [x + 18, y], [x, y + 18], [x - 18, y]], np.int32)
    cv2.fillPoly(img, [pts], (30, 150, 255))
    cv2.circle(img, (x, y), 8, (255, 255, 255), 2)


def ult_template() -> np.ndarray:
    tpl = np.zeros((44, 44, 3), np.uint8)
    draw_ult_icon(tpl, 22, 22)
    return tpl


# ---------------------- botao de ultimate e killfeed ------------------------
#
# Estas tres coisas -- botao de ultimate carregado, marcador de acerto critico
# e linha de killfeed -- sao as que os detectores novos leem. Aqui elas sao
# desenhadas com a mesma geometria que a HUD real usa, medida nas gravacoes de
# referencia; o que se testa e a canalizacao (achar a regiao, recortar o
# glifo, casar, virar evento), nao a precisao do casamento, que foi medida em
# gameplay de verdade.

#: as marcas usadas no video de exemplo. Sao poligonos e nao arquivos porque o
#: mesmo desenho precisa sair em dois lugares -- na tela e no banco de icones
#: contra o qual o detector compara --, e desenha-lo garante que os dois
#: combinem sem depender de nenhum asset do jogo.
GLIFOS: dict[str, list[tuple[float, float]]] = {
    # seta larga para cima, com entalhe: assimetrica na vertical e na
    # horizontal, entao nao casa consigo mesma girada
    "self_ult": [(0.5, 0.05), (0.95, 0.55), (0.68, 0.55), (0.68, 0.95),
                 (0.32, 0.95), (0.32, 0.55), (0.05, 0.55)],
    # ampulheta deitada
    "ability_kill": [(0.05, 0.08), (0.05, 0.92), (0.5, 0.5),
                     (0.95, 0.92), (0.95, 0.08), (0.5, 0.5)],
}


def glyph_mask(key: str, side: int) -> np.ndarray:
    """A marca, branca sobre preto, no tamanho pedido."""
    m = np.zeros((side, side), np.uint8)
    pts = np.array([[int(x * side), int(y * side)] for x, y in GLIFOS[key]], np.int32)
    cv2.fillPoly(m, [pts], 255)
    return m


def glyph_template(key: str, side: int = 128) -> np.ndarray:
    """A mesma marca no formato do banco de icones: preta sobre branco."""
    return 255 - glyph_mask(key, side)


def _stamp(img: np.ndarray, mask: np.ndarray, x: int, y: int,
           cor: tuple[int, int, int]) -> None:
    """Pinta a marca por transparencia -- e nao cola um quadrado opaco."""
    h, w = mask.shape
    alvo = img[y:y + h, x:x + w]
    if alvo.shape[:2] != mask.shape:
        return
    alvo[mask > 0] = cor


#: geometria do botao de ultimate, igual a do perfil `ow2_default`
ULT_CX, ULT_CY = 0.5, 0.86
ULT_DISC_R = 26
ULT_RING_R0, ULT_RING_R1 = 30, 38


def draw_ult_button(img: np.ndarray, carregada: bool) -> None:
    """O botao do rodape nos seus dois estados.

    Carregado e um disco BRANCO com a marca do heroi em preto, cercado por um
    anel CIANO; descarregado e so um anel escuro. O detector exige as duas
    coisas juntas, entao desenhar so uma delas nao produziria evento -- e e
    justamente isso que faz do estado descarregado uma armadilha util.
    """
    cx, cy = int(ULT_CX * W), int(ULT_CY * H)
    if not carregada:
        cv2.circle(img, (cx, cy), ULT_DISC_R + 4, (210, 210, 210), 2)
        return
    cv2.circle(img, (cx, cy), ULT_RING_R1, (235, 190, 60), -1)   # ciano BGR
    cv2.circle(img, (cx, cy), ULT_RING_R0, (40, 45, 50), -1)
    cv2.circle(img, (cx, cy), ULT_DISC_R, (250, 250, 250), -1)
    lado = int(ULT_DISC_R * 1.15)
    _stamp(img, glyph_mask("self_ult", lado),
           cx - lado // 2, cy - lado // 2, (20, 20, 20))


def draw_crit_marker(img: np.ndarray) -> None:
    """Marcador de acerto critico: quatro tracos vermelhos em X na mira.

    As quatro direcoes RETAS ficam limpas de proposito -- e a diferenca entre
    este marcador e a caveira de eliminacao, que preenche as oito.
    """
    cx, cy = W // 2, H // 2
    r0, r1 = 14, 40
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        p0 = (cx + int(dx * r0 * 0.71), cy + int(dy * r0 * 0.71))
        p1 = (cx + int(dx * r1 * 0.71), cy + int(dy * r1 * 0.71))
        cv2.line(img, p0, p1, (60, 40, 235), 5)


#: geometria de uma linha do killfeed, dentro da ROI `killfeed`
KF_Y0, KF_H = 30, 26
KF_ALLY = (900, 1060)     # placa ciano: quem matou
KF_ENEMY = (1120, 1270)   # placa vermelha: quem morreu


def draw_killfeed_row(img: np.ndarray, vitima: int = 0,
                      killer: str = PLAYER_NAME) -> None:
    """`[ placa de quem matou ] [ icone ] > [ placa de quem morreu ]`.

    `vitima` encurta a placa vermelha. No jogo o comprimento de
    cada placa e o do nome escrito nela, entao duas eliminacoes so tem a mesma
    largura se forem do mesmo jogador **sobre a mesma vitima**; desenhar todas
    identicas faria o exemplo testar um killfeed que nao existe. E e por essa
    largura que o detector reconhece uma linha de um quadro para o outro.

    `killer` e o nome escrito na placa azul. A cor NAO diz de quem foi a
    eliminacao -- azul e quem matou e vermelha e quem morreu, dos dois lados da
    partida --, entao e esse nome, e so ele, que separa a eliminacao do jogador
    da do colega de time.
    """
    y0, y1 = KF_Y0, KF_Y0 + KF_H
    cv2.rectangle(img, (KF_ALLY[0], y0), (KF_ALLY[1], y1), (190, 140, 70), -1)
    _draw_name(img, killer, KF_ALLY[0] + 6, y0, KF_H, 0.5, 2)
    cv2.rectangle(img, (KF_ENEMY[0], y0), (KF_ENEMY[1] + vitima, y1),
                  (90, 60, 200), -1)
    # a caixinha do icone e o chevron ocupam o vao entre as duas placas
    cv2.rectangle(img, (KF_ALLY[1] + 4, y0 - 3), (KF_ALLY[1] + 36, y1 + 3),
                  (55, 52, 50), -1)
    lado = 22
    _stamp(img, glyph_mask("ability_kill", lado),
           KF_ALLY[1] + 9, (y0 + y1) // 2 - lado // 2, (240, 240, 240))
    # o `>` ocupa o ULTIMO quarto do vao, como na HUD real: medida na gravacao
    # de referencia, a caixinha do icone vai ate ~0.73 do vao e o chevron
    # comeca ali. `killfeed.icon_span` para em 0.66 justamente para nao
    # encostar nele -- desenhar o chevron mais a esquerda que no jogo faria o
    # video de exemplo testar um layout que nao existe.
    cv2.putText(img, ">", (KF_ALLY[1] + int(0.76 * (KF_ENEMY[0] - KF_ALLY[1])), y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)


#: geometria da faixa de aviso, igual a do perfil `ow2_default`
BANNER_Y0, BANNER_Y1 = 0.695, 0.727
BANNER_W = 0.167


#: cores de faixa, em BGR, dentro das duas faixas HSV do profile
COR_CIANO = (196, 150, 74)
COR_VERDE = (110, 165, 60)


def _icone(nome: str) -> np.ndarray:
    """Carrega um molde real usado pelo detector, em tons de cinza.

    O video sintetico existe para exercitar o *pipeline* -- achar a faixa,
    recortar o icone na posicao certa, casar e virar evento. A precisao do
    casamento em si foi medida em gameplay real; aqui o que se testa e a
    canalizacao, e para isso o icone tem de ser o mesmo que o detector procura.
    """
    caminho = Path(__file__).resolve().parents[1] / "config" / "shapes" / nome
    img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"molde de icone nao encontrado: {caminho}")
    return img


def draw_banner(
    img: np.ndarray,
    texto: str,
    icone: np.ndarray | None,
    cor: tuple[int, int, int] = COR_CIANO,
) -> None:
    """Faixa de aviso do rodape. Com `icone` None desenha uma isca: mesma faixa,
    mesma cor, mesma posicao -- outro simbolo."""
    y0, y1 = int(BANNER_Y0 * H), int(BANNER_Y1 * H)
    largura = int(BANNER_W * W)
    x0 = W // 2 - largura // 2
    alt = y1 - y0
    cv2.rectangle(img, (x0, y0), (x0 + largura, y1), cor, -1)
    lado = alt
    ax = x0 + int(alt * 0.18)
    if icone is not None:
        # composto por transparencia, e nao colado como um quadrado opaco: na
        # HUD do OW2 o icone e um desenho claro POR CIMA da faixa colorida.
        # Colar o molde inteiro punha uma moldura escura em volta dele, e a
        # normalizacao de contraste do detector passava a enxergar essa moldura
        # em vez do desenho -- o recorte deixava de parecer o proprio molde.
        alpha = (
            cv2.resize(icone, (lado, lado), interpolation=cv2.INTER_AREA)
            .astype(np.float32) / 255.0
        )[:, :, None]
        fundo = img[y0:y0 + lado, ax:ax + lado].astype(np.float32)
        img[y0:y0 + lado, ax:ax + lado] = (
            fundo * (1.0 - alpha) + 255.0 * alpha
        ).astype(np.uint8)
    else:
        cv2.circle(img, (ax + lado // 2, y0 + lado // 2), lado // 3,
                   (255, 255, 255), 2)
    cv2.putText(img, texto, (ax + lado + 6, y1 - int(alt * 0.28)),
                cv2.FONT_HERSHEY_SIMPLEX, alt / 46.0, (255, 255, 255), 1)


def background(t: float) -> np.ndarray:
    """Cenario jogavel: gradiente esverdeado + blobs em movimento.

    A paleta e escolhida para ficar longe das duas cores que os detectores
    procuram: o magenta da caveira de eliminacao (hue 156-178) e o ciano da
    faixa de aviso do rodape (hue 90-115). Assim, um evento detectado no video
    sintetico so pode ter vindo da HUD desenhada, nunca do cenario.
    """
    yy = np.linspace(60, 150, H, dtype=np.float32)[:, None]
    xx = np.linspace(40, 120, W, dtype=np.float32)[None, :]
    img = np.zeros((H, W, 3), np.float32)
    img[:, :, 0] = 30 + xx * 0.12               # azul baixo
    img[:, :, 1] = yy * 0.85 + xx * 0.35        # verde dominante
    img[:, :, 2] = 50 + xx * 0.30               # vermelho medio
    img = np.clip(img, 0, 255).astype(np.uint8)

    for k in range(6):
        px = int((math.sin(t * 0.6 + k) * 0.4 + 0.5) * W)
        py = int((math.cos(t * 0.45 + k * 1.7) * 0.4 + 0.5) * H)
        cv2.circle(img, (px, py), 60 + k * 12, (55, 130 + k * 12, 85), -1)
    rng = np.random.default_rng(int(t * FPS))
    noise = rng.integers(-8, 8, (H, W, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


#: geometria da barra de vida, igual a do perfil `ow2_default`
BAR_X0, BAR_X1 = 0.09, 0.225
BAR_Y0, BAR_Y1 = 0.855, 0.895
N_TICKS = 26


def draw_hud(img: np.ndarray, hp: float | None) -> None:
    # mira
    cv2.line(img, (W // 2 - 12, H // 2), (W // 2 + 12, H // 2), (230, 230, 230), 2)
    cv2.line(img, (W // 2, H // 2 - 12), (W // 2, H // 2 + 12), (230, 230, 230), 2)
    if hp is None:
        return  # sem HUD: menu, troca de round

    # a placa do jogador acompanha o resto da HUD: some junto no menu e na
    # troca de round, como no jogo
    draw_player_card(img)

    # Barra de vida como o OW2 desenha: tracinhos verticais claros sobre um
    # trilho escuro, com a largura total normalizada. E a *alternancia* desses
    # tracinhos que o detector le -- por isso eles precisam existir de verdade,
    # e nao serem um retangulo solido.
    x0, x1 = int(BAR_X0 * W), int(BAR_X1 * W)
    y0, y1 = int(BAR_Y0 * H), int(BAR_Y1 * H)
    cv2.rectangle(img, (x0, y0), (x1, y1), (38, 34, 30), -1)
    step = (x1 - x0) / N_TICKS
    filled = int(round(N_TICKS * max(0.0, min(1.0, hp))))
    for i in range(filled):
        tx = int(x0 + i * step)
        cv2.rectangle(img, (tx + 1, y0 + 2), (int(tx + step) - 1, y1 - 2),
                      (235, 240, 240), -1)


#: a placa do proprio jogador, igual a do perfil `ow2_default` (roi `player`).
#: Um pouco por dentro da ROI: no jogo a placa nao encosta na borda do recorte,
#: e uma letra colada na borda sai partida do recorte.
CARD_X0, CARD_X1 = 0.095, 0.232
CARD_Y0, CARD_Y1 = 0.899, 0.929


def draw_player_card(img: np.ndarray, nome: str = PLAYER_NAME) -> None:
    """A placa com o nome do jogador.

    E o unico lugar da tela que diz quem esta jogando. O detector de killfeed
    le daqui para saber quais das eliminacoes anunciadas sao do usuario.

    A escala da letra e maior que a do killfeed de proposito: na HUD de verdade
    as duas escritas tem tamanhos e espacamentos diferentes, e comparar duas
    escritas identicas faria o exemplo testar uma comparacao que nao existe.
    """
    x0, x1 = int(CARD_X0 * W), int(CARD_X1 * W)
    y0, y1 = int(CARD_Y0 * H), int(CARD_Y1 * H)
    cv2.rectangle(img, (x0, y0), (x1, y1), (120, 62, 44), -1)
    _draw_name(img, nome, x0 + int((x1 - x0) * 0.22), y0, y1 - y0, 0.62, 2)


def _draw_name(img: np.ndarray, nome: str, x: int, y: int, alt: int,
               escala: float, grossura: int) -> None:
    """Escreve o nome numa placa, centrado na vertical.

    Abaixo de 0.5 de escala as letras da Hershey se encostam e viram uma so --
    o que ja e um nome diferente. Medido no tamanho da placa do killfeed: com
    0.45 e grossura 1, `JOGADOR` sai com 6 letras em vez de 7; com 0.5 e
    grossura 2 sai com as 7 e casa com a placa do rodape em 0.70, contra 0.24
    de `PATRICK`, que tem o mesmo comprimento.
    """
    (_tw, th), _ = cv2.getTextSize(nome, cv2.FONT_HERSHEY_SIMPLEX, escala, grossura)
    cv2.putText(img, nome, (x, y + (alt + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                escala, (240, 240, 240), grossura, cv2.LINE_AA)


def draw_damage_vignette(img: np.ndarray, strength: float) -> None:
    """Vinheta vermelha nas bordas -- no OW2 isto e *dano recebido*, nao vida
    baixa. Fica aqui de proposito: numa partida real ela pisca o tempo todo, e
    o detector precisa continuar acertando mesmo com ela na tela."""
    band_y, band_x = int(H * 0.11), int(W * 0.11)
    layer = img.copy()
    cv2.rectangle(layer, (0, 0), (W, band_y), (30, 30, 245), -1)
    cv2.rectangle(layer, (0, H - band_y), (W, H), (30, 30, 245), -1)
    cv2.rectangle(layer, (0, 0), (band_x, H), (30, 30, 245), -1)
    cv2.rectangle(layer, (W - band_x, 0), (W, H), (30, 30, 245), -1)
    cv2.addWeighted(layer, 0.75 * strength, img, 1 - 0.75 * strength, 0, img)


def health_at(t: float) -> float | None:
    """Vida no instante t. None = sem HUD na tela."""
    for d in DEATHS:
        if d <= t < d + 0.35:
            return 0.0
    for start, length in LOW_HP:
        if start <= t < start + length:
            return LOW_HP_FRAC
    return FULL_HP_FRAC


def frame_at(t: float) -> np.ndarray:
    img = background(t)
    hp = health_at(t)

    # A vinheta de dano acompanha os momentos de vida baixa, como no jogo --
    # e serve de armadilha: ela e vermelha e cobre as bordas, mas nao deve
    # produzir evento nenhum sozinha.
    for s, ln in LOW_HP:
        if s <= t < s + ln:
            draw_damage_vignette(img, 0.6 + 0.4 * abs(math.sin((t - s) * 6.0)))
    draw_hud(img, hp)

    for k in KILLS:
        if k <= t < k + SKULL_DURATION:
            phase = (t - k) / SKULL_DURATION
            alpha = min(1.0, (1.0 - phase) * 2.2)
            # centrada em (0.50, 0.485), como medido em gameplay real
            draw_skull(img, W // 2, int(H * 0.485), 34, alpha)

    for s0 in SLEEPS:
        if s0 <= t < s0 + BANNER_S:
            draw_banner(img, "PUT MERCY (TESTE) TO SLEEP",
                        _icone("ana_sleep_icon.png"), COR_CIANO)
    for s0 in STUNS:
        if s0 <= t < s0 + BANNER_S:
            draw_banner(img, "ORISA (TESTE) STUNNED BY ACCRETION",
                        _icone("sigma_accretion_icon.png"), COR_VERDE)
    for s0 in AVISOS_ISCA:
        if s0 <= t < s0 + BANNER_S:
            draw_banner(img, "ORB OF HARMONY FROM TESTE", None, COR_CIANO)

    for u in ULTS:
        if u <= t < u + 2.5:
            draw_ult_icon(img, int(W * 0.80), int(H * 0.12))
            cv2.putText(img, "ULTIMATE", (int(W * 0.66), int(H * 0.20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

    draw_ult_button(img, ult_carregada(t))
    for hs in HEADSHOTS:
        if hs <= t < hs + HEADSHOT_S:
            draw_crit_marker(img)
    for i, ak in enumerate(ABILITY_KILLS):
        if ak <= t < ak + ABILITY_ROW_S:
            # vitimas diferentes, placas de comprimentos diferentes
            draw_killfeed_row(img, vitima=-40 * i)
    for i, tk in enumerate(TEAMMATE_KILLS):
        if tk <= t < tk + ABILITY_ROW_S:
            # larguras que nao coincidem com as de ABILITY_KILLS: no jogo o
            # comprimento da placa e o do nome escrito nela, e duas linhas so
            # tem a mesma largura quando sao a mesma dupla. Desenhar duas
            # eliminacoes diferentes com a mesma largura, a menos de `hold_s`
            # uma da outra, faria o rastreador ve-las como uma linha so que
            # sumiu e voltou -- que e o que ele existe para nao confundir.
            draw_killfeed_row(img, vitima=-60 - 35 * i, killer=TEAMMATE_NAME)
    return img


def ult_carregada(t: float) -> bool:
    """O botao fica carregado nos segundos que antecedem cada uso.

    E tambem nas piscadas de `ULT_FLASHES`, que nao sao uso nenhum: sao a
    armadilha que o detector tem de recusar pela duracao.
    """
    if any(u - SELF_ULT_CHARGE_S <= t < u for u in SELF_ULTS):
        return True
    return any(f <= t < f + ULT_FLASH_S for f in ULT_FLASHES)


# -------------------------------- audio -------------------------------------


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def game_audio(duration: float) -> np.ndarray:
    """Ambiente baixo + estouros altos nas ults (pista de audio dos detectores)."""
    n = int(duration * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    sig = 0.02 * rng.standard_normal(n) + 0.02 * np.sin(2 * np.pi * 110 * t)
    for u in ULTS:
        i0 = int(u * SR)
        if i0 >= n:
            continue  # video mais curto que o gabarito completo
        i1 = min(n, i0 + int(1.2 * SR))
        env = np.exp(-np.linspace(0, 5, i1 - i0))
        sig[i0:i1] += 0.8 * env * np.sin(2 * np.pi * 220 * t[i0:i1])
    return sig


def click_track(duration: float, bpm: float = 120.0) -> np.ndarray:
    """Musica de teste: bumbo no tempo + baixo. BPM conhecido."""
    n = int(duration * SR)
    t = np.arange(n) / SR
    sig = 0.10 * np.sin(2 * np.pi * 55 * t)
    period = 60.0 / bpm
    beat = 0.0
    while beat < duration:
        i0 = int(beat * SR)
        i1 = min(n, i0 + int(0.12 * SR))
        env = np.exp(-np.linspace(0, 14, i1 - i0))
        tt = t[i0:i1] - beat
        sig[i0:i1] += 0.9 * env * np.sin(2 * np.pi * 65 * tt)
        sig[i0:i1] += 0.35 * env * np.sin(2 * np.pi * 2500 * tt)
        beat += period
    return sig


# ------------------------------- geracao ------------------------------------


def render(out: Path, duration: float, ffmpeg: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_path = out.with_name("game_audio.wav")
    write_wav(audio_path, game_audio(duration))

    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a", "-shortest",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = int(duration * FPS)
    assert proc.stdin is not None
    for i in range(total):
        proc.stdin.write(frame_at(i / FPS).tobytes())
        if i % (FPS * 5) == 0:
            print(f"  {i / FPS:5.1f}s / {duration:.0f}s", file=sys.stderr)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg falhou ao montar o video de exemplo")
    audio_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/sample/match.mp4"))
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--music", type=Path, default=None,
                    help="tambem gera uma musica de teste (bumbo em 120 BPM)")
    ap.add_argument("--ult-templates", type=Path, default=None,
                    help="salva o icone de ult usado, para calibrar o detector")
    ap.add_argument("--ability-icons", type=Path, default=None,
                    help="salva as marcas do botao de ult e do killfeed no "
                         "formato de templates/abilities/, para o detector "
                         "poder dizer QUAL habilidade era")
    args = ap.parse_args()

    print(f"gerando {args.out} ({args.duration:.0f}s)...", file=sys.stderr)
    render(args.out, args.duration, args.ffmpeg)

    truth = {
        "duration_s": args.duration,
        "fps": FPS,
        "size": [W, H],
        "kills": KILLS,
        "deaths": list(DEATHS),
        "low_hp": [d[0] for d in LOW_HP],
        "ults": ULTS,
        "sleeps": SLEEPS,
        "stuns": STUNS,
        "avisos_isca": AVISOS_ISCA,
        "self_ults": SELF_ULTS,
        "ult_flashes": ULT_FLASHES,
        "headshots": HEADSHOTS,
        "ability_kills": ABILITY_KILLS,
        # as do colega de time: o killfeed as anuncia igual, e o detector tem
        # de recusa-las pelo nome escrito na placa
        "teammate_kills": TEAMMATE_KILLS,
        "player_name": PLAYER_NAME,
    }
    truth_path = args.out.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

    if args.music:
        args.music.parent.mkdir(parents=True, exist_ok=True)
        write_wav(args.music, click_track(args.duration, 120.0))
        print(f"musica de teste: {args.music}", file=sys.stderr)

    if args.ult_templates:
        args.ult_templates.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.ult_templates / "sample_ult.png"), ult_template())
        print(f"template de ult: {args.ult_templates}", file=sys.stderr)

    if args.ability_icons:
        # a estrutura e a mesma de templates/abilities/: uma pasta por heroi
        heroi = args.ability_icons / "sample"
        heroi.mkdir(parents=True, exist_ok=True)
        for key in GLIFOS:
            cv2.imwrite(str(heroi / f"{key}.png"), glyph_template(key))
        print(f"icones de habilidade: {heroi}", file=sys.stderr)

    print(f"pronto. gabarito em {truth_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
