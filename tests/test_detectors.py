"""Precisao dos detectores contra o gabarito do video sintetico.

E o teste que sustenta a tese do projeto: recortar uma faixa minuscula da tela,
em FPS baixo, basta para recuperar os eventos da partida.
"""

from __future__ import annotations

import json

import pytest

from conftest import (
    ABILITY_ICONS,
    MUSIC,
    SAMPLE,
    TRUTH,
    ULT_TEMPLATES,
    needs_sample,
    service_module,
)
from owcore.ffmpeg import extract_audio, extract_rois, probe
from owcore.models import EventKind
from owcore.profiles import load_profile

TOLERANCIA_S = 0.35

pytestmark = needs_sample


@pytest.fixture(scope="module")
def truth() -> dict:
    return json.loads(TRUTH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rois(tmp_path_factory):
    prof = load_profile("ow2_default")
    out = tmp_path_factory.mktemp("rois")
    crops = extract_rois(
        SAMPLE,
        prof.rois(["kills", "health", "killfeed", "banner", "ult", "player"]),
        out,
    )
    crops["audio"] = extract_audio(SAMPLE, out / "audio.wav")
    return crops


def casam(detectados: list[float], esperados: list[float]) -> bool:
    if len(detectados) != len(esperados):
        return False
    return all(
        abs(d - e) <= TOLERANCIA_S
        for d, e in zip(sorted(detectados), sorted(esperados))
    )


def test_recorte_e_muito_menor_que_o_original(rois):
    """O ponto da arquitetura: o detector recebe uma fracao dos bytes."""
    assert rois["kills"].stat().st_size < SAMPLE.stat().st_size / 50


def test_kills_batem_com_o_gabarito(rois, truth):
    detect = service_module("detector_kills")
    ev = detect.detect_kills(rois["kills"], load_profile("ow2_default"))
    assert [e.kind for e in ev] == [EventKind.KILL] * len(ev)
    assert casam([e.t for e in ev], truth["kills"])


def test_sobrevivencia_bate_com_o_gabarito(rois, truth):
    detect = service_module("detector_survival")
    ev = detect.detect_survival(rois["health"], load_profile("ow2_default"))

    low = [e.t for e in ev if e.kind == EventKind.LOW_HP]
    deaths = [e.t for e in ev if e.kind == EventKind.DEATH]
    escapes = [e.t for e in ev if e.kind == EventKind.ESCAPE]

    assert casam(low, truth["low_hp"])
    assert casam(deaths, truth["deaths"])
    # nenhum episodio do gabarito termina em morte, entao todos sao fugas
    assert len(escapes) == len(truth["low_hp"])


def test_vinheta_de_dano_sozinha_nao_vira_evento(rois, truth):
    """Regressao do bug que motivou a reescrita: a vinheta vermelha das bordas
    e *dano recebido*, nao vida baixa. Ela cobre as bordas nos mesmos instantes
    do gabarito, mas quem decide e a barra de vida."""
    detect = service_module("detector_survival")
    ev = detect.detect_survival(rois["health"], load_profile("ow2_default"))
    low = [e.t for e in ev if e.kind == EventKind.LOW_HP]
    assert len(low) == len(truth["low_hp"]), (
        "mais episodios que o gabarito indica que a vinheta esta contaminando"
    )


def test_leitura_da_barra_de_vida(rois):
    """A barra e lida pela alternancia dos tracinhos, entao um fundo claro atras
    da HUD nao pode virar 'vida cheia'."""
    import numpy as np

    from owcore.vision import iter_frames

    detect = service_module("detector_survival")
    prof = load_profile("ow2_default")
    leituras = [
        detect.read_health_fraction(f.bgr, energy_floor=2.0, tick_threshold=0.25)
        for f in iter_frames(rois["health"], prof.roi("health").fps)
    ]
    cheias = [v for v in leituras if v is not None and v > 0.7]
    baixas = [v for v in leituras if v is not None and 0.05 < v < 0.4]
    assert cheias, "nunca leu a barra cheia"
    assert baixas, "nunca leu a barra baixa"
    assert float(np.median(cheias)) > 0.85


def test_sem_template_e_sem_audio_o_detector_nao_inventa(rois, tmp_path):
    """Contrato explicito: sem os icones do jogo e com a via de audio desligada
    (o padrao), o detector devolve zero -- em vez de fingir deteccao."""
    detect = service_module("detector_ults")
    ev = detect.detect_ults(
        rois["killfeed"], rois["audio"], load_profile("ow2_default"), tmp_path / "vazio"
    )
    assert ev == []


def test_via_de_audio_quando_ligada_explicitamente(rois, truth, tmp_path):
    """Ligada, ela acha as ultimates do video sintetico -- onde a fala e o unico
    som alto. Vem desligada porque em partida real isso nao vale."""
    detect = service_module("detector_ults")
    profile = load_profile("ow2_default")
    profile.data["ults"] = {**profile.data["ults"], "audio_enabled": True,
                            "audio_spike_db": 8.0}
    try:
        ev = detect.detect_ults(
            rois["killfeed"], rois["audio"], profile, tmp_path / "vazio"
        )
        assert casam([e.t for e in ev], truth["ults"])
        assert all(e.meta["source"] == "audio" for e in ev)
    finally:
        load_profile.cache_clear()


@pytest.mark.skipif(not ULT_TEMPLATES.exists(), reason="sem templates de exemplo")
def test_templates_de_ult_acham_as_ultimates(rois, truth):
    """Com os icones o killfeed passa a valer, e ele nao depende do audio."""
    detect = service_module("detector_ults")
    ev = detect.detect_ults(
        rois["killfeed"], rois["audio"], load_profile("ow2_default"), ULT_TEMPLATES
    )
    assert casam([e.t for e in ev], truth["ults"])
    assert all(e.meta["source"] == "killfeed" for e in ev)


@pytest.mark.skipif(not MUSIC.exists(), reason="sem musica de exemplo")
def test_bpm_da_musica_de_teste(tmp_path):
    detect = service_module("beats")
    grid = detect.analyze_track(MUSIC, tmp_path).grid
    assert 110 <= grid.bpm <= 130  # a faixa foi gerada a 120 BPM
    assert len(grid.beats) > 50


@pytest.mark.skipif(not MUSIC.exists(), reason="sem musica de exemplo")
def test_estimador_proprio_acha_o_bpm_sem_librosa(tmp_path):
    """Caminho de fallback: precisa continuar montando no ritmo sem librosa."""
    from owcore.audio import read_wav

    detect = service_module("beats")
    wav = detect._decode_to_wav(MUSIC, tmp_path / "m.wav")
    # a leitura do WAV mora em `owcore.audio` desde que a partida tambem passou
    # a ter forma de onda: sao duas faixas desenhadas, e uma conta so
    data, sr = read_wav(wav)
    grid = detect._estimate_beats(data, sr, data.size / sr)
    assert 110 <= grid.bpm <= 130


def test_probe_le_o_video(truth):
    info = probe(SAMPLE)
    assert info.duration_s == pytest.approx(truth["duration_s"], abs=0.5)
    assert [info.width, info.height] == truth["size"]
    assert info.has_audio


# ── habilidades anunciadas no rodapé ───────────────────────────────────────


def avisos(rois):
    from conftest import ROOT

    detect = service_module("detector_banner")
    return detect.detect_abilities(
        rois["banner"], load_profile("ow2_default"), ROOT / "config" / "shapes"
    )


def test_dardos_da_ana_batem_com_o_gabarito(rois, truth):
    ev = [e for e in avisos(rois) if e.kind == EventKind.SLEEP]
    assert casam([e.t for e in ev], truth["sleeps"])


def test_pedradas_do_sigma_batem_com_o_gabarito(rois, truth):
    """A faixa da pedrada é desenhada em **verde**, e a do dardo em ciano: a cor
    do aviso muda de gravação para gravação, e o mesmo detector tem de achar as
    duas."""
    ev = [e for e in avisos(rois) if e.kind == EventKind.STUN]
    assert casam([e.t for e in ev], truth["stuns"])


def test_cada_habilidade_fica_com_o_seu_evento(rois, truth):
    """O risco de ter dois moldes é um disparar no aviso do outro: a mesma faixa
    viraria dois eventos."""
    ev = avisos(rois)
    assert {e.kind for e in ev} == {EventKind.SLEEP, EventKind.STUN}
    for e in ev:
        esperado = truth["sleeps"] if e.kind == EventKind.SLEEP else truth["stuns"]
        outro = truth["stuns"] if e.kind == EventKind.SLEEP else truth["sleeps"]
        assert any(abs(e.t - x) <= TOLERANCIA_S for x in esperado)
        assert not any(abs(e.t - x) <= TOLERANCIA_S for x in outro), (
            f"{e.kind} em {e.t}s caiu em cima da outra habilidade"
        )


def test_avisos_do_mesmo_rodape_nao_viram_habilidade(rois, truth):
    """O rodapé mostra vários avisos com a mesma cor, forma e posição — só o
    ícone os separa. O vídeo sintético desenha iscas justamente para provar
    que a faixa sozinha não basta."""
    detectados = [e.t for e in avisos(rois)]
    for isca in truth["avisos_isca"]:
        assert not any(abs(t - isca) <= TOLERANCIA_S for t in detectados), (
            f"o aviso-isca em {isca}s virou habilidade"
        )


def test_sem_o_template_o_detector_nao_inventa(rois, tmp_path):
    detect = service_module("detector_banner")
    ev = detect.detect_abilities(
        rois["banner"], load_profile("ow2_default"), tmp_path / "vazio"
    )
    assert ev == []


# ── acertos criticos, no mesmo recorte da mira ─────────────────────────────


def test_headshots_batem_com_o_gabarito(rois, truth):
    detect = service_module("detector_kills")
    ev = detect.detect_headshots(rois["kills"], load_profile("ow2_default"))
    assert [e.kind for e in ev] == [EventKind.HEADSHOT] * len(ev)
    assert casam([e.t for e in ev], truth["headshots"])


def test_caveira_de_eliminacao_nao_vira_headshot(rois, truth):
    """A caveira e vermelha e nasce na mesma mira que o marcador critico. O que
    separa os dois e a forma: o X deixa as quatro direcoes retas limpas, e a
    caveira preenche as oito. Sem esse segundo teste toda eliminacao viraria
    headshot tambem."""
    detect = service_module("detector_kills")
    detectados = [e.t for e in detect.detect_headshots(
        rois["kills"], load_profile("ow2_default")
    )]
    for k in truth["kills"]:
        if any(abs(k - h) <= 1.5 for h in truth["headshots"]):
            continue  # ha um headshot de verdade por perto; nada a provar aqui
        assert not any(abs(t - k) <= 0.5 for t in detectados), (
            f"a caveira em {k}s virou headshot"
        )


# ── ultimate do proprio jogador, lida no botao do rodape ───────────────────


def ults_do_jogador(rois, icones):
    detect = service_module("detector_ults")
    return detect.detect_self_ults(rois["ult"], load_profile("ow2_default"), icones)


def test_ultimate_do_jogador_bate_com_o_gabarito(rois, truth):
    ev = ults_do_jogador(rois, ABILITY_ICONS)
    assert [e.kind for e in ev] == [EventKind.ULT_USED] * len(ev)
    assert casam([e.t for e in ev], truth["self_ults"])
    assert all(e.meta["side"] == "self" for e in ev)


def test_o_evento_e_o_instante_em_que_a_ultimate_e_USADA(rois, truth):
    """O botao fica carregado por varios segundos antes -- e a borda de descida
    que marca o instante, nao a presenca do disco branco."""
    ev = ults_do_jogador(rois, ABILITY_ICONS)
    assert ev, "nenhuma ultimate detectada"
    for e in ev:
        assert e.meta["charged_s"] > 1.0, (
            "o botao carregado durou menos que a janela desenhada: o evento "
            "provavelmente saiu do lugar errado da faixa"
        )


@pytest.mark.skipif(not ABILITY_ICONS.exists(), reason="sem icones de exemplo")
def test_o_icone_do_disco_diz_qual_ultimate_era(rois):
    ev = ults_do_jogador(rois, ABILITY_ICONS)
    assert ev
    for e in ev:
        assert e.meta["hero"] == "sample"
        assert e.meta["ability"] == "self_ult"


def test_o_botao_piscando_nao_vira_ultimate(rois, truth):
    """Nem tudo que e claro, redondo e centrado naquela janela e o botao: a
    kill cam desenha um disco com o rosto de quem matou, e clarao de explosao
    passa por ali. O que os separa e o relogio -- eles duram um punhado de
    quadros, e uma ultimate fica carregada segundos antes de ser usada."""
    ev = ults_do_jogador(rois, ABILITY_ICONS)
    for piscada in truth["ult_flashes"]:
        assert not any(abs(e.t - piscada) < 1.5 for e in ev)
    assert casam([e.t for e in ev], truth["self_ults"])


def test_sem_icones_a_ultimate_continua_sendo_detectada(rois, truth, tmp_path):
    """Contrato: os icones dizem *qual* ultimate foi, e nada alem disso. Sem
    eles o evento continua saindo -- so sem rotulo."""
    ev = ults_do_jogador(rois, tmp_path / "vazio")
    assert casam([e.t for e in ev], truth["self_ults"])
    assert all("hero" not in e.meta for e in ev)


# ── eliminacao com habilidade, lida no killfeed ────────────────────────────


def mortes_por_habilidade(rois, icones, player=...):
    detect = service_module("detector_killfeed")
    return detect.detect_ability_kills(
        rois["killfeed"],
        rois["player"] if player is ... else player,
        load_profile("ow2_default"),
        icones,
    )


def test_mortes_por_habilidade_batem_com_o_gabarito(rois, truth):
    ev = mortes_por_habilidade(rois, ABILITY_ICONS)
    assert [e.kind for e in ev] == [EventKind.ABILITY_KILL] * len(ev)
    assert casam([e.t for e in ev], truth["ability_kills"])
    assert all(e.meta["ability"] == "sample/ability_kill" for e in ev)


def test_eliminacao_de_colega_de_time_nao_entra(rois, truth):
    """O killfeed anuncia as dez pessoas da partida, nao so o jogador.

    A cor da placa nao resolve: medido em gravacao real, azul e quem matou e
    vermelha e quem morreu, dos dois lados. Quem separa e o nome escrito na
    placa azul -- e no video de exemplo o colega mata com a MESMA habilidade,
    com um nome do MESMO comprimento, para nao dar para acertar por acaso.
    """
    ev = mortes_por_habilidade(rois, ABILITY_ICONS)
    for t in truth["teammate_kills"]:
        assert not any(abs(e.t - t) <= TOLERANCIA_S for e in ev), (
            f"a eliminacao do colega em {t}s entrou como se fosse do jogador"
        )


def test_sem_a_placa_do_jogador_nao_da_para_atribuir(rois):
    """Sem saber quem e o jogador nao ha eliminacao a reportar: devolver todas
    seria entregar as dos outros como se fossem dele."""
    assert mortes_por_habilidade(rois, ABILITY_ICONS, player=None) == []


def test_a_linha_do_killfeed_vale_UMA_eliminacao(rois, truth):
    """A linha fica segundos na tela. Contar quadros acima do limiar daria uma
    eliminacao a cada quadro; o que conta e ela aparecer."""
    ev = mortes_por_habilidade(rois, ABILITY_ICONS)
    assert len(ev) == len(truth["ability_kills"])


def test_sem_icones_o_killfeed_nao_inventa(rois, tmp_path):
    """Sem o banco nao da para dizer QUAL habilidade foi -- e uma eliminacao
    sem essa resposta o detector da mira ja reporta."""
    assert mortes_por_habilidade(rois, tmp_path / "vazio") == []


# ── ler o nome escrito, sem ler as letras ──────────────────────────────────


def _escrito(texto: str, escala: float, grossura: int,
             fundo: tuple[int, int, int] = (190, 140, 70)) -> "np.ndarray":
    """Uma placa da HUD com um nome escrito nela."""
    import cv2
    import numpy as np

    (largura, alt_letra), _ = cv2.getTextSize(
        texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grossura
    )
    alt = int(alt_letra * 2.6)
    img = np.full((alt, largura + 16, 3), fundo, np.uint8)
    cv2.putText(img, texto, (8, (alt + alt_letra) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                escala, (240, 240, 240), grossura, cv2.LINE_AA)
    return img


def test_o_mesmo_nome_em_tamanhos_diferentes_e_o_mesmo_nome():
    """As duas escritas da HUD nao tem o mesmo tamanho nem o mesmo
    espacamento: na gravacao de referencia o mesmo nome sai 50% mais largo no
    killfeed do que na placa do rodape, normalizado pela altura. E por isso que
    a comparacao e letra a letra, com cada letra normalizada sozinha."""
    from owcore.nameplate import read_name, same_name

    grande = read_name(_escrito("JOGADOR", 0.9, 2, (120, 62, 44)))
    pequeno = read_name(_escrito("JOGADOR", 0.5, 2))
    assert grande is not None and pequeno is not None
    assert len(grande) == len(pequeno) == 7
    assert same_name(grande, pequeno) > 0.5


def test_nome_de_outro_tamanho_e_recusado_de_saida():
    from owcore.nameplate import read_name, same_name

    assert same_name(read_name(_escrito("JOGADOR", 0.9, 2)),
                     read_name(_escrito("COLEGA", 0.9, 2))) == 0.0


def test_nome_do_mesmo_comprimento_ainda_e_outro_nome():
    """O atalho do comprimento resolve a maioria dos casos e esconde o resto:
    numa partida real havia dois nomes de nove letras. Quem separa e o desenho
    de cada letra."""
    from owcore.nameplate import read_name, same_name

    jogador = read_name(_escrito("JOGADOR", 0.9, 2, (120, 62, 44)))
    outro = read_name(_escrito("PATRICK", 0.5, 2))
    assert len(jogador) == len(outro) == 7
    assert same_name(jogador, outro) < 0.4


def test_placa_sem_escrita_nenhuma_nao_inventa_nome():
    import numpy as np
    from owcore.nameplate import read_name

    assert read_name(np.full((30, 160, 3), (190, 140, 70), np.uint8)) is None


#: uma linha de killfeed, ja acompanhada desde t=10.0. As placas que a
#: originaram: quem matou em x 100..250 e quem morreu em x 290..410.
def _tracked_line():
    kf = service_module("detector_killfeed")
    line = kf._Line(
        inner_left=250, inner_right=290, outer_left=100, outer_right=410, h=30,
        start=10.0, last_seen=10.0, key="a/b", score=0.8, style="ability",
    )
    return kf, line


def test_a_linha_entrando_nao_vira_uma_segunda_eliminacao():
    """Enquanto a linha desliza para dentro, as placas ainda estao se abrindo:
    a borda de fora anda dezenas de pixels de um quadro para o outro. Exigir
    ela ali faria a mesma eliminacao sair duas vezes."""
    kf, line = _tracked_line()
    aliado = kf.Plate(x=100, y=10, w=150, h=30)
    crescendo = kf.Plate(x=290, y=10, w=90, h=30)  # ainda nao acabou de abrir
    assert line.same_as(aliado, crescendo, 10.2, slide=0.6)
    # passada a entrada, a mesma diferenca ja nao e a mesma linha
    assert not line.same_as(aliado, crescendo, 12.0, slide=0.6)


def test_duas_eliminacoes_do_mesmo_jogador_sao_duas_linhas():
    """As bordas de dentro cercam o icone e sao iguais nas duas -- e por isso
    que as de fora, que sao o comprimento dos nomes, tem de contar."""
    kf, line = _tracked_line()
    aliado = kf.Plate(x=100, y=10, w=150, h=30)       # mesmo matador
    outra_vitima = kf.Plate(x=290, y=10, w=60, h=30)  # nome mais curto
    assert not line.same_as(aliado, outra_vitima, 12.0, slide=0.6)


def test_a_linha_e_reconhecida_mesmo_deslizando_para_baixo():
    """Quando uma eliminacao nova chega, a pilha inteira desce. A identidade da
    linha nao pode depender da altura, ou ela trocaria de linha justamente ai."""
    kf, line = _tracked_line()
    aliado = kf.Plate(x=100, y=70, w=150, h=30)
    inimigo = kf.Plate(x=290, y=70, w=120, h=30)
    assert line.same_as(aliado, inimigo, 12.0, slide=0.6)
