"""Precisao dos detectores contra o gabarito do video sintetico.

E o teste que sustenta a tese do projeto: recortar uma faixa minuscula da tela,
em FPS baixo, basta para recuperar os eventos da partida.
"""

from __future__ import annotations

import json

import pytest

from conftest import MUSIC, SAMPLE, TRUTH, ULT_TEMPLATES, needs_sample, service_module
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
        SAMPLE, prof.rois(["kills", "health", "killfeed", "banner"]), out
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
    grid = detect.analyze_music(MUSIC, tmp_path)
    assert 110 <= grid.bpm <= 130  # a faixa foi gerada a 120 BPM
    assert len(grid.beats) > 50


@pytest.mark.skipif(not MUSIC.exists(), reason="sem musica de exemplo")
def test_estimador_proprio_acha_o_bpm_sem_librosa(tmp_path):
    """Caminho de fallback: precisa continuar montando no ritmo sem librosa."""
    detect = service_module("beats")
    wav = detect._decode_to_wav(MUSIC, tmp_path / "m.wav")
    data, sr = detect._read_wav(wav)
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
