"""Regras de highlight -- a parte do sistema que decide o que e um 'melhor
momento'. Funcao pura, entao da para cobrir bem sem tocar em video."""

from __future__ import annotations

from owcore.models import (
    BeatGrid,
    ClipOptions,
    DetectionEvent,
    EventKind,
    HighlightKind,
    JobParams,
)
from owcore.rules import (
    build_highlights,
    derive_negated_ults,
    fit_to_window,
    montage_segments,
)


def kills(*ts: float) -> list[DetectionEvent]:
    return [DetectionEvent(kind=EventKind.KILL, t=t) for t in ts]


def test_tres_kills_juntas_viram_multikill():
    hs = build_highlights(kills(10.0, 11.0, 12.5), JobParams(), 60.0)
    rajada = next(h for h in hs if h.kind == HighlightKind.MULTIKILL)
    assert rajada.meta["kills"] == 3
    # a rajada vem primeiro: pontua mais que a montagem das mesmas eliminações
    assert hs[0].kind == HighlightKind.MULTIKILL


def test_kills_espalhadas_nao_viram_multikill():
    hs = build_highlights(kills(5.0, 25.0, 45.0), JobParams(), 60.0)
    assert [h.kind for h in hs] == [HighlightKind.BEAT_MONTAGE]


def test_quatro_kills_sem_morrer_viram_solo_wipe():
    hs = build_highlights(kills(10.0, 11.0, 12.0, 13.0), JobParams(), 60.0)
    assert hs[0].kind == HighlightKind.SOLO_WIPE
    assert hs[0].meta["solo"] is True


def test_morrer_no_meio_rebaixa_para_multikill():
    ev = kills(10.0, 11.0, 12.0, 13.0) + [
        DetectionEvent(kind=EventKind.DEATH, t=11.5)
    ]
    hs = build_highlights(ev, JobParams(), 60.0)
    assert hs[0].kind == HighlightKind.MULTIKILL
    assert hs[0].meta["solo"] is False


def test_pre_e_pos_roll_nao_saem_do_video():
    hs = build_highlights(kills(1.0, 2.0, 3.0), JobParams(), 4.0)
    assert hs[0].start == 0.0
    assert hs[0].end == 4.0


def test_fugas_seguidas_viram_highlight_de_escape():
    ev = [DetectionEvent(kind=EventKind.ESCAPE, t=t) for t in (10.0, 18.0)]
    hs = build_highlights(ev, JobParams(), 60.0)
    assert [h.kind for h in hs] == [HighlightKind.ESCAPE]


def test_fuga_que_termina_em_morte_nao_conta():
    ev = [DetectionEvent(kind=EventKind.ESCAPE, t=t) for t in (10.0, 18.0)]
    ev.append(DetectionEvent(kind=EventKind.DEATH, t=14.0))
    hs = build_highlights(ev, JobParams(), 60.0)
    assert HighlightKind.ESCAPE not in [h.kind for h in hs]


def test_kill_usada_numa_rajada_tambem_entra_na_montagem():
    """Um trecho aproveitado num vídeo continua disponível para os outros: cada
    vídeo é uma montagem independente, não uma partilha do material."""
    ev = kills(10.0, 11.0, 12.0, 40.0)
    hs = build_highlights(ev, JobParams(), 60.0)
    assert HighlightKind.MULTIKILL in [h.kind for h in hs]
    montagem = next(h for h in hs if h.kind == HighlightKind.BEAT_MONTAGE)
    assert montagem.beats_at == [10.0, 11.0, 12.0, 40.0]


def test_montagem_pode_ser_desligada():
    hs = build_highlights(kills(5.0, 25.0), JobParams(make_beat_montage=False), 60.0)
    assert hs == []


# ── ultimates anuladas: correlacao entre dois detectores ────────────────────


def test_ult_seguida_de_kill_vira_ult_anulada():
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)] + kills(41.5)
    out = derive_negated_ults(ev, 6.0)
    assert len(out) == 1
    assert out[0].t == 41.5
    assert out[0].meta["delay_s"] == 1.5


def test_ult_sem_kill_na_janela_nao_conta():
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)] + kills(50.0)
    assert derive_negated_ults(ev, 6.0) == []


def test_kill_antes_da_ult_nao_conta():
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)] + kills(39.0)
    assert derive_negated_ults(ev, 6.0) == []


def test_build_highlights_deriva_a_montagem_de_ults():
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=t) for t in (10.0, 30.0)]
    ev += kills(11.0, 31.0)
    hs = build_highlights(ev, JobParams(), 60.0)
    assert HighlightKind.ULT_MONTAGE in [h.kind for h in hs]


# ── corte no ritmo ─────────────────────────────────────────────────────────


def test_segmentos_da_montagem_duram_multiplos_da_batida():
    grid = BeatGrid(bpm=120.0, beats=[i * 0.5 for i in range(40)])
    segs = montage_segments([5.0, 15.0, 25.0], grid, 2, 60.0)
    assert len(segs) == 3
    for start, end in segs:
        assert abs((end - start) - 1.0) < 1e-6  # 2 batidas de 0.5s


def test_sem_musica_a_montagem_usa_duracao_fixa():
    segs = montage_segments([5.0, 15.0], None, 2, 60.0)
    assert len(segs) == 2
    assert all(end > start for start, end in segs)


def test_momentos_proximos_viram_um_segmento_so():
    grid = BeatGrid(bpm=120.0, beats=[i * 0.5 for i in range(40)])
    segs = montage_segments([10.0, 10.3], grid, 2, 60.0)
    assert len(segs) == 1  # nao repete a mesma imagem duas vezes


def test_segmento_nao_passa_do_fim_do_video():
    grid = BeatGrid(bpm=120.0, beats=[i * 0.5 for i in range(40)])
    segs = montage_segments([9.9], grid, 2, 10.0)
    assert segs[-1][1] <= 10.0


# ── janela de música: quanto a montagem deve durar ─────────────────────────

SEGS = [(0.0, 1.0), (5.0, 6.0), (10.0, 11.0)]  # 3 trechos de 1s


def total(segments) -> float:
    return round(sum(e - s for s, e in segments), 3)


def test_sem_janela_a_montagem_fica_do_tamanho_natural():
    assert fit_to_window(SEGS, None, loop=False) == SEGS
    assert fit_to_window(SEGS, None, loop=True) == SEGS


def test_com_loop_a_montagem_tem_exatamente_a_duracao_pedida():
    out = fit_to_window(SEGS, 5.0, loop=True)
    assert total(out) == 5.0
    assert len(out) == 5  # repetiu dois trechos


def test_loop_apara_o_ultimo_trecho_para_fechar_a_duracao():
    out = fit_to_window(SEGS, 2.5, loop=True)
    assert total(out) == 2.5
    # os dois primeiros entram inteiros e o terceiro entra pela metade; qual
    # deles é qual depende do sorteio, então o que se afirma é o formato
    assert len(out) == 3
    assert round(out[-1][1] - out[-1][0], 3) == 0.5


def test_sem_loop_nunca_passa_da_duracao_pedida():
    out = fit_to_window(SEGS, 2.5, loop=False)
    assert total(out) == 2.0
    assert total(out) <= 2.5


def test_sem_loop_entrega_tudo_que_couber():
    out = fit_to_window(SEGS, 10.0, loop=False)
    assert out == SEGS  # nao repete mesmo sobrando janela


def test_janela_curta_demais_nao_devolve_trecho_nenhum():
    assert fit_to_window(SEGS, 0.05, loop=True) == []


# ── ordem sorteada ao repetir ──────────────────────────────────────────────

QUATRO = [(0.0, 1.0), (5.0, 6.0), (10.0, 11.0), (20.0, 21.0)]


def test_ao_repetir_a_ordem_e_sorteada():
    import random

    a = fit_to_window(QUATRO, 8.0, loop=True, rng=random.Random(1))
    b = fit_to_window(QUATRO, 8.0, loop=True, rng=random.Random(2))
    assert [s for s, _ in a] != [s for s, _ in b]
    assert total(a) == total(b) == 8.0


def test_a_primeira_passada_usa_cada_trecho_uma_vez():
    """Sortear nao pode fazer um trecho reaparecer antes de todos entrarem."""
    import random

    out = fit_to_window(QUATRO, 4.0, loop=True, rng=random.Random(3))
    assert len(out) == 4
    assert len({s for s, _ in out}) == 4


def test_sem_repetir_a_ordem_cronologica_e_mantida():
    import random

    out = fit_to_window(QUATRO, 10.0, loop=False, rng=random.Random(4))
    assert out == QUATRO


def test_sortear_e_reprodutivel_com_a_mesma_semente():
    import random

    a = fit_to_window(QUATRO, 8.0, loop=True, rng=random.Random(7))
    b = fit_to_window(QUATRO, 8.0, loop=True, rng=random.Random(7))
    assert a == b


def test_janela_de_musica_derivada_das_opcoes_do_video():
    """A janela e por video, nao por job: e assim que dois videos da mesma
    partida podem ter musicas e duracoes diferentes."""
    assert ClipOptions(music_start_s=10, music_end_s=40).music_window_s == 30.0
    assert ClipOptions().music_window_s is None


def test_fim_antes_do_inicio_e_rejeitado():
    import pytest

    with pytest.raises(ValueError):
        ClipOptions(music_start_s=30, music_end_s=10)


def test_sleeps_viram_montagem_propria():
    ev = [DetectionEvent(kind=EventKind.SLEEP, t=t) for t in (10.0, 30.0, 50.0)]
    hs = build_highlights(ev, JobParams(), 60.0)
    montagem = next(h for h in hs if h.kind == HighlightKind.SLEEP_MONTAGE)
    assert montagem.beats_at == [10.0, 30.0, 50.0]


def test_pedradas_viram_montagem_propria():
    ev = [DetectionEvent(kind=EventKind.STUN, t=t) for t in (5.0, 20.0)]
    hs = build_highlights(ev, JobParams(), 60.0)
    montagem = next(h for h in hs if h.kind == HighlightKind.STUN_MONTAGE)
    assert montagem.beats_at == [5.0, 20.0]


def test_dardo_e_pedrada_nao_se_misturam():
    """Habilidades diferentes, montagens diferentes -- mesmo saindo do mesmo
    detector."""
    ev = [DetectionEvent(kind=EventKind.SLEEP, t=t) for t in (10.0, 30.0)] + [
        DetectionEvent(kind=EventKind.STUN, t=t) for t in (15.0, 35.0)
    ]
    hs = build_highlights(ev, JobParams(), 60.0)
    dardos = next(h for h in hs if h.kind == HighlightKind.SLEEP_MONTAGE)
    pedradas = next(h for h in hs if h.kind == HighlightKind.STUN_MONTAGE)
    assert dardos.beats_at == [10.0, 30.0]
    assert pedradas.beats_at == [15.0, 35.0]


def test_sleep_e_kill_geram_montagens_separadas():
    ev = kills(5.0, 25.0) + [
        DetectionEvent(kind=EventKind.SLEEP, t=t) for t in (12.0, 40.0)
    ]
    tipos = [h.kind for h in build_highlights(ev, JobParams(), 60.0)]
    assert HighlightKind.BEAT_MONTAGE in tipos
    assert HighlightKind.SLEEP_MONTAGE in tipos
