"""A montagem que o **usuario** faz: blocos posicionados a mao na musica.

Duas camadas, como no resto do projeto:

* a matematica da linha do tempo (`owcore.timeline`), sem ffmpeg e sem banco --
  e onde se verifica a promessa central da tela: um bloco sai exatamente no
  ponto onde foi posto, custe o que custar aos vizinhos;
* o caminho inteiro pelos microsservicos, do upload da musica ao mp4 -- e onde
  se verifica que gateway, ritmo e editor concordam sobre o formato.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import MUSIC, service_module
from owcore.db import session
from owcore.models import (
    MIN_CUT_S,
    STREAM_THUMBS,
    STREAM_RENDER,
    STREAM_RENDER_READY,
    STREAM_TRACK,
    Job,
    Timeline,
    TimelineCut,
)
from owcore.timeline import plan, snap, total_duration_s
from test_pipeline import api, drain, run_analysis

# ── a matematica, sozinha ───────────────────────────────────────────────────


def corte(at: float, dur: float, start: float = 10.0, **kw) -> TimelineCut:
    return TimelineCut(at_s=at, duration_s=dur, start_s=start, **kw)


def test_blocos_encostados_viram_so_cortes():
    pecas = plan([corte(0, 2, start=5), corte(2, 1.5, start=30)])

    assert [p.is_cut for p in pecas] == [True, True]
    assert [(p.start_s, p.end_s) for p in pecas] == [(5.0, 7.0), (30.0, 31.5)]
    assert total_duration_s(pecas) == pytest.approx(3.5)


def test_buraco_entre_dois_blocos_vira_preto():
    """O buraco NAO encurta o video.

    E a promessa da tela: o segundo bloco foi posto aos 5s da musica e tem de
    sair aos 5s do video. Emendar os blocos economizaria uma codificacao e
    moveria o corte para longe da batida onde o usuario o encaixou.
    """
    pecas = plan([corte(0, 2), corte(5, 1)])

    assert [p.black for p in pecas] == [False, True, False]
    assert pecas[1].duration_s == pytest.approx(3.0)
    assert total_duration_s(pecas) == pytest.approx(6.0)


def test_espaco_antes_do_primeiro_bloco_tambem_vira_preto():
    """Comecar o video com a musica sozinha e uma escolha legitima."""
    pecas = plan([corte(4, 2)])

    assert pecas[0].black and pecas[0].duration_s == pytest.approx(4.0)
    assert total_duration_s(pecas) == pytest.approx(6.0)


def test_preto_no_fim_nao_entra():
    """O video acaba no ultimo corte: ninguem quer 8s de tela preta no fim."""
    pecas = plan([corte(0, 2)])

    assert len(pecas) == 1 and pecas[0].is_cut


def test_corte_que_passa_do_fim_da_gravacao_e_aparado_sem_mover_os_outros():
    pecas = plan(
        [corte(0, 3, start=59), corte(4, 1, start=1)], source_duration_s=60
    )

    assert pecas[0].is_cut and pecas[0].duration_s == pytest.approx(1.0)
    # os 2s aparados viram preto, e nao um adiantamento do bloco seguinte
    assert pecas[1].black and pecas[1].duration_s == pytest.approx(3.0)
    assert total_duration_s(pecas[:2]) == pytest.approx(4.0)


def test_buraco_de_menos_de_um_quadro_nao_vira_peca():
    """Emendar 20ms custaria uma codificacao inteira para ninguem ver nada."""
    pecas = plan([corte(0, 2), corte(2.02, 1)])

    assert [p.is_cut for p in pecas] == [True, True]


def test_pretos_seguidos_viram_um_so():
    """Cada peca custa uma codificacao; duas de preto seguidas sao desperdicio."""
    pecas = plan(
        [corte(0, 3, start=59), corte(5, 1, start=1)], source_duration_s=60
    )

    assert [p.black for p in pecas] == [False, True, False]


def test_ima_gruda_na_batida_perto_e_ignora_a_longe():
    batidas = [0.0, 0.5, 1.0, 1.5]

    assert snap(0.52, batidas) == 0.5
    assert snap(0.75, batidas) == 0.75  # equidistante das duas, longe demais


def test_linha_do_tempo_ordena_e_recusa_sobreposicao():
    spec = Timeline(cuts=[corte(4, 1), corte(0, 2)])
    assert [c.at_s for c in spec.cuts] == [0.0, 4.0]
    assert spec.duration_s == pytest.approx(5.0)

    with pytest.raises(ValueError, match="sobrep"):
        Timeline(cuts=[corte(0, 2), corte(1, 1)])

    with pytest.raises(ValueError):
        Timeline(cuts=[corte(0, MIN_CUT_S / 2)])


# ── o caminho inteiro ───────────────────────────────────────────────────────


def subir_musica(job_id: str, music: Path = MUSIC) -> str:
    """Manda a musica e roda o worker que a ouve, como acontece em producao."""
    resp = api().post(
        f"/api/jobs/{job_id}/tracks",
        files={"audio": ("music.wav", music.read_bytes(), "audio/wav")},
    )
    assert resp.status_code == 201, resp.text
    track_id = resp.json()["id"]
    assert resp.json()["status"] == "pending"

    analisador = service_module("beats", "main").TrackAnalyzer()
    for payload in drain(STREAM_TRACK, "tracks"):
        analisador.handle(payload)
    return track_id


def montar(job_id: str, timelines: list[dict]) -> str:
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps(timelines)},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def run_render() -> None:
    beats = service_module("beats", "main").BeatsService()
    for payload in drain(STREAM_RENDER, "beats"):
        beats.handle(payload)
    editor = service_module("editor", "main").Editor()
    for payload in drain(STREAM_RENDER_READY, "editor"):
        editor.handle(payload)


@pytest.mark.skipif(not MUSIC.exists(), reason="precisa do data/sample/music.wav")
def test_musica_sobe_antes_de_existir_video_e_volta_pronta_para_desenhar(
    isolated, short_sample
):
    """O app precisa da musica *analisada* para desenhar a tela de montagem."""
    job_id = run_analysis(short_sample)
    track_id = subir_musica(job_id)

    track = api().get(f"/api/tracks/{track_id}").json()
    assert track["status"] == "ready", track["error"]
    assert track["duration_s"] > 5
    assert track["bpm"] > 0
    assert len(track["beats"]) > 4, "sem batidas nao da para grudar corte nenhum"
    assert len(track["peaks"]) > 100, "sem forma de onda nao da para achar o refrao"
    assert all(0.0 <= v <= 1.0 for v in track["peaks"])
    assert track["audio_url"].endswith(f"/api/tracks/{track_id}/audio")

    # e ela aparece no job, para o app nao ter de guardar id nenhum
    detail = api().get(f"/api/jobs/{job_id}").json()
    assert [t["id"] for t in detail["tracks"]] == [track_id]


@pytest.mark.skipif(not MUSIC.exists(), reason="precisa do data/sample/music.wav")
def test_o_audio_da_musica_e_servido_com_range_para_o_player(isolated, short_sample):
    """Sem Range o player nao consegue pular para o refrao."""
    job_id = run_analysis(short_sample)
    track_id = subir_musica(job_id)

    resp = api().get(f"/api/tracks/{track_id}/audio", headers={"range": "bytes=0-99"})
    assert resp.status_code == 206
    assert len(resp.content) == 100
    assert resp.headers["content-range"].startswith("bytes 0-99/")


def test_a_gravacao_e_servida_com_range_para_o_preview(isolated, short_sample):
    """O monitor da tela de montagem busca dentro da gravacao original.

    Sem `Range` ele teria de baixar a partida inteira para mostrar um quadro
    dos 3 minutos -- e renderizar de verdade a cada ajuste custaria uma volta
    pelo ffmpeg por arrasto.
    """
    job_id = run_analysis(short_sample)

    detail = api().get(f"/api/jobs/{job_id}").json()
    assert detail["video_url"] == f"/api/jobs/{job_id}/video"

    resp = api().get(f"/api/jobs/{job_id}/video", headers={"range": "bytes=0-511"})
    assert resp.status_code == 206
    assert len(resp.content) == 512
    assert resp.headers["content-type"] == "video/mp4"
    assert resp.headers["accept-ranges"] == "bytes"

    inteiro = api().get(f"/api/jobs/{job_id}/video")
    assert inteiro.status_code == 200
    assert int(inteiro.headers["content-length"]) == short_sample.stat().st_size


def test_preview_de_job_inexistente_e_404(isolated):
    assert api().get("/api/jobs/naoexiste/video").status_code == 404


# ── miniaturas dos momentos ─────────────────────────────────────────────────


def rodar_thumbs() -> int:
    worker = service_module("thumbs", "main").Thumbs()
    quantos = 0
    for payload in drain(STREAM_THUMBS, "thumbs"):
        worker.handle(payload)
        quantos += 1
    return quantos


def test_a_analise_ja_pede_as_miniaturas_dos_momentos(isolated, short_sample):
    """A barra lateral do editor precisa de imagem para escolher entre trinta
    eliminacoes; sem ela, sao trinta relogios iguais."""
    job_id = run_analysis(short_sample)
    assert rodar_thumbs() >= 1, "o planejador nao pediu as miniaturas"

    detail = api().get(f"/api/jobs/{job_id}").json()
    momentos = [
        e["t"] for e in detail["events"]
        if e["kind"] in {"kill", "sleep", "stun", "ult_negated", "escape"}
    ]
    assert momentos, "a analise nao achou momento nenhum"

    resp = api().get(f"/api/jobs/{job_id}/frame", params={"t": momentos[0]})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 500, "a miniatura saiu vazia"
    # o quadro de um instante nunca muda
    assert "max-age" in resp.headers.get("cache-control", "")


def test_momento_sem_miniatura_responde_404_em_vez_de_quebrar(
    isolated, short_sample
):
    """404 aqui quer dizer 'ainda nao extraida' -- o app mostra o lugar dela."""
    job_id = run_analysis(short_sample)
    assert api().get(f"/api/jobs/{job_id}/frame", params={"t": 999}).status_code == 404


def test_pedir_de_novo_nao_reextrai_o_que_ja_existe(isolated, short_sample):
    """O app pede ao abrir o editor; o servico tem de pular o que ja esta la."""
    job_id = run_analysis(short_sample)
    rodar_thumbs()

    resp = api().post(f"/api/jobs/{job_id}/frames")
    assert resp.status_code == 202

    worker = service_module("thumbs", "main").Thumbs()
    # o segundo pedido nao acha nada a extrair, e diz isso sem estourar
    for payload in drain(STREAM_THUMBS, "thumbs"):
        worker.handle(payload)

    detail = api().get(f"/api/jobs/{job_id}").json()
    t = next(e["t"] for e in detail["events"] if e["kind"] == "kill")
    assert api().get(f"/api/jobs/{job_id}/frame", params={"t": t}).status_code == 200


def test_pedir_miniatura_de_job_inexistente_e_404(isolated):
    assert api().post("/api/jobs/naoexiste/frames").status_code == 404


@pytest.mark.skipif(not MUSIC.exists(), reason="precisa do data/sample/music.wav")
def test_montagem_manual_vira_video_com_os_blocos_onde_o_usuario_pos(
    isolated, short_sample
):
    job_id = run_analysis(short_sample)
    track_id = subir_musica(job_id)

    detail = api().get(f"/api/jobs/{job_id}").json()
    momentos = [e["t"] for e in detail["events"] if e["kind"] == "kill"][:2]
    assert momentos, "a analise nao achou eliminacao nenhuma para montar"

    # dois blocos com um buraco de 1s entre eles: o video tem de durar
    # 1.5 + 1.0 + 1.5 = 4s, e nao os 3s dos cortes
    cuts = [
        {"source_t": momentos[0], "start_s": max(0.0, momentos[0] - 1.0),
         "duration_s": 1.5, "at_s": 0.0, "kind": "kill"},
        {"source_t": momentos[-1], "start_s": max(0.0, momentos[-1] - 1.0),
         "duration_s": 1.5, "at_s": 2.5, "kind": "kill"},
    ]
    render_id = montar(
        job_id,
        [{"title": "Minha montagem", "track_id": track_id,
          "music_start_s": 2.0, "cuts": cuts}],
    )
    run_render()

    pedido = api().get(f"/api/renders/{render_id}").json()
    assert pedido["status"] == "done", pedido["error"]
    assert len(pedido["clips"]) == 1
    clip = pedido["clips"][0]

    assert clip["kind"] == "custom"
    assert clip["title"] == "Minha montagem"
    assert clip["meta"]["hand_made"] is True
    assert clip["meta"]["segments"] == 2
    assert clip["meta"]["blackfill_s"] == pytest.approx(1.0, abs=0.05)
    assert clip["meta"]["music_start_s"] == pytest.approx(2.0)
    assert clip["video_url"], "o video nao saiu"
    assert clip["segments_zip_url"], "os cortes avulsos nao sairam"

    # o arquivo mesmo: o buraco esta la, e a musica tocando por cima dele
    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    local = local_copy(key, Path(isolated.work_dir) / "conferencia")
    info = ffmpeg.probe(local)
    assert info.duration_s == pytest.approx(4.0, abs=0.35)
    assert info.has_audio


def test_sem_musica_a_montagem_manual_fica_com_o_audio_da_partida(
    isolated, short_sample
):
    """E o preto do buraco tem de sair com silencio *compativel*.

    Sem trilha os cortes mantem o audio da partida, e o `concat` recusa juntar
    pedacos cujo audio nao bate -- um trecho com som e um preto mudo nao
    concatenam. O preto sai com silencio na mesma taxa, e e isto que este teste
    guarda: um buraco no meio de uma montagem sem musica.
    """
    job_id = run_analysis(short_sample)
    render_id = montar(
        job_id,
        [{"cuts": [
            {"start_s": 1.0, "duration_s": 1.5, "at_s": 0.0},
            {"start_s": 6.0, "duration_s": 1.5, "at_s": 3.0},
        ]}],
    )
    run_render()

    clip = api().get(f"/api/renders/{render_id}").json()["clips"][0]
    assert clip["meta"]["original_audio"] is True
    assert clip["meta"]["music_name"] is None
    assert clip["meta"]["blackfill_s"] == pytest.approx(1.5, abs=0.05)
    assert clip["video_url"], "a montagem com buraco e sem musica nao saiu"

    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    info = ffmpeg.probe(local_copy(key, Path(isolated.work_dir) / "sem_musica"))
    assert info.duration_s == pytest.approx(4.5, abs=0.35)
    assert info.has_audio, "o audio da partida se perdeu na emenda com o preto"


def test_pedido_vazio_e_recusado(isolated, short_sample):
    job_id = run_analysis(short_sample)
    resp = api().post(f"/api/jobs/{job_id}/renders", data={"timelines": "[]"})
    assert resp.status_code == 422
    assert "linha do tempo" in resp.json()["detail"]


def test_musica_de_outro_job_e_recusada(isolated, short_sample):
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([
            {"track_id": "naoexiste",
             "cuts": [{"start_s": 1, "duration_s": 1, "at_s": 0}]}
        ])},
    )
    assert resp.status_code == 422
    assert "musica desconhecida" in resp.json()["detail"]


@pytest.mark.skipif(not MUSIC.exists(), reason="precisa do data/sample/music.wav")
def test_musica_ainda_nao_analisada_barra_o_pedido(isolated, short_sample):
    """Sem batidas nem duracao a tela nao teria como ter posicionado nada --
    um pedido assim so pode ser engano do app."""
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/tracks",
        files={"audio": ("music.wav", MUSIC.read_bytes(), "audio/wav")},
    )
    track_id = resp.json()["id"]  # de proposito: sem rodar o analisador

    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([
            {"track_id": track_id,
             "cuts": [{"start_s": 1, "duration_s": 1, "at_s": 0}]}
        ])},
    )
    assert resp.status_code == 409
    assert "ainda nao foi analisada" in resp.json()["detail"]


@pytest.mark.skipif(not MUSIC.exists(), reason="precisa do data/sample/music.wav")
def test_musica_ilegivel_falha_sozinha_sem_derrubar_o_job(isolated, short_sample):
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/tracks",
        files={"audio": ("quebrada.mp3", b"isto nao e audio", "audio/mpeg")},
    )
    track_id = resp.json()["id"]

    analisador = service_module("beats", "main").TrackAnalyzer()
    for payload in drain(STREAM_TRACK, "tracks"):
        try:
            analisador.handle(payload)
        except Exception as exc:  # o worker real transforma isto em on_error
            analisador.on_error(payload, exc)

    track = api().get(f"/api/tracks/{track_id}").json()
    assert track["status"] == "failed"
    assert track["error"]
    # e a analise da partida continua valendo
    assert api().get(f"/api/jobs/{job_id}").json()["status"] == "ready"


# ── o rascunho: a montagem sobrevive a um F5 ────────────────────────────────


def test_a_montagem_em_andamento_volta_junto_com_o_job(isolated, short_sample):
    """Recarregar a pagina custava a montagem inteira; agora ela e do job."""
    job_id = run_analysis(short_sample)
    assert api().get(f"/api/jobs/{job_id}").json()["draft"] == {}

    rascunho = {
        "title": "Em construcao",
        "music_start_s": 12.5,
        "cuts": [
            {"source_t": 30.0, "start_s": 29.0, "duration_s": 1.5, "at_s": 0.0,
             "kind": "kill"},
            {"source_t": 75.0, "start_s": 74.0, "duration_s": 1.0, "at_s": 2.0,
             "kind": "sleep"},
        ],
    }
    resp = api().put(f"/api/jobs/{job_id}/draft", json=rascunho)
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_cuts"] == 2

    volta = api().get(f"/api/jobs/{job_id}").json()["draft"]
    assert volta["title"] == "Em construcao"
    assert volta["music_start_s"] == 12.5
    assert [c["at_s"] for c in volta["cuts"]] == [0.0, 2.0]
    assert volta["cuts"][0]["kind"] == "kill"


def test_o_rascunho_lembra_as_correcoes_da_grade_de_batidas(
    isolated, short_sample
):
    """Elas nao mudam o video -- o corte guarda instantes absolutos --, mas
    mudam onde o ima gruda. Consertar a grade duas vezes irrita."""
    job_id = run_analysis(short_sample)
    api().put(
        f"/api/jobs/{job_id}/draft",
        json={"cuts": [], "beat_offset_s": 0.12, "beat_multiplier": 2.0,
              "beat_bar": 4},
    )

    draft = api().get(f"/api/jobs/{job_id}").json()["draft"]
    assert draft["beat_offset_s"] == pytest.approx(0.12)
    assert draft["beat_multiplier"] == pytest.approx(2.0)
    assert draft["beat_bar"] == 4


def test_rascunho_sem_corte_nenhum_e_valido(isolated, short_sample):
    """Um rascunho existe desde antes de o primeiro bloco entrar -- salvar so a
    musica escolhida tem de funcionar."""
    job_id = run_analysis(short_sample)
    resp = api().put(
        f"/api/jobs/{job_id}/draft",
        json={"title": "so a musica por enquanto", "cuts": []},
    )
    assert resp.status_code == 200
    assert resp.json()["n_cuts"] == 0


def test_rascunho_com_corte_impossivel_e_recusado(isolated, short_sample):
    """Guardar lixo agora seria devolver lixo na proxima abertura."""
    job_id = run_analysis(short_sample)
    resp = api().put(
        f"/api/jobs/{job_id}/draft",
        json={"cuts": [{"start_s": -5, "duration_s": 1, "at_s": 0}]},
    )
    assert resp.status_code == 422
    assert "rascunho invalido" in resp.json()["detail"]


def test_salvar_de_novo_substitui_o_anterior(isolated, short_sample):
    job_id = run_analysis(short_sample)
    for n in (1, 2, 3):
        api().put(
            f"/api/jobs/{job_id}/draft",
            json={"cuts": [
                {"start_s": 1, "duration_s": 1, "at_s": float(i)}
                for i in range(n)
            ]},
        )
    assert len(api().get(f"/api/jobs/{job_id}").json()["draft"]["cuts"]) == 3


def test_descartar_o_rascunho(isolated, short_sample):
    job_id = run_analysis(short_sample)
    api().put(f"/api/jobs/{job_id}/draft",
              json={"cuts": [{"start_s": 1, "duration_s": 1, "at_s": 0}]})

    assert api().delete(f"/api/jobs/{job_id}/draft").status_code == 204
    assert api().get(f"/api/jobs/{job_id}").json()["draft"] == {}


def test_rascunho_de_job_inexistente_e_404(isolated):
    assert api().put("/api/jobs/naoexiste/draft", json={"cuts": []}).status_code == 404
    assert api().delete("/api/jobs/naoexiste/draft").status_code == 404


def test_gerar_nao_apaga_o_rascunho(isolated, short_sample):
    """Depois de gerar, o normal e querer ajustar e gerar de novo -- perder a
    montagem nesse ponto seria o mesmo estrago do F5."""
    job_id = run_analysis(short_sample)
    cuts = [{"source_t": 3.0, "start_s": 1.0, "duration_s": 1.5, "at_s": 0.0}]
    api().put(f"/api/jobs/{job_id}/draft", json={"title": "v1", "cuts": cuts})

    montar(job_id, [{"title": "v1", "cuts": cuts}])
    run_render()

    assert api().get(f"/api/jobs/{job_id}").json()["draft"]["title"] == "v1"


# ── o proxy e a onda da partida (Fase 2) ────────────────────────────────────


def test_o_proxy_sai_da_mesma_decodificacao_e_e_muito_menor(
    isolated, short_sample
):
    """A cópia reduzida existe para o monitor do editor.

    Buscar dentro da gravação original dezenas de vezes por segundo derrubava o
    elemento de vídeo do navegador. O proxy sai como mais uma saída da
    decodificação que já acontece, então o custo é perto de zero -- e é isso
    que este teste guarda junto com o tamanho.
    """
    from owcore import ffmpeg
    from owcore.models import Job
    from owcore.storage import get_storage, local_copy

    job_id = run_analysis(short_sample)

    with session() as s:
        job = s.get(Job, job_id)
        assert job.proxy_key, "o preprocessador nao gerou o proxy"
        key = job.proxy_key
    storage = get_storage()
    assert storage.exists(key)

    proxy = local_copy(key, Path(isolated.work_dir) / "conferencia")
    info = ffmpeg.probe(proxy)
    original = ffmpeg.probe(short_sample)

    # mesma partida: a duracao tem de bater
    assert info.duration_s == pytest.approx(original.duration_s, abs=0.5)
    # e a tela inteira, nao um recorte
    assert info.width / info.height == pytest.approx(
        original.width / original.height, abs=0.02
    )
    assert info.width <= 640
    assert proxy.stat().st_size < short_sample.stat().st_size


def test_o_proxy_e_servido_com_range(isolated, short_sample):
    job_id = run_analysis(short_sample)

    detail = api().get(f"/api/jobs/{job_id}").json()
    assert detail["proxy_url"] == f"/api/jobs/{job_id}/proxy"

    resp = api().get(f"/api/jobs/{job_id}/proxy", headers={"range": "bytes=0-511"})
    assert resp.status_code == 206
    assert len(resp.content) == 512
    assert resp.headers["content-type"] == "video/mp4"


def test_partida_analisada_antes_do_proxy_diz_isso_em_vez_de_quebrar(isolated):
    """O app cai na gravacao original quando `proxy_url` vem nulo."""
    from owcore.models import Job

    with session() as s:
        s.add(Job(id="antigo0000000001", video_key="k", video_name="v.mp4"))

    assert api().get("/api/jobs/antigo0000000001").json()["proxy_url"] is None
    assert api().get("/api/jobs/antigo0000000001/proxy").status_code == 404


def test_a_onda_da_partida_volta_com_o_job(isolated, short_sample):
    """É ela que mostra o tiro e a explosão na régua do editor."""
    job_id = run_analysis(short_sample)

    detail = api().get(f"/api/jobs/{job_id}").json()
    onda = detail["waveform"]

    assert len(onda) > 100, "sem onda nao da para casar o corte com o som"
    assert all(0.0 <= v <= 1.0 for v in onda)
    assert max(onda) == pytest.approx(1.0), "a onda e normalizada pelo pico"


def test_a_onda_nao_vai_na_listagem(isolated, short_sample):
    """São alguns milhares de números por partida, e a lista não os usa."""
    run_analysis(short_sample)
    jobs = api().get("/api/jobs").json()["jobs"]

    assert jobs, "nenhuma partida na listagem"
    assert "waveform" not in jobs[0]
    # o proxy, esse sim, vai: a lista e por onde o app decide o que abrir
    assert "proxy_url" in jobs[0]
