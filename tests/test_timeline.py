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
import subprocess
from pathlib import Path

import pytest

from conftest import MUSIC, service_module
from owcore.db import session
from owcore.models import (
    MIN_CUT_S,
    STREAM_THUMBS,
    STREAM_RENDER,
    STREAM_RENDER_READY,
    STREAM_MEDIA,
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

    analisador = service_module("beats", "main").MediaAnalyzer()
    for payload in drain(STREAM_MEDIA, "media"):
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
    # a URL canonica agora e a da biblioteca; a musica e um item dela
    assert track["audio_url"].endswith(f"/api/media/{track_id}/file")
    assert track["kind"] == "audio"
    # e a rota antiga continua respondendo, porque o app ainda a usa
    assert api().get(f"/api/tracks/{track_id}/audio",
                     headers={"range": "bytes=0-31"}).status_code == 206

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

    analisador = service_module("beats", "main").MediaAnalyzer()
    for payload in drain(STREAM_MEDIA, "media"):
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


# ── montagem em camadas (Fase 3) ────────────────────────────────────────────


def test_o_formato_da_v1_continua_entrando_e_sai_em_camadas(isolated):
    """Nenhuma migracao roda no banco: o formato velho e entrada valida.

    Um rascunho salvo antes desta versao, ou um pedido guardado num render
    antigo, chega com `cuts` e e convertido na leitura -- uma camada so, de
    clipes de gravacao.
    """
    from owcore.models import ClipSource, Timeline

    velha = Timeline(
        cuts=[
            {"start_s": 10, "duration_s": 2, "at_s": 0, "kind": "kill"},
            {"start_s": 30, "duration_s": 1, "at_s": 3},
        ]
    )

    assert len(velha.layers) == 1
    assert [c.at_s for c in velha.clips] == [0.0, 3.0]
    assert velha.clips[0].source is ClipSource.RECORDING
    assert velha.clips[0].kind == "kill"
    assert velha.duration_s == pytest.approx(4.0)
    # e continua sabendo se apresentar como V1, para o caminho antigo
    assert [c.duration_s for c in velha.cuts] == [2.0, 1.0]
    assert velha.de_uma_camada_so


def test_camada_ou_transformacao_tira_a_montagem_do_caminho_antigo(isolated):
    """A escolha do caminho e o que protege o render.

    Corte-e-emenda e mais resistente -- um corte ruim custa so ele --, entao ele
    fica com o caso comum. O grafo entra so quando e preciso.
    """
    from owcore.models import Layer, Timeline, TimelineClip

    simples = Timeline(layers=[Layer(clips=[TimelineClip(at_s=0, duration_s=1)])])
    assert simples.de_uma_camada_so

    duas = Timeline(
        layers=[
            Layer(clips=[TimelineClip(at_s=0, duration_s=1)]),
            Layer(clips=[TimelineClip(at_s=0, duration_s=1)]),
        ]
    )
    assert not duas.de_uma_camada_so

    com_zoom = Timeline(
        layers=[
            Layer(clips=[
                TimelineClip(at_s=0, duration_s=1, transform={"scale": 1.5})
            ])
        ]
    )
    assert not com_zoom.de_uma_camada_so

    # camada escondida nao conta: sobra uma so, e ela e simples
    com_escondida = Timeline(
        layers=[
            Layer(clips=[TimelineClip(at_s=0, duration_s=1)]),
            Layer(hidden=True, clips=[TimelineClip(at_s=0, duration_s=1)]),
        ]
    )
    assert com_escondida.de_uma_camada_so


def test_duas_camadas_viram_um_video_com_a_de_cima_por_cima(
    isolated, short_sample
):
    """O caminho novo, do pedido ao mp4."""
    job_id = run_analysis(short_sample)

    camadas = [
        {"clips": [
            {"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0, "kind": "kill"},
            {"at_s": 3.0, "duration_s": 1.5, "start_s": 6.0},
        ]},
        {"name": "canto", "clips": [
            {"at_s": 0.5, "duration_s": 1.5, "start_s": 9.0,
             "transform": {"scale": 0.35, "x": 0.6, "y": -0.6, "opacity": 0.9}},
        ]},
    ]
    render_id = montar(job_id, [{"title": "Em camadas", "layers": camadas}])
    run_render()

    pedido = api().get(f"/api/renders/{render_id}").json()
    assert pedido["status"] == "done", pedido["error"]
    clip = pedido["clips"][0]

    assert clip["meta"]["composed"] is True
    assert clip["meta"]["layers"] == 2
    assert clip["meta"]["segments"] == 3
    assert clip["video_url"], "o video nao saiu"
    # o pedido sabe se contar
    assert pedido["timelines"][0]["n_layers"] == 2
    assert pedido["timelines"][0]["n_cuts"] == 3

    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    saida = local_copy(key, Path(isolated.work_dir) / "camadas")
    info = ffmpeg.probe(saida)
    original = ffmpeg.probe(short_sample)

    # 0 -> 4.5s: o ultimo clipe termina em 4.5
    assert info.duration_s == pytest.approx(4.5, abs=0.35)
    # a tela e a da gravacao: a sobreposicao nao muda o quadro
    assert (info.width, info.height) == (original.width, original.height)


def test_montagem_de_uma_camada_so_continua_pelo_caminho_antigo(
    isolated, short_sample
):
    """E o caminho que sobrevive a um corte ruim, entao ele fica com o comum."""
    job_id = run_analysis(short_sample)
    render_id = montar(
        job_id,
        [{"layers": [{"clips": [
            {"at_s": 0.0, "duration_s": 1.5, "start_s": 1.0},
            {"at_s": 3.0, "duration_s": 1.5, "start_s": 6.0},
        ]}]}],
    )
    run_render()

    clip = api().get(f"/api/renders/{render_id}").json()["clips"][0]
    assert "composed" not in clip["meta"]
    # e o zip dos cortes, que so o caminho antigo produz, continua vindo
    assert clip["segments_zip_url"]
    assert clip["meta"]["blackfill_s"] == pytest.approx(1.5, abs=0.05)


def test_clipe_fora_da_gravacao_vira_fundo_sem_mover_os_outros(isolated):
    """A mesma promessa da V1, agora no grafo."""
    from owcore.compose import compor
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=59),   # so 1s existe
        TimelineClip(at_s=4, duration_s=1, start_s=1),
    ])])
    c = compor(t, source=Path("x.mp4"), width=640, height=360, fps=30,
               source_duration_s=60)

    # o primeiro entra aparado em 1s, e o segundo continua entrando aos 4s
    assert "trim=duration=1.000" in c.filter_complex
    assert "between(t,4.000,5.000)" in c.filter_complex
    assert c.duracao_s == pytest.approx(5.0)


def test_fonte_que_ainda_nao_da_para_montar_e_recusada(isolated):
    """Ignorar em silencio seria pior do que nao aceitar."""
    from owcore.compose import compor
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=1, source="color", fill="black"),
    ])])
    with pytest.raises(ValueError, match="ainda nao e montavel"):
        compor(t, source=Path("x.mp4"), width=640, height=360, fps=30)


def test_camada_muda_entra_sem_som(isolated):
    from owcore.compose import compor
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[
        Layer(clips=[TimelineClip(at_s=0, duration_s=1, start_s=1)]),
        Layer(muted=True, clips=[TimelineClip(at_s=0, duration_s=1, start_s=5)]),
    ])
    c = compor(t, source=Path("x.mp4"), width=640, height=360, fps=30)

    # dois videos, um audio so
    assert c.filter_complex.count("overlay=") == 2
    assert "amix" not in c.filter_complex
    assert c.mapa_audio == "[aout]"


# ── a biblioteca de midia (Fase 4) ──────────────────────────────────────────


def subir_media(job_id: str, nome: str, dados: bytes) -> str:
    """Traz um arquivo e roda o worker que o analisa."""
    resp = api().post(
        f"/api/jobs/{job_id}/media", files={"file": (nome, dados, "application/octet-stream")}
    )
    assert resp.status_code == 201, resp.text
    media_id = resp.json()["id"]

    worker = service_module("beats", "main").MediaAnalyzer()
    for payload in drain(STREAM_MEDIA, "media"):
        worker.handle(payload)
    return media_id


def png_de_teste(destino: Path, cor: str = "red") -> Path:
    """Uma imagem qualquer, feita pelo proprio ffmpeg."""
    from owcore.config import get_settings

    subprocess.run(
        [get_settings().ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c={cor}:s=320x180", "-frames:v", "1", str(destino)],
        check=True,
    )
    return destino


def test_a_musica_virou_um_item_da_biblioteca(isolated, short_sample):
    """Generalizar `Track` custou uma coluna e evitou um segundo sistema de
    upload vivendo ao lado do primeiro."""
    job_id = run_analysis(short_sample)
    track_id = subir_musica(job_id)

    detail = api().get(f"/api/jobs/{job_id}").json()
    assert [m["id"] for m in detail["media"]] == [track_id]
    assert detail["media"][0]["kind"] == "audio"
    # e continua aparecendo como musica, que e o que o seletor de trilha usa
    assert [t["id"] for t in detail["tracks"]] == [track_id]


def test_um_video_importado_ganha_dimensoes_miniatura_e_proxy(
    isolated, short_sample
):
    job_id = run_analysis(short_sample)
    media_id = subir_media(job_id, "clipe.mp4", short_sample.read_bytes())

    item = api().get(f"/api/media/{media_id}").json()
    assert item["status"] == "ready", item["error"]
    assert item["kind"] == "video"
    assert item["width"] > 0 and item["height"] > 0
    assert item["fps"] > 0
    assert item["duration_s"] > 5
    assert item["thumb_url"], "sem miniatura nao da para escolher na lista"
    assert item["proxy_url"], "sem proxy o monitor arrastaria o arquivo cheio"

    # e os dois sao servidos
    assert api().get(item["thumb_url"]).status_code == 200
    assert api().get(item["proxy_url"]).status_code == 200


def test_uma_imagem_ganha_dimensoes_e_miniatura_mas_nao_duracao(
    isolated, short_sample, tmp_path
):
    """Quanto uma imagem fica na tela e escolha da montagem, nao propriedade do
    arquivo."""
    job_id = run_analysis(short_sample)
    png = png_de_teste(tmp_path / "logo.png")
    media_id = subir_media(job_id, "logo.png", png.read_bytes())

    item = api().get(f"/api/media/{media_id}").json()
    assert item["status"] == "ready", item["error"]
    assert item["kind"] == "image"
    assert (item["width"], item["height"]) == (320, 180)
    assert item["duration_s"] == 0
    assert item["thumb_url"]
    assert item["proxy_url"] is None, "imagem nao precisa de proxy"


def test_arquivo_de_tipo_desconhecido_e_recusado(isolated, short_sample):
    """Aceitar e falhar depois seria pior do que dizer nao agora."""
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/media",
        files={"file": ("planilha.xlsx", b"nao sou midia", "application/octet-stream")},
    )
    assert resp.status_code == 422
    assert "nao sei o que fazer" in resp.json()["detail"]


def test_uma_imagem_entra_na_montagem_como_qualquer_clipe(
    isolated, short_sample, tmp_path
):
    """O caminho inteiro: importar, montar por cima e conferir o mp4."""
    job_id = run_analysis(short_sample)
    png = png_de_teste(tmp_path / "selo.png", cor="blue")
    media_id = subir_media(job_id, "selo.png", png.read_bytes())

    camadas = [
        {"clips": [{"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0}]},
        {"name": "selo", "clips": [
            {"at_s": 0.5, "duration_s": 1.0, "source": "media",
             "media_id": media_id,
             "transform": {"scale": 0.5, "x": 0.5, "y": -0.5}},
        ]},
    ]
    render_id = montar(job_id, [{"title": "Com selo", "layers": camadas}])
    run_render()

    pedido = api().get(f"/api/renders/{render_id}").json()
    assert pedido["status"] == "done", pedido["error"]
    clip = pedido["clips"][0]
    assert clip["meta"]["media"] == 1
    assert clip["video_url"], "o video nao saiu"

    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    saida = local_copy(key, Path(isolated.work_dir) / "com_selo")
    assert ffmpeg.probe(saida).duration_s == pytest.approx(2.0, abs=0.35)


def test_midia_de_outro_job_e_recusada_no_pedido(isolated, short_sample):
    """A montagem sairia sem ela, e sem aviso."""
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([
            {"layers": [{"clips": [
                {"at_s": 0, "duration_s": 1, "source": "media",
                 "media_id": "naoexiste"},
            ]}]}
        ])},
    )
    assert resp.status_code == 422
    assert "midia desconhecida" in resp.json()["detail"]


def test_tirar_da_biblioteca(isolated, short_sample, tmp_path):
    job_id = run_analysis(short_sample)
    png = png_de_teste(tmp_path / "x.png")
    media_id = subir_media(job_id, "x.png", png.read_bytes())

    assert api().delete(f"/api/media/{media_id}").status_code == 204
    assert api().get(f"/api/media/{media_id}").status_code == 404
    assert api().get(f"/api/jobs/{job_id}").json()["media"] == []


# ── efeitos (Fase 5) ────────────────────────────────────────────────────────


def test_velocidade_muda_quanto_da_fonte_o_clipe_come(isolated):
    """Nao a duracao dele no video -- essa e o que o usuario arrasta."""
    from owcore.models import TimelineClip

    lento = TimelineClip(at_s=0, duration_s=2, start_s=10, speed=0.5)
    rapido = TimelineClip(at_s=0, duration_s=2, start_s=10, speed=2.0)

    assert lento.fonte_consumida_s == pytest.approx(1.0)
    assert rapido.fonte_consumida_s == pytest.approx(4.0)
    # e onde ele termina na gravacao muda junto
    assert lento.end_s == pytest.approx(11.0)
    assert rapido.end_s == pytest.approx(14.0)
    # mas os dois ocupam os mesmos 2s do video
    assert lento.until_s == rapido.until_s == pytest.approx(2.0)


def test_o_grafo_acelera_imagem_e_som_juntos(isolated):
    """Descompasso entre imagem e som e pior do que nao ter som."""
    from owcore.compose import compor
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=10, speed=0.4),
    ])])
    g = compor(t, source=Path("x.mp4"), width=640, height=360, fps=30).filter_complex

    # 2s de video a 0.4x comem 0.8s de gravacao
    assert "trim=duration=0.800" in g
    assert "setpts=PTS/0.4000" in g
    # `atempo` so aceita de 0.5 em diante, entao 0.4 vira 0.5 x 0.8
    assert "atempo=0.5" in g and "atempo=0.8000" in g


def test_a_ordem_dos_filtros_poe_o_fade_no_relogio_do_video(isolated):
    """Um fade de meio segundo dura meio segundo no video, nao na fonte."""
    from owcore.compose import compor
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1, speed=2.0,
                     fade={"in_s": 0.5, "out_s": 0.5}),
    ])])
    g = compor(t, source=Path("x.mp4"), width=640, height=360, fps=30).filter_complex

    # a velocidade vem antes do fade: ela muda o relogio do clipe
    assert g.index("setpts=PTS/2.0000") < g.index("fade=t=in")
    # e o fade de saida comeca contando a duracao no *video*
    assert "fade=t=out:st=1.500:d=0.500" in g


def test_cor_e_aplicada_e_o_neutro_nao_polui_o_grafo(isolated):
    from owcore.compose import compor
    from owcore.models import Layer, Timeline, TimelineClip

    def grafo(**kw):
        t = Timeline(layers=[Layer(clips=[
            TimelineClip(at_s=0, duration_s=1, start_s=1, **kw),
        ])])
        return compor(t, source=Path("x.mp4"), width=640, height=360,
                      fps=30).filter_complex

    assert "eq=" not in grafo()
    assert "saturation=1.4000" in grafo(color={"saturation": 1.4})


def test_a_musica_deixa_o_jogo_aparecer_por_baixo(isolated):
    """Com `game_volume` em 0 ela substitui, como na V1; acima disso, mistura."""
    from owcore.compose import compor
    from owcore.models import Layer, Timeline, TimelineClip

    def grafo(**kw):
        t = Timeline(layers=[Layer(clips=[
            TimelineClip(at_s=0, duration_s=1, start_s=1),
        ])], **kw)
        return compor(t, source=Path("x.mp4"), width=640, height=360, fps=30,
                      music=Path("m.mp3")).filter_complex

    # o padrao continua sendo o da V1: a trilha manda sozinha
    assert "[jogo]" not in grafo()
    misturado = grafo(game_volume=0.5, music_volume=0.8)
    assert "volume=0.8000[trilha]" in misturado
    assert "[trilha][jogo]amix" in misturado


def test_efeito_tira_a_montagem_do_caminho_de_corte_e_emenda(isolated):
    """Corte-e-emenda nao sabe fazer nada disto."""
    from owcore.models import Layer, Timeline, TimelineClip

    def so_uma_camada(**kw):
        return Timeline(layers=[Layer(clips=[
            TimelineClip(at_s=0, duration_s=1, start_s=1, **kw),
        ])]).de_uma_camada_so

    assert so_uma_camada()
    assert not so_uma_camada(speed=2.0)
    assert not so_uma_camada(fade={"in_s": 0.2})
    assert not so_uma_camada(color={"contrast": 1.2})


def test_efeitos_absurdos_sao_recusados(isolated):
    """Guardar lixo agora seria um render quebrado depois."""
    from owcore.models import TimelineClip

    with pytest.raises(ValueError, match="speed"):
        TimelineClip(at_s=0, duration_s=1, start_s=0, speed=50)
    with pytest.raises(ValueError, match="fades somados"):
        TimelineClip(at_s=0, duration_s=1, start_s=0,
                     fade={"in_s": 0.7, "out_s": 0.7})
    with pytest.raises(ValueError, match="saturation"):
        TimelineClip(at_s=0, duration_s=1, start_s=0,
                     color={"saturation": 9})


@pytest.mark.skipif(not MUSIC.exists(), reason="precisa do data/sample/music.wav")
def test_camera_lenta_e_fade_viram_video_de_verdade(isolated, short_sample):
    """Do pedido ao mp4, com o relogio conferido."""
    job_id = run_analysis(short_sample)
    track_id = subir_musica(job_id)

    camadas = [{"clips": [
        {"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0, "speed": 0.5,
         "fade": {"in_s": 0.4}, "color": {"saturation": 1.3}},
        {"at_s": 2.0, "duration_s": 1.5, "start_s": 6.0, "speed": 2.0,
         "fade": {"out_s": 0.5}},
    ]}]
    render_id = montar(job_id, [{
        "title": "Com efeitos", "track_id": track_id,
        "music_volume": 0.9, "game_volume": 0.3, "layers": camadas,
    }])
    run_render()

    pedido = api().get(f"/api/renders/{render_id}").json()
    assert pedido["status"] == "done", pedido["error"]
    clip = pedido["clips"][0]
    assert clip["meta"]["composed"] is True
    assert clip["video_url"], "o video nao saiu"

    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    saida = local_copy(key, Path(isolated.work_dir) / "efeitos")
    info = ffmpeg.probe(saida)

    # 2s + 1.5s: a velocidade muda o que se consome da fonte, nao o que se ve
    assert info.duration_s == pytest.approx(3.5, abs=0.35)
    assert info.has_audio
