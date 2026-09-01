"""Integracao: o job atravessa todos os microsservicos.

Nao sobe processos -- instancia cada worker e entrega a ele as mensagens que o
barramento produziu. E isso que verifica os *contratos*: se o preprocessor
mudar o formato de `RoiReady`, ou um detector parar de avisar o planejador,
este teste quebra.

O sistema tem duas fases, e os testes seguem essa divisao:

* **analise** -- roda uma vez por gravacao e termina em `ready`, com a lista de
  videos que da para gerar. Nenhuma musica passa por aqui.
* **geracao** -- o usuario escolhe o que quer, com a musica de cada video, e
  pode pedir quantas vezes quiser sobre a mesma analise.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import service_module
from owcore.bus import get_bus
from owcore.db import session
from owcore.models import (
    DETECTORS,
    STREAM_EDIT,
    STREAM_JOBS,
    STREAM_RENDER_READY,
    STREAM_ROI,
    Job,
    JobStatus,
    Render,
    RenderStatus,
)


def drain(stream: str, group: str) -> list[dict]:
    """Tira do barramento tudo que estiver esperando naquele grupo."""
    bus = get_bus()
    out: list[dict] = []
    while True:
        msgs = list(bus.consume(stream, group, "teste", block_ms=0))
        if not msgs:
            return out
        for m in msgs:
            out.append(m.payload)
            bus.ack(stream, group, m.id)


def api() -> TestClient:
    return TestClient(service_module("gateway", "app").app)


# ── fase 1: analise ─────────────────────────────────────────────────────────


def run_analysis(video_path, params: str = "{}", *, detectores=None) -> str:
    """Sobe a gravacao pela API e roda a analise inteira na mao."""
    client = api()
    resp = client.post(
        "/api/jobs",
        files={"video": ("match.mp4", video_path.read_bytes(), "video/mp4")},
        data={"params": params},
    )
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["id"]

    preprocessor = service_module("preprocessor", "main").Preprocessor()
    for payload in drain(STREAM_JOBS, "preprocessor"):
        preprocessor.handle(payload)

    workers = detectores if detectores is not None else todos_os_detectores()
    for w in workers:
        for payload in drain(STREAM_ROI, w.group):
            if w.accepts(payload):
                w.handle(payload)

    planner = service_module("planner", "main").Planner()
    for payload in drain(STREAM_EDIT, "planner"):
        planner.handle(payload)

    return job_id


def todos_os_detectores() -> list:
    return [
        service_module("detector_kills", "main").KillsDetector(),
        service_module("detector_survival", "main").SurvivalDetector(),
        service_module("detector_ults", "main").UltsDetector(),
        service_module("detector_banner", "main").BannerDetector(),
        service_module("detector_killfeed", "main").KillfeedDetector(),
    ]


# ── fase 2: geracao ─────────────────────────────────────────────────────────

#: os eventos que o editor poe na prateleira -- e de onde saem os cortes daqui
MOMENTOS = {"kill", "headshot", "ability_kill", "sleep", "stun", "ult_negated",
            "escape"}


def momentos_de(job_id: str) -> list[float]:
    detail = api().get(f"/api/jobs/{job_id}").json()
    return [e["t"] for e in detail["events"] if e["kind"] in MOMENTOS]


def montagem(
    job_id: str,
    *,
    quantos: int = 3,
    duracao: float = 1.0,
    espaco: float = 0.0,
    titulo: str = "Montagem",
) -> dict:
    """Uma linha do tempo como a que sai do editor: N momentos em sequencia.

    E o unico jeito de gerar video no sistema. Ate a Fase 11 havia outro -- o
    app escolhia propostas prontas e o servidor decidia os cortes --, e era ele
    que estes testes usavam.
    """
    instantes = sorted(set(momentos_de(job_id)))[:quantos]
    assert instantes, "a analise nao achou momento nenhum para montar"
    return {
        "title": titulo,
        "cuts": [
            {
                "source_t": t,
                # meio segundo de embalo, sem sair do comeco da gravacao
                "start_s": max(0.0, t - 0.5),
                "duration_s": duracao,
                "at_s": i * (duracao + espaco),
            }
            for i, t in enumerate(instantes)
        ],
    }


def pedir(job_id: str, timelines: list[dict] | None = None) -> str:
    """Manda gerar, como o app faz."""
    if timelines is None:
        timelines = [montagem(job_id)]
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps(timelines)},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def run_render() -> None:
    """Roda a edicao para todos os pedidos que estiverem na fila."""
    editor = service_module("editor", "main").Editor()
    for payload in drain(STREAM_RENDER_READY, "editor"):
        editor.handle(payload)


def run_pipeline(
    video_path, params: str = "{}", *, timelines: list[dict] | None = None
) -> tuple[str, str]:
    """Analise + uma geracao. Devolve (job_id, render_id)."""
    job_id = run_analysis(video_path, params)
    render_id = pedir(job_id, timelines)
    run_render()
    return job_id, render_id


# ── a analise, sozinha ──────────────────────────────────────────────────────


def test_analise_para_em_ready_com_os_momentos_da_partida(isolated, short_sample):
    job_id = run_analysis(short_sample)

    with session() as s:
        job = s.get(Job, job_id)
        assert job.status == JobStatus.READY, job.error
        assert job.duration_s > 10
        assert {r.detector for r in job.reports} == set(DETECTORS)
        assert all(r.ok for r in job.reports)
        assert any(e.kind == "kill" for e in job.events)
        # a analise nao gera video nenhum: isso e a segunda fase
        assert job.clips == []
        assert job.renders == []


def test_a_analise_entrega_os_momentos_que_o_editor_mostra(isolated, short_sample):
    """O contrato entre a analise e o editor: o que o detector achou aparece na
    prateleira. Um tipo detectado que nao chega ate aqui e trabalho jogado
    fora -- foi o que aconteceu com os headshots e as mortes por habilidade."""
    job_id = run_analysis(short_sample)
    detail = api().get(f"/api/jobs/{job_id}").json()

    assert detail["status"] == "ready"
    tipos = {e["kind"] for e in detail["events"]}
    assert tipos & MOMENTOS, "nenhum momento montavel"
    # o `meta` vem junto: e nele que esta *qual* habilidade matou
    assert all("meta" in e for e in detail["events"])


def test_o_fim_da_analise_nao_chega_enquanto_falta_detector(isolated, short_sample):
    kills = service_module("detector_kills", "main").KillsDetector()
    job_id = run_analysis(short_sample, detectores=[kills])

    with session() as s:
        job = s.get(Job, job_id)
        assert job.status == JobStatus.DETECTING


def test_falha_de_um_detector_nao_derruba_o_job(isolated, short_sample):
    """Contrato importante: se o detector de ults quebrar, o usuario ainda
    recebe os momentos que os outros acharam."""
    client = api()
    resp = client.post(
        "/api/jobs",
        files={"video": ("m.mp4", short_sample.read_bytes(), "video/mp4")},
        data={"params": "{}"},
    )
    job_id = resp.json()["id"]

    preprocessor = service_module("preprocessor", "main").Preprocessor()
    for payload in drain(STREAM_JOBS, "preprocessor"):
        preprocessor.handle(payload)

    for w in todos_os_detectores():
        for payload in drain(STREAM_ROI, w.group):
            if not w.accepts(payload):
                continue
            if w.detector == "ults":
                w._dispatch(_FakeBus(), _FakeMsg(payload))  # estoura de proposito
            else:
                w.handle(payload)

    planner = service_module("planner", "main").Planner()
    for payload in drain(STREAM_EDIT, "planner"):
        planner.handle(payload)

    with session() as s:
        job = s.get(Job, job_id)
        assert job.status == JobStatus.READY, job.error
        ults = next(r for r in job.reports if r.detector == "ults")
        assert not ults.ok and ults.error
        assert job.events, "deveria entregar o que os outros acharam"


class _FakeMsg:
    def __init__(self, payload: dict):
        self.id = "x"
        self.payload = dict(payload)
        # artefato inexistente: o detector estoura ao tentar baixar
        self.payload["artifacts"] = [
            {"key": "nao/existe.mp4", "kind": "roi", "meta": {"roi": "killfeed"}}
        ]


class _FakeBus:
    def ack(self, *_a): ...


def test_a_ultimate_anulada_e_gravada_como_evento(isolated):
    """Regressao: ela existia so na cabeca do gerador de propostas.

    Uma ultimate anulada e ultimate inimiga seguida de eliminacao -- nenhum
    detector sozinho ve as duas metades, entao quem fecha a analise cruza os
    dois e grava o resultado. Enquanto havia propostas, esse cruzamento
    acontecia dentro do gerador de propostas e morria ali: o tipo aparecia na
    lista do editor e na extracao de miniaturas, mas nenhum evento desse tipo
    chegava a existir no banco.
    """
    from owcore.jobs import load_events, save_events
    from owcore.models import DetectionEvent, EventKind, Job, JobStatus

    with session() as s:
        s.add(Job(
            id="j1", video_key="k", video_name="v.mp4",
            status=JobStatus.DETECTING, duration_s=60.0,
        ))
    save_events("j1", "ults", [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)])
    save_events("j1", "kills", [DetectionEvent(kind=EventKind.KILL, t=41.5)])

    service_module("planner", "main").Planner()._close_analysis("j1")

    anuladas = [e for e in load_events("j1") if e.kind == EventKind.ULT_NEGATED]
    assert len(anuladas) == 1, "o cruzamento nao virou evento"
    assert anuladas[0].t == 41.5
    assert anuladas[0].meta["delay_s"] == 1.5

    with session() as s:
        assert s.get(Job, "j1").status == JobStatus.READY


def test_fechar_duas_vezes_nao_duplica_o_que_foi_cruzado(isolated):
    """Todos os detectores avisam quase juntos; so um pode fechar a analise."""
    from owcore.jobs import load_events, save_events
    from owcore.models import DetectionEvent, EventKind, Job, JobStatus

    with session() as s:
        s.add(Job(
            id="j1", video_key="k", video_name="v.mp4",
            status=JobStatus.DETECTING, duration_s=60.0,
        ))
    save_events("j1", "ults", [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)])
    save_events("j1", "kills", [DetectionEvent(kind=EventKind.KILL, t=41.5)])

    planner = service_module("planner", "main").Planner()
    planner._close_analysis("j1")
    planner._close_analysis("j1")

    anuladas = [e for e in load_events("j1") if e.kind == EventKind.ULT_NEGATED]
    assert len(anuladas) == 1


# ── a geracao ───────────────────────────────────────────────────────────────


def test_geracao_produz_os_clipes_montados(isolated, short_sample):
    job_id, render_id = run_pipeline(short_sample)

    with session() as s:
        pedido = s.get(Render, render_id)
        assert pedido.status == RenderStatus.DONE, pedido.error
        assert pedido.clips, "nenhum clipe foi gerado"
        for c in pedido.clips:
            assert c.key, "clipe sem blob no storage"
            assert c.kind == "custom", "todo video sai do editor"
            assert c.meta["hand_made"] is True


def test_o_mesmo_momento_serve_a_varios_videos(isolated, short_sample):
    """Usar um corte num video nao o consome: da para montar de novo com os
    mesmos instantes, e o pedido anterior continua intacto."""
    job_id = run_analysis(short_sample)
    pedir(job_id, [montagem(job_id, titulo="Curto", duracao=0.8)])
    run_render()
    pedir(job_id, [montagem(job_id, titulo="Longo", duracao=1.5)])
    run_render()

    with session() as s:
        job = s.get(Job, job_id)
        assert len(job.renders) == 2
        assert all(r.status == RenderStatus.DONE for r in job.renders)
        assert all(r.clips for r in job.renders)


def test_dois_videos_no_mesmo_pedido(isolated, short_sample):
    """Um pedido pode levar mais de uma montagem -- e o que faz o corte para o
    Shorts e a versao longa sairem juntos, da mesma partida."""
    job_id = run_analysis(short_sample)
    render_id = pedir(job_id, [
        montagem(job_id, titulo="Shorts", quantos=2, duracao=0.8),
        montagem(job_id, titulo="Completo", quantos=3, duracao=1.5),
    ])
    run_render()

    with session() as s:
        pedido = s.get(Render, render_id)
        assert pedido.status == RenderStatus.DONE, pedido.error
        assert {c.title for c in pedido.clips} == {"Shorts", "Completo"}


def test_pedido_antes_da_analise_terminar_e_recusado(isolated, short_sample):
    client = api()
    resp = client.post(
        "/api/jobs",
        files={"video": ("m.mp4", short_sample.read_bytes(), "video/mp4")},
        data={"params": "{}"},
    )
    job_id = resp.json()["id"]
    resp = client.post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([
            {"cuts": [{"start_s": 0, "duration_s": 1, "at_s": 0}]}
        ])},
    )
    assert resp.status_code == 409


def test_pedido_sem_montagem_e_recusado(isolated, short_sample):
    job_id = run_analysis(short_sample)
    resp = api().post(f"/api/jobs/{job_id}/renders", data={"timelines": "[]"})
    assert resp.status_code == 422


def test_montagem_com_midia_de_outro_job_e_recusada(isolated, short_sample):
    job_id = run_analysis(short_sample)
    spec = montagem(job_id)
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([{
            "layers": [{"clips": [{
                "at_s": 0, "duration_s": 1, "start_s": 0,
                "source": "media", "media_id": "inventada",
            }]}],
        }])},
    )
    assert resp.status_code == 422, spec


def test_apagar_um_pedido_nao_apaga_a_partida(isolated, short_sample):
    job_id, render_id = run_pipeline(short_sample)
    client = api()
    assert client.delete(f"/api/renders/{render_id}").status_code == 204

    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["events"], "os momentos nao podiam sumir com o pedido"
    assert detail["renders"] == []
    assert detail["clips"] == []


# ── sem musica, o audio original ────────────────────────────────────────────


def _tem_audio(client, video_url: str, tmp_path: Path) -> bool:
    from owcore import ffmpeg

    tmp_path.mkdir(parents=True, exist_ok=True)
    dest = tmp_path / "baixado.mp4"
    dest.write_bytes(client.get(video_url).content)
    return ffmpeg.probe(dest).has_audio


def test_sem_musica_o_video_sai_com_o_audio_original(
    isolated, short_sample, tmp_path
):
    """Requisito explicito: quem nao poe musica na regua fica com o som da
    partida."""
    job_id, render_id = run_pipeline(short_sample)
    client = api()
    pedido = client.get(f"/api/renders/{render_id}").json()

    clipes = [c for c in pedido["clips"] if c["video_url"]]
    assert clipes
    for c in clipes:
        assert c["meta"]["original_audio"] is True
        assert c["meta"].get("music_name") is None
        assert _tem_audio(client, c["video_url"], tmp_path / c["id"])


# ── entrada da API ─────────────────────────────────────────────────────────


def test_upload_sem_video_e_rejeitado(isolated):
    assert api().post("/api/jobs", data={"params": "{}"}).status_code == 422


def test_parametros_invalidos_sao_rejeitados(isolated, short_sample):
    resp = api().post(
        "/api/jobs",
        files={"video": ("m.mp4", b"x", "video/mp4")},
        data={"params": "isso nao e json"},
    )
    assert resp.status_code == 422


def test_upload_truncado_e_recusado_na_porta(isolated, short_sample):
    """Um envio pela metade nao se parece com erro nenhum.

    O multipart fecha direito, o `Content-Length` bate com o que de fato
    chegou, e o que sobra e meia gravacao guardada como se estivesse inteira --
    o estrago so aparecia fases depois, no preprocessador, como um `ffprobe
    saiu com 1`. Comparar com o tamanho que o cliente diz ter enviado devolve o
    problema para a tela de envio, que e onde da para fazer algo a respeito.
    """
    bytes_do_video = short_sample.read_bytes()
    resp = api().post(
        "/api/jobs",
        files={"video": ("m.mp4", bytes_do_video[: len(bytes_do_video) // 2],
                         "video/mp4")},
        data={"params": "{}", "size": str(len(bytes_do_video))},
    )
    assert resp.status_code == 400
    assert "incompleto" in resp.json()["detail"]
    # e nada ficou para tras: nem job na fila, nem blob pela metade
    assert api().get("/api/jobs").json()["jobs"] == []
    assert not list(isolated.blob_dir.rglob("*.mp4"))


def test_upload_inteiro_passa_com_o_tamanho_conferido(isolated, short_sample):
    bytes_do_video = short_sample.read_bytes()
    resp = api().post(
        "/api/jobs",
        files={"video": ("m.mp4", bytes_do_video, "video/mp4")},
        data={"params": "{}", "size": str(len(bytes_do_video))},
    )
    assert resp.status_code == 201, resp.text


def test_job_inexistente_da_404(isolated):
    assert api().get("/api/jobs/naoexiste").status_code == 404


def test_as_horas_saem_com_fuso(isolated, short_sample):
    """Uma data sem fuso e lida como hora *local* por quem recebe -- o Dart faz
    isso. Como o que sai daqui e UTC, sem o sufixo o app mostrava todo horario
    adiantado, e a conta de "quanto falta" dava tempo negativo."""
    from datetime import datetime, timezone

    client = api()
    job_id = client.post(
        "/api/jobs",
        files={"video": ("m.mp4", short_sample.read_bytes(), "video/mp4")},
        data={"params": "{}"},
    ).json()["id"]
    d = client.get(f"/api/jobs/{job_id}").json()
    for campo in ("created_at", "updated_at"):
        lido = datetime.fromisoformat(d[campo])
        assert lido.tzinfo is not None, f"{campo} veio sem fuso: {d[campo]!r}"
        # e a hora tem de ser agora, nao daqui a tres horas
        atraso = abs((datetime.now(timezone.utc) - lido).total_seconds())
        assert atraso < 300, f"{campo} destoa do relogio em {atraso:.0f}s"


def test_pedido_inexistente_da_404(isolated):
    assert api().get("/api/renders/naoexiste").status_code == 404


def test_clipes_aparecem_na_api(isolated, short_sample):
    job_id, _render = run_pipeline(short_sample)
    client = api()

    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["clips"]

    clip = detail["clips"][0]
    video = client.get(clip["video_url"])
    assert video.status_code == 200
    assert video.content[4:8] == b"ftyp"

    parcial = client.get(clip["video_url"], headers={"Range": "bytes=0-99"})
    assert parcial.status_code == 206
    assert len(parcial.content) == 100


# ── zip com os cortes ──────────────────────────────────────────────────────
#
# O zip so existe no caminho de corte-e-emenda -- uma camada, sem musica --,
# porque so ali cada corte chega a ser um arquivo. Montagem em camadas vai por
# um grafo de filtros, onde os pedacos nunca existem separados.


def _abrir_zip(resp):
    import io
    import zipfile

    return zipfile.ZipFile(io.BytesIO(resp.content))


def test_montagem_oferece_os_cortes_em_zip(isolated, short_sample):
    job_id, _render = run_pipeline(short_sample)
    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    montada = detail["clips"][0]

    assert montada["segments_zip_url"]
    resp = client.get(montada["segments_zip_url"])
    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"]

    with _abrir_zip(resp) as zf:
        nomes = zf.namelist()
        assert nomes == sorted(nomes), "os cortes vêm em ordem cronológica"
        assert len(nomes) == montada["meta"]["segments"]
        assert all(n.endswith(".mp4") for n in nomes)
        assert zf.testzip() is None
        # o nome diz de onde o corte saiu na gravação
        assert all("m" in n and "s.mp4" in n for n in nomes)


def test_o_mesmo_trecho_montado_duas_vezes_entra_uma_vez_no_zip(
    isolated, short_sample
):
    """O zip é material para reedição: guardar o mesmo corte duas vezes não
    acrescentaria nada a quem vai reeditar, ainda que o vídeo o repita."""
    job_id = run_analysis(short_sample)
    instante = sorted(set(momentos_de(job_id)))[0]
    corte = {"source_t": instante, "start_s": max(0.0, instante - 0.5),
             "duration_s": 1.0}
    render_id = pedir(job_id, [{
        "title": "Repetida",
        "cuts": [{**corte, "at_s": 0.0}, {**corte, "at_s": 1.0}],
    }])
    run_render()

    detail = api().get(f"/api/jobs/{job_id}").json()
    montada = detail["clips"][0]
    assert montada["meta"]["segments"] == 2

    with _abrir_zip(api().get(montada["segments_zip_url"])) as zf:
        nomes = zf.namelist()
    assert len(nomes) == 1, "o mesmo trecho nao se guarda duas vezes"
    assert render_id


def test_clipe_sem_cortes_separados_da_404(isolated, short_sample):
    """Montagem em camadas nao gera zip: os pedacos nunca viram arquivo."""
    job_id = run_analysis(short_sample)
    instante = sorted(set(momentos_de(job_id)))[0]
    pedir(job_id, [{
        "title": "Em camadas",
        "layers": [
            {"clips": [{"at_s": 0, "duration_s": 1.0,
                        "start_s": max(0.0, instante - 0.5),
                        "source_t": instante}]},
            {"name": "texto", "clips": [{"at_s": 0, "duration_s": 1.0,
                                         "start_s": 0, "source": "text",
                                         "text": "OI"}]},
        ],
    }])
    run_render()

    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    unico = detail["clips"][0]
    assert unico["meta"]["composed"] is True
    assert unico["segments_zip_url"] is None
    assert client.get(f"/api/clips/{unico['id']}/cortes.zip").status_code == 404


# ── pacote da partida inteira ──────────────────────────────────────────────


def test_zip_da_partida_traz_videos_e_cortes(isolated, short_sample):
    """O pacote geral tem de estar acessível pelo job, sem passar por clipe
    nenhum -- é para isso que ele existe."""
    job_id, _render = run_pipeline(short_sample)
    client = api()

    # a URL vem já na listagem: não precisa abrir o job nem o vídeo
    listagem = client.get("/api/jobs").json()["jobs"]
    resumo = next(j for j in listagem if j["id"] == job_id)
    assert resumo["zip_url"] == f"/api/jobs/{job_id}/cortes.zip"
    assert resumo["has_cuts"] is True

    resp = client.get(resumo["zip_url"])
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]

    with _abrir_zip(resp) as zf:
        nomes = zf.namelist()
        assert zf.testzip() is None
        assert [n for n in nomes if "/videos/" in n], "faltam os vídeos finais"
        assert [n for n in nomes if "/cortes/" in n], "faltam os cortes avulsos"
        assert all(n.endswith(".mp4") for n in nomes)
        # cada pedido tem a sua pasta: dois pedidos nao se sobrescrevem
        assert all(n.startswith("pedido_") for n in nomes)


def test_zip_da_partida_separa_os_pedidos(isolated, short_sample):
    job_id = run_analysis(short_sample)
    pedir(job_id, [montagem(job_id, titulo="A", duracao=0.8)])
    run_render()
    pedir(job_id, [montagem(job_id, titulo="B", duracao=1.2)])
    run_render()

    with _abrir_zip(api().get(f"/api/jobs/{job_id}/cortes.zip")) as zf:
        pastas = {n.split("/", 1)[0] for n in zf.namelist()}
    assert pastas == {"pedido_01", "pedido_02"}


def test_zip_de_partida_inexistente_da_404(isolated):
    assert api().get("/api/jobs/naoexiste/cortes.zip").status_code == 404


# ── entrega dos cortes mesmo quando o vídeo não sai ────────────────────────


def test_montagem_que_falha_ainda_entrega_os_cortes(
    isolated, short_sample, monkeypatch
):
    """Material já cortado não se joga fora porque a etapa seguinte quebrou."""
    from owcore import ffmpeg

    render = service_module("editor", "render")
    original = ffmpeg.concat

    def concat_quebrado(*a, **k):
        raise ffmpeg.FFmpegError("falha simulada ao juntar os trechos")

    monkeypatch.setattr(render.ffmpeg, "concat", concat_quebrado)
    try:
        job_id, render_id = run_pipeline(short_sample)
    finally:
        monkeypatch.setattr(render.ffmpeg, "concat", original)

    with session() as s:
        pedido = s.get(Render, render_id)
        assert pedido.status == RenderStatus.DONE, pedido.error
        montada = pedido.clips[0]
        assert montada.key == "", "não deveria haver vídeo final"
        assert montada.meta["segments_zip_key"], "os cortes se perderam"
        assert montada.meta["render_error"]
        assert "cortes" in pedido.stage or "video" in pedido.stage

    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    clipe = detail["clips"][0]
    assert clipe["video_url"] is None
    assert clipe["segments_zip_url"]
    assert client.get(clipe["segments_zip_url"]).status_code == 200
    # e o pacote da partida continua servindo, só que sem vídeo dentro
    assert detail["clips_only_cuts"] >= 1
    assert client.get(detail["zip_url"]).status_code == 200


def test_video_de_clipe_sem_montagem_da_404(isolated, short_sample, monkeypatch):
    from owcore import ffmpeg

    render = service_module("editor", "render")
    monkeypatch.setattr(
        render.ffmpeg, "concat",
        lambda *a, **k: (_ for _ in ()).throw(ffmpeg.FFmpegError("falha")),
    )
    job_id, _render = run_pipeline(short_sample)

    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    clipe = detail["clips"][0]
    assert client.get(f"/api/clips/{clipe['id']}/video").status_code == 404
