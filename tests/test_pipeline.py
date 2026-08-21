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
    STREAM_EDIT,
    STREAM_JOBS,
    STREAM_RENDER,
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
    ]


# ── fase 2: geracao ─────────────────────────────────────────────────────────


def pedir(
    job_id: str,
    *,
    music_path: Path | None = None,
    options: dict | None = None,
    kinds: set[str] | None = None,
) -> str:
    """Escolhe propostas e pede a geracao, como o app faz.

    A musica so vai nas propostas que aceitam trilha -- um trecho corrido da
    partida sai com o audio original.
    """
    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    propostas = [
        p for p in detail["proposals"]
        if kinds is None or p["kind"] in kinds
    ]
    assert propostas, f"nenhuma proposta para escolher (kinds={kinds})"

    selections = [
        {"proposal_id": p["id"],
         "options": options or {} if p["accepts_music"] else {}}
        for p in propostas
    ]
    files = {}
    if music_path is not None:
        dados = music_path.read_bytes()
        for p in propostas:
            if p["accepts_music"]:
                files[f"music_{p['id']}"] = ("music.wav", dados, "audio/wav")

    resp = client.post(
        f"/api/jobs/{job_id}/renders",
        data={"selections": json.dumps(selections)},
        files=files or None,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def run_render() -> None:
    """Roda o ritmo e a edicao para todos os pedidos que estiverem na fila."""
    beats = service_module("beats", "main").BeatsService()
    for payload in drain(STREAM_RENDER, "beats"):
        beats.handle(payload)

    editor = service_module("editor", "main").Editor()
    for payload in drain(STREAM_RENDER_READY, "editor"):
        editor.handle(payload)


def run_pipeline(
    video_path,
    music_path=None,
    params: str = "{}",
    *,
    options: dict | None = None,
    kinds: set[str] | None = None,
) -> tuple[str, str]:
    """Analise + uma geracao com tudo que apareceu. Devolve (job_id, render_id)."""
    job_id = run_analysis(video_path, params)
    render_id = pedir(job_id, music_path=music_path, options=options, kinds=kinds)
    run_render()
    return job_id, render_id


# ── a analise, sozinha ──────────────────────────────────────────────────────


def test_analise_para_em_ready_com_a_lista_do_que_da_para_gerar(
    isolated, short_sample
):
    job_id = run_analysis(short_sample)

    with session() as s:
        job = s.get(Job, job_id)
        assert job.status == JobStatus.READY, job.error
        assert job.duration_s > 10
        assert {r.detector for r in job.reports} == {
            "kills", "survival", "ults", "banner",
        }
        assert all(r.ok for r in job.reports)
        assert any(e.kind == "kill" for e in job.events)
        assert job.proposals, "nenhuma proposta foi montada"
        # a analise nao gera video nenhum: isso e a segunda fase
        assert job.clips == []
        assert job.renders == []


def test_propostas_dizem_quais_aceitam_trilha(isolated, short_sample):
    job_id = run_analysis(short_sample)
    detail = api().get(f"/api/jobs/{job_id}").json()

    assert detail["status"] == "ready"
    montagens = [p for p in detail["proposals"] if p["accepts_music"]]
    assert montagens, "faltou a montagem no ritmo"
    assert all(p["n_moments"] >= 1 for p in montagens)
    # trecho corrido da partida nao aceita trilha: sai com o audio do jogo
    corridos = [p for p in detail["proposals"] if not p["accepts_music"]]
    assert all(p["kind"] in {"multikill", "solo_wipe", "escape"} for p in corridos)


def test_planejador_nao_age_enquanto_falta_detector(isolated, short_sample):
    kills = service_module("detector_kills", "main").KillsDetector()
    job_id = run_analysis(short_sample, detectores=[kills])

    with session() as s:
        job = s.get(Job, job_id)
        assert job.status == JobStatus.DETECTING
        assert job.proposals == []


def test_falha_de_um_detector_nao_derruba_o_job(isolated, short_sample):
    """Contrato importante: se o detector de ults quebrar, o usuario ainda
    recebe as propostas de eliminacao."""
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
        assert job.proposals, "deveria propor com o que os outros acharam"


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


# ── a geracao ───────────────────────────────────────────────────────────────


def test_geracao_produz_os_clipes_escolhidos(isolated, short_sample):
    job_id, render_id = run_pipeline(short_sample)

    with session() as s:
        pedido = s.get(Render, render_id)
        assert pedido.status == RenderStatus.DONE, pedido.error
        assert pedido.clips, "nenhum clipe foi gerado"
        for c in pedido.clips:
            assert c.key, "clipe sem blob no storage"
            assert c.proposal_id, "clipe sem vinculo com a proposta"


def test_com_musica_a_montagem_fica_no_ritmo(isolated, short_sample, tmp_path):
    musica = _musica(tmp_path)
    _job, render_id = run_pipeline(short_sample, musica)

    with session() as s:
        pedido = s.get(Render, render_id)
        assert pedido.status == RenderStatus.DONE, pedido.error
        assert pedido.beats, "nenhuma grade de batidas foi guardada"
        bpm = next(iter(pedido.beats.values()))["bpm"]
        assert 110 <= bpm <= 130
        montagem = next(c for c in pedido.clips if c.kind == "beat_montage")
        assert montagem.meta["beat_synced"] is True


def test_musicas_diferentes_no_mesmo_pedido(isolated, short_sample, tmp_path):
    """A exigencia central da segunda fase: cada video com a sua trilha."""
    lenta = _musica(tmp_path / "a", bpm=90.0)
    rapida = _musica(tmp_path / "b", bpm=160.0)

    job_id = run_analysis(short_sample, json.dumps({"multikill_min": 3}))
    client = api()
    propostas = [
        p for p in client.get(f"/api/jobs/{job_id}").json()["proposals"]
        if p["accepts_music"]
    ]
    if len(propostas) < 2:
        # o video sintetico curto nem sempre rende duas montagens; entao
        # verifica-se o mesmo contrato com uma musica so por proposta
        propostas = propostas[:1]

    files = {}
    for p, musica in zip(propostas, (lenta, rapida)):
        files[f"music_{p['id']}"] = ("m.wav", musica.read_bytes(), "audio/wav")

    resp = client.post(
        f"/api/jobs/{job_id}/renders",
        data={"selections": json.dumps(
            [{"proposal_id": p["id"]} for p in propostas]
        )},
        files=files,
    )
    assert resp.status_code == 201, resp.text
    render_id = resp.json()["id"]
    run_render()

    with session() as s:
        pedido = s.get(Render, render_id)
        assert pedido.status == RenderStatus.DONE, pedido.error
        # cada escolha guardou o seu proprio arquivo de musica
        chaves = {sel["music_key"] for sel in pedido.selections}
        assert len(chaves) == len(propostas)
        assert len(pedido.beats) == len(propostas)


def test_o_mesmo_momento_serve_a_varios_videos(isolated, short_sample, tmp_path):
    """Usar um corte num video nao o consome: da para pedir de novo, com outra
    musica, e as propostas continuam intactas."""
    job_id = run_analysis(short_sample)
    pedir(job_id, music_path=_musica(tmp_path / "a", bpm=100.0))
    run_render()
    pedir(job_id, music_path=_musica(tmp_path / "b", bpm=150.0))
    run_render()

    with session() as s:
        job = s.get(Job, job_id)
        assert len(job.renders) == 2
        assert all(r.status == RenderStatus.DONE for r in job.renders)
        a = {c.proposal_id for c in job.renders[0].clips}
        b = {c.proposal_id for c in job.renders[1].clips}
        assert a and a == b, "o segundo pedido deveria repetir os mesmos momentos"


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
        data={"selections": json.dumps([{"proposal_id": "seja_la_qual"}])},
    )
    assert resp.status_code == 409


def test_pedido_sem_escolha_e_recusado(isolated, short_sample):
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/renders", data={"selections": "[]"}
    )
    assert resp.status_code == 422


def test_proposta_de_outro_job_e_recusada(isolated, short_sample):
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"selections": json.dumps([{"proposal_id": "inventada"}])},
    )
    assert resp.status_code == 422


def test_apagar_um_pedido_preserva_as_propostas(isolated, short_sample):
    job_id, render_id = run_pipeline(short_sample)
    client = api()
    assert client.delete(f"/api/renders/{render_id}").status_code == 204

    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["proposals"], "as propostas nao podiam sumir com o pedido"
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
    """Requisito explicito: quem nao escolhe trilha fica com o som da partida --
    tanto no trecho corrido quanto na montagem."""
    job_id, render_id = run_pipeline(short_sample)
    client = api()
    pedido = client.get(f"/api/renders/{render_id}").json()

    clipes = [c for c in pedido["clips"] if c["video_url"]]
    assert clipes
    for c in clipes:
        assert c["meta"]["original_audio"] is True
        assert c["meta"].get("music_name") is None
        assert _tem_audio(client, c["video_url"], tmp_path / c["id"])


def test_com_musica_o_audio_da_partida_da_lugar_a_trilha(
    isolated, short_sample, tmp_path
):
    _job, render_id = run_pipeline(short_sample, _musica(tmp_path))
    pedido = api().get(f"/api/renders/{render_id}").json()
    montagem = next(c for c in pedido["clips"] if c["kind"] == "beat_montage")
    assert montagem["meta"]["original_audio"] is False
    assert montagem["meta"]["music_name"] == "music.wav"


# ── janela de música escolhida pelo usuário ────────────────────────────────


def _musica(dest_dir, segundos: float = 30.0, bpm: float = 120.0) -> Path:
    from conftest import tools_module

    make_sample = tools_module("make_sample")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "music.wav"
    make_sample.write_wav(dest, make_sample.click_track(segundos, bpm))
    return dest


def _montagem(pedido) -> object | None:
    return next((c for c in pedido.clips if c.kind == "beat_montage"), None)


def test_com_loop_a_montagem_tem_a_duracao_do_trecho(
    isolated, short_sample, tmp_path
):
    _job, render_id = run_pipeline(
        short_sample,
        _musica(tmp_path),
        # exigencia alta de rajada para as eliminacoes sobrarem avulsas e irem
        # todas para a montagem, que e o que este teste mede
        params=json.dumps({"multikill_min": 9}),
        options={"music_start_s": 2.0, "music_end_s": 10.0, "montage_loop": True},
    )
    with session() as s:
        pedido = s.get(Render, render_id)
        assert pedido.status == RenderStatus.DONE, pedido.error
        montagem = _montagem(pedido)
        assert montagem is not None
        assert montagem.meta["looped"] is True
        # 8s de janela, e a montagem natural do trecho curto e bem menor
        assert abs(montagem.meta["duration_s"] - 8.0) < 0.35


def test_sem_loop_a_montagem_nao_passa_do_trecho(isolated, short_sample, tmp_path):
    _job, render_id = run_pipeline(
        short_sample,
        _musica(tmp_path),
        params=json.dumps({"multikill_min": 9}),
        options={"music_start_s": 0.0, "music_end_s": 3.0, "montage_loop": False},
    )
    with session() as s:
        pedido = s.get(Render, render_id)
        assert pedido.status == RenderStatus.DONE, pedido.error
        montagem = _montagem(pedido)
        assert montagem is not None
        assert montagem.meta["looped"] is False
        assert montagem.meta["duration_s"] <= 3.0 + 1e-6


def test_janela_invalida_e_recusada_pela_api(isolated, short_sample):
    job_id = run_analysis(short_sample)
    detail = api().get(f"/api/jobs/{job_id}").json()
    proposta = next(p for p in detail["proposals"] if p["accepts_music"])
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"selections": json.dumps([{
            "proposal_id": proposta["id"],
            "options": {"music_start_s": 30, "music_end_s": 10},
        }])},
    )
    assert resp.status_code == 422


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


def test_job_inexistente_da_404(isolated):
    assert api().get("/api/jobs/naoexiste").status_code == 404


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


def test_montagem_oferece_os_cortes_em_zip(isolated, short_sample, tmp_path):
    import io
    import zipfile

    job_id, _render = run_pipeline(
        short_sample, _musica(tmp_path), params=json.dumps({"multikill_min": 9})
    )
    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    montagem = next(c for c in detail["clips"] if c["kind"] == "beat_montage")

    assert montagem["segments_zip_url"]
    resp = client.get(montagem["segments_zip_url"])
    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        nomes = zf.namelist()
        assert nomes == sorted(nomes), "os cortes vêm em ordem cronológica"
        assert len(nomes) == montagem["meta"]["segments"]
        assert all(n.endswith(".mp4") for n in nomes)
        assert zf.testzip() is None
        # o nome diz de onde o corte saiu na gravação
        assert all("m" in n and "s.mp4" in n for n in nomes)


def test_zip_nao_repete_o_trecho_aparado_do_final(isolated, short_sample, tmp_path):
    """O último trecho da montagem em loop é uma aparação de um corte que já
    está no zip; entregar as duas versões seria entregar o mesmo material duas
    vezes, uma pela metade."""
    import io
    import zipfile

    job_id, _render = run_pipeline(
        short_sample,
        _musica(tmp_path),
        params=json.dumps({"multikill_min": 9}),
        options={"music_start_s": 0, "music_end_s": 7.3, "montage_loop": True},
    )
    detail = api().get(f"/api/jobs/{job_id}").json()
    montagem = next(c for c in detail["clips"] if c["kind"] == "beat_montage")
    resp = api().get(montagem["segments_zip_url"])
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        nomes = zf.namelist()
    # um arquivo por instante distinto, sem o aparado a mais
    instantes = {n.split("_", 1)[1] for n in nomes}
    assert len(instantes) == len(nomes)


def test_zip_repetido_guarda_cada_corte_uma_vez(isolated, short_sample, tmp_path):
    """Repetir trechos alonga o vídeo, mas o zip é material para reedição:
    guardar o mesmo corte várias vezes não acrescentaria nada."""
    import io
    import zipfile

    job_id, _render = run_pipeline(
        short_sample,
        _musica(tmp_path),
        params=json.dumps({"multikill_min": 9}),
        options={"music_start_s": 0, "music_end_s": 12, "montage_loop": True},
    )
    detail = api().get(f"/api/jobs/{job_id}").json()
    montagem = next(c for c in detail["clips"] if c["kind"] == "beat_montage")
    assert montagem["meta"]["looped"] is True

    resp = api().get(montagem["segments_zip_url"])
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        nomes = zf.namelist()
    # a montagem repetiu trechos, o zip nao
    assert len(nomes) < montagem["meta"]["segments"]
    assert len(set(nomes)) == len(nomes)


def test_clipe_sem_cortes_separados_da_404(isolated, short_sample):
    job_id, _render = run_pipeline(short_sample)
    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    unico = next(c for c in detail["clips"] if c["kind"] != "beat_montage")
    assert unico["segments_zip_url"] is None
    assert client.get(f"/api/clips/{unico['id']}/cortes.zip").status_code == 404


# ── pacote da partida inteira ──────────────────────────────────────────────


def test_zip_da_partida_traz_videos_e_cortes(isolated, short_sample, tmp_path):
    """O pacote geral tem de estar acessível pelo job, sem passar por clipe
    nenhum -- é para isso que ele existe."""
    import io
    import zipfile

    job_id, _render = run_pipeline(
        short_sample, _musica(tmp_path), params=json.dumps({"multikill_min": 9})
    )
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

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        nomes = zf.namelist()
        assert zf.testzip() is None
        videos = [n for n in nomes if "/videos/" in n]
        cortes = [n for n in nomes if "/cortes/" in n]
        assert videos, "faltam os vídeos finais"
        assert cortes, "faltam os cortes avulsos"
        assert all(n.endswith(".mp4") for n in nomes)
        # cada pedido tem a sua pasta: dois pedidos nao se sobrescrevem
        assert all(n.startswith("pedido_") for n in nomes)


def test_zip_da_partida_separa_os_pedidos(isolated, short_sample, tmp_path):
    import io
    import zipfile

    job_id = run_analysis(short_sample, json.dumps({"multikill_min": 9}))
    pedir(job_id, music_path=_musica(tmp_path / "a", bpm=100.0))
    run_render()
    pedir(job_id, music_path=_musica(tmp_path / "b", bpm=150.0))
    run_render()

    resp = api().get(f"/api/jobs/{job_id}/cortes.zip")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        pastas = {n.split("/", 1)[0] for n in zf.namelist()}
    assert pastas == {"pedido_01", "pedido_02"}


def test_zip_de_partida_inexistente_da_404(isolated):
    assert api().get("/api/jobs/naoexiste/cortes.zip").status_code == 404


# ── entrega dos cortes mesmo quando o vídeo não sai ────────────────────────


def test_montagem_que_falha_ainda_entrega_os_cortes(
    isolated, short_sample, tmp_path, monkeypatch
):
    """Material já cortado não se joga fora porque a etapa seguinte quebrou."""
    from owcore import ffmpeg

    render = service_module("editor", "render")
    original = ffmpeg.concat

    def concat_quebrado(*a, **k):
        raise ffmpeg.FFmpegError("falha simulada ao juntar os trechos")

    monkeypatch.setattr(render.ffmpeg, "concat", concat_quebrado)
    try:
        job_id, render_id = run_pipeline(
            short_sample, params=json.dumps({"multikill_min": 9})
        )
    finally:
        monkeypatch.setattr(render.ffmpeg, "concat", original)

    with session() as s:
        pedido = s.get(Render, render_id)
        assert pedido.status == RenderStatus.DONE, pedido.error
        montagem = next(c for c in pedido.clips if c.kind == "beat_montage")
        assert montagem.key == "", "não deveria haver vídeo final"
        assert montagem.meta["segments_zip_key"], "os cortes se perderam"
        assert montagem.meta["render_error"]
        assert "cortes" in pedido.stage or "video" in pedido.stage

    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    clipe = next(c for c in detail["clips"] if c["kind"] == "beat_montage")
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
    job_id, _render = run_pipeline(
        short_sample, params=json.dumps({"multikill_min": 9})
    )

    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    clipe = next(c for c in detail["clips"] if c["kind"] == "beat_montage")
    assert client.get(f"/api/clips/{clipe['id']}/video").status_code == 404
