"""Barramento, storage e visao -- as pecas de infra que os microsservicos
assumem que funcionam."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from owcore.audio import peaks_for, read_wav, waveform, waveform_of
from owcore.bus import LocalBus
from owcore.storage import LocalStorage
from owcore.vision import (
    GLYPH_SIDE,
    IconBank,
    TemplateBank,
    border_mask,
    find_pulses,
    glyph_in_disc,
    glyph_on_dark,
    hsv_ratio,
    normalized_glyph,
)


# ────────────────────────────── barramento ─────────────────────────────────


def test_mensagem_publicada_e_entregue(tmp_path):
    bus = LocalBus(tmp_path)
    bus.publish("s", {"job_id": "a"})
    got = list(bus.consume("s", "g", "c1", block_ms=0))
    assert [m.payload for m in got] == [{"job_id": "a"}]


def test_cada_grupo_recebe_a_mesma_mensagem(tmp_path):
    """Fan-out: o preprocessor manda uma vez e todo detector precisa ver."""
    bus = LocalBus(tmp_path)
    bus.publish("s", {"n": 1})
    a = list(bus.consume("s", "grupo-a", "c", block_ms=0))
    b = list(bus.consume("s", "grupo-b", "c", block_ms=0))
    assert len(a) == len(b) == 1


def test_dentro_do_grupo_so_um_consumidor_pega(tmp_path):
    """Competicao: duas replicas do mesmo detector nao processam em dobro."""
    bus = LocalBus(tmp_path)
    bus.publish("s", {"n": 1})
    primeiro = list(bus.consume("s", "g", "c1", block_ms=0))
    segundo = list(bus.consume("s", "g", "c2", block_ms=0))
    assert len(primeiro) == 1
    assert segundo == []


def test_ordem_de_publicacao_e_preservada(tmp_path):
    bus = LocalBus(tmp_path)
    for i in range(5):
        bus.publish("s", {"n": i})
    vistos = []
    for _ in range(5):
        vistos += [m.payload["n"] for m in bus.consume("s", "g", "c", block_ms=0)]
    assert vistos == [0, 1, 2, 3, 4]


def test_consume_sem_mensagem_retorna_vazio(tmp_path):
    bus = LocalBus(tmp_path)
    assert list(bus.consume("vazio", "g", "c", block_ms=0)) == []


def test_mensagem_consumida_por_todos_e_varrida(tmp_path):
    """A fila em disco tem de esquecer o que ja passou por ela.

    Nada era apagado: cada `consume` relistava `sorted(glob("*.json"))` inteiro
    a cada 150 ms, entao o custo de um worker **ocioso** crescia com todo o
    trabalho que o sistema ja tinha feito.
    """
    bus = LocalBus(tmp_path, retention_s=0.0001)
    for i in range(4):
        bus.publish("s", {"n": i})
    for _ in range(4):
        for m in bus.consume("s", "g", "c1"):
            bus.ack("s", "g", m.id)

    time.sleep(0.01)
    assert bus._sweep("s") == 4
    assert list((tmp_path / "s").glob("*.json")) == []


def test_a_varredura_respeita_grupo_que_ainda_nao_concluiu(tmp_path):
    """Apagar cedo demais custaria a mensagem de um servico que esta fora do ar.

    So some o que **todos** os grupos existentes carimbaram como concluido.
    """
    bus = LocalBus(tmp_path, retention_s=0.0001)
    bus.publish("s", {"n": 1})
    for m in bus.consume("s", "a", "c1"):
        bus.ack("s", "a", m.id)
    bus._group_dir("s", "b")  # existe, mas nunca consumiu

    time.sleep(0.01)
    assert bus._sweep("s") == 0
    assert len(list((tmp_path / "s").glob("*.json"))) == 1

    # e o grupo atrasado ainda recebe a mensagem
    assert [m.payload["n"] for m in bus.consume("s", "b", "c2")] == [1]


def test_mensagem_entregue_mas_nao_concluida_nao_e_varrida(tmp_path):
    """Entregue != concluido: um worker que morreu no meio do handler nao pode
    ver a mensagem sumir debaixo dele."""
    bus = LocalBus(tmp_path, retention_s=0.0001)
    bus.publish("s", {"n": 1})
    for _m in bus.consume("s", "g", "c1"):
        pass  # entregue, sem ack -- como um processo que caiu

    time.sleep(0.01)
    assert bus._sweep("s") == 0


# ──────────────────────────────── storage ───────────────────────────────────


def test_grava_e_le_arquivo(tmp_path):
    st = LocalStorage(tmp_path / "blobs")
    src = tmp_path / "x.bin"
    src.write_bytes(b"conteudo")
    st.put_file("a/b/x.bin", src)
    assert st.exists("a/b/x.bin")
    assert st.size("a/b/x.bin") == 8
    dest = st.get_file("a/b/x.bin", tmp_path / "out.bin")
    assert dest.read_bytes() == b"conteudo"


def test_leitura_por_faixa_de_bytes(tmp_path):
    st = LocalStorage(tmp_path / "blobs")
    src = tmp_path / "x.bin"
    src.write_bytes(bytes(range(256)))
    st.put_file("x.bin", src)
    assert st.open_range("x.bin", 10, 5) == bytes(range(10, 15))


def test_chave_nao_pode_escapar_da_raiz(tmp_path):
    st = LocalStorage(tmp_path / "blobs")
    with pytest.raises(ValueError):
        st.put_file("../fora.bin", tmp_path / "x.bin")


# ────────────────────────────────── audio ───────────────────────────────────
#
# O que se cobre aqui e o **teto de memoria**, e nao so o resultado. A versao
# anterior punha o WAV inteiro na RAM tres vezes (bytes crus, cópia em float32,
# mistura para mono) para no fim entregar alguns milhares de numeros: um audio
# de 20 min custava ~370 MB de pico no preprocessador. O resultado estava
# certo; o custo e que nao.


def _wav(path, *, segundos: float, sr: int = 22050, canais: int = 1):
    import wave

    n = int(segundos * sr)
    sinal = (np.sin(np.arange(n * canais) / 40.0) * 20000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(canais)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(sinal.tobytes())
    return path


def _onda_de_referencia(data, n):
    """A implementacao ingenua, para conferir que o resultado nao mudou."""
    n = min(n, data.size)
    corte = (data.size // n) * n
    blocos = np.abs(data[:corte]).reshape(n, -1).max(axis=1)
    topo = float(blocos.max())
    return [round(float(v), 3) for v in (blocos / topo)]


@pytest.mark.parametrize("canais", [1, 2])
def test_a_onda_lida_em_blocos_e_igual_a_ingenua(tmp_path, canais):
    wav = _wav(tmp_path / "a.wav", segundos=12.0, canais=canais)
    data, sr = read_wav(wav)
    esperado = _onda_de_referencia(data, peaks_for(data.size / sr))

    assert waveform(data, peaks_for(data.size / sr)) == esperado
    onda, duracao = waveform_of(wav)
    assert onda == esperado
    assert duracao == pytest.approx(12.0, abs=0.01)


def test_a_onda_nao_carrega_o_arquivo_inteiro(tmp_path):
    """`waveform_of` tem de custar o mesmo com 10 s e com 10 min de audio.

    E o caminho do preprocessador: ele so quer a onda para o editor desenhar, e
    nao tem por que segurar o audio da partida na memoria para isso.
    """
    curto = _wav(tmp_path / "curto.wav", segundos=5.0)
    longo = _wav(tmp_path / "longo.wav", segundos=600.0)
    assert longo.stat().st_size > 20 * curto.stat().st_size

    tracemalloc.start()
    try:
        waveform_of(curto)
        _, pico_curto = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        waveform_of(longo)
        _, pico_longo = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # o arquivo longo tem 120x o tamanho do curto; o pico nao pode acompanhar.
    # A folga e generosa de proposito: o que se trava aqui e a ordem de
    # grandeza -- constante em vez de proporcional ao arquivo.
    assert pico_longo < pico_curto + 32 * 1024 * 1024, (
        f"pico subiu de {pico_curto/1e6:.1f} MB para {pico_longo/1e6:.1f} MB: "
        "a onda voltou a carregar o arquivo inteiro"
    )
    # e o tamanho do arquivo tambem nao pode virar o pico
    assert pico_longo < longo.stat().st_size


def test_read_wav_nao_multiplica_o_sinal_na_memoria(tmp_path):
    """Quem precisa do sinal inteiro (o rastreador de batidas) paga por **uma**
    copia dele, e nao por tres."""
    wav = _wav(tmp_path / "m.wav", segundos=300.0)
    tracemalloc.start()
    try:
        data, _sr = read_wav(wav)
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert pico < data.nbytes * 1.75, (
        f"pico {pico/1e6:.1f} MB para um sinal de {data.nbytes/1e6:.1f} MB"
    )


def test_wav_ilegivel_vira_onda_vazia_em_vez_de_erro(tmp_path):
    ruim = tmp_path / "ruim.wav"
    ruim.write_bytes(b"nao sou um wav")
    assert waveform_of(ruim) == ([], 0.0)


# ───────────────────────────────── visao ────────────────────────────────────


def test_pulso_precisa_cruzar_o_limiar_de_subida():
    t = [i * 0.1 for i in range(20)]
    v = [0.0] * 5 + [1.0] * 5 + [0.0] * 10
    p = find_pulses(t, v, rise=0.5, fall=0.2)
    assert len(p) == 1
    assert p[0].start == pytest.approx(0.5)


def test_histerese_evita_contar_piscada_como_dois_eventos():
    """O icone oscila; com histerese isso continua sendo um evento so."""
    t = [i * 0.1 for i in range(20)]
    v = [0.0] * 3 + [1.0, 0.3, 1.0, 0.3, 1.0] + [0.0] * 12
    assert len(find_pulses(t, v, rise=0.5, fall=0.2)) == 1


def test_min_gap_funde_pulsos_colados():
    t = [i * 0.1 for i in range(30)]
    v = [0.0] * 3 + [1.0] * 2 + [0.0] * 2 + [1.0] * 2 + [0.0] * 21
    assert len(find_pulses(t, v, rise=0.5, fall=0.2)) == 2
    assert len(find_pulses(t, v, rise=0.5, fall=0.2, min_gap=1.0)) == 1


def test_pulso_curto_demais_e_descartado():
    t = [i * 0.1 for i in range(20)]
    v = [0.0] * 5 + [1.0] + [0.0] * 14
    assert find_pulses(t, v, rise=0.5, fall=0.2, min_duration=0.5) == []


def test_pulso_aberto_no_fim_do_video_e_fechado():
    t = [i * 0.1 for i in range(10)]
    v = [0.0] * 3 + [1.0] * 7
    p = find_pulses(t, v, rise=0.5, fall=0.2)
    assert len(p) == 1


def test_hsv_ratio_conta_so_a_faixa_pedida():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:5, :, 2] = 255  # metade vermelha pura
    ranges = [{"lo": [0, 120, 90], "hi": [10, 255, 255]}]
    assert hsv_ratio(img, ranges) == pytest.approx(0.5)
    assert hsv_ratio(img, []) == 0.0


def test_border_mask_cobre_so_a_moldura():
    m = border_mask((100, 100), 0.1)
    assert m[0, 0] and m[-1, -1]
    assert not m[50, 50]


def test_banco_de_templates_vazio_nao_quebra(tmp_path):
    bank = TemplateBank.from_dir(tmp_path / "nao_existe")
    assert not bank
    assert bank.best_match(np.zeros((10, 10, 3), np.uint8)) == (None, 0.0)


def test_template_encontra_a_si_mesmo(tmp_path):
    import cv2

    img = np.zeros((60, 60, 3), np.uint8)
    cv2.circle(img, (30, 30), 15, (255, 255, 255), -1)
    tdir = tmp_path / "t"
    tdir.mkdir()
    cv2.imwrite(str(tdir / "alvo.png"), img[15:45, 15:45])
    name, score = TemplateBank.from_dir(tdir).best_match(img)
    assert name == "alvo"
    assert score > 0.9


# ── glifos: a marca de um ícone, sem posição, escala nem polaridade ────────


def _seta(side: int) -> np.ndarray:
    """Uma marca assimétrica, para giro e espelho não passarem por ela."""
    import cv2

    m = np.zeros((side, side), np.uint8)
    pts = np.array([[side // 2, 0], [side - 1, side // 2], [int(side * 0.68), side // 2],
                    [int(side * 0.68), side - 1], [int(side * 0.32), side - 1],
                    [int(side * 0.32), side // 2], [0, side // 2]], np.int32)
    cv2.fillPoly(m, [pts], 255)
    return m


def test_glifo_normalizado_ignora_tamanho_e_posicao():
    """O mesmo desenho, pequeno num canto e grande no meio, tem de sair igual --
    é isso que dispensa o casamento em várias escalas."""
    pequeno = np.zeros((80, 80), np.uint8)
    pequeno[5:25, 5:25] = _seta(20)
    grande = np.zeros((80, 80), np.uint8)
    grande[20:76, 12:68] = _seta(56)

    a, b = normalized_glyph(pequeno), normalized_glyph(grande)
    assert a is not None and b is not None
    assert a.shape == b.shape == (GLYPH_SIDE, GLYPH_SIDE)
    assert float(np.mean((a > 127) == (b > 127))) > 0.93


def test_glifo_de_marca_pequena_demais_nao_existe():
    assert normalized_glyph(np.zeros((40, 40), np.uint8)) is None


def _grava_icone(pasta, heroi: str, nome: str, side: int = 128) -> None:
    import cv2

    (pasta / heroi).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pasta / heroi / f"{nome}.png"), 255 - _seta(side))


def test_banco_de_icones_vazio_nao_quebra(tmp_path):
    bank = IconBank.from_dir(tmp_path / "nao_existe")
    assert not bank
    assert bank.best_match(np.zeros((GLYPH_SIDE, GLYPH_SIDE), np.uint8)) == (None, 0.0)


def test_o_banco_reconhece_a_marca_nas_duas_polaridades(tmp_path):
    """Na HUD o mesmo ícone aparece preto sobre disco branco (ultimate) e branco
    sobre caixa escura (habilidade comum). Um molde só serve para os dois --
    é a marca que é comparada, não os pixels."""
    import cv2

    _grava_icone(tmp_path, "sample", "seta")
    bank = IconBank.from_dir(tmp_path)
    assert len(bank) == 1

    # ultimate: disco branco, marca preta
    disco = np.full((60, 60, 3), 20, np.uint8)
    cv2.circle(disco, (30, 30), 26, (250, 250, 250), -1)
    marca = _seta(30)
    disco[15:45, 15:45][marca > 0] = (15, 15, 15)
    key, score = bank.best_match(glyph_in_disc(disco))
    assert key == "sample/seta" and score > 0.85

    # habilidade comum: caixa escura, marca clara
    caixa = np.full((40, 40, 3), 50, np.uint8)
    marca = _seta(30)
    caixa[5:35, 5:35][marca > 0] = (240, 240, 240)
    key, score = bank.best_match(glyph_on_dark(caixa))
    assert key == "sample/seta" and score > 0.85


def test_a_chave_do_icone_traz_heroi_e_habilidade(tmp_path):
    _grava_icone(tmp_path, "orisa", "energy_javelin")
    _grava_icone(tmp_path, "domina", "panopticon")
    assert set(IconBank.from_dir(tmp_path).keys) == {
        "orisa/energy_javelin", "domina/panopticon",
    }


def test_a_mira_dentro_da_roi_sai_da_geometria_da_roi():
    """A ROI de eliminações é deslocada para cima, então a mira -- que é o centro
    da TELA -- não cai no centro dela. Derivar isso da geometria evita um
    segundo número no profile para esquecer de corrigir junto."""
    from owcore.models import RoiSpec

    roi = RoiSpec(name="kills", x=0.42, y=0.40, w=0.16, h=0.18)
    x, y = roi.relative(0.5, 0.5)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5556, abs=1e-3)


def test_recortes_saem_com_cor_canonica(tmp_path):
    """Regressao de um bug que so aparecia entre maquinas.

    Os detectores decidem por saturacao, e a matriz YUV->RGB muda exatamente a
    saturacao. Com o recorte marcado como BT.709, o mesmo arquivo era lido com
    saturacao 231 no host e 205 dentro do container -- e o detector achava 20
    eliminacoes num lugar e 10 no outro, com o mesmo codigo. Marcar BT.601, que
    e o que os decodificadores assumem para quadros pequenos, faz os dois lerem
    igual. Este teste trava a marcacao.
    """
    import json
    import subprocess

    from owcore.config import get_settings
    from owcore.ffmpeg import extract_rois
    from owcore.models import RoiSpec

    src = tmp_path / "src.mp4"
    subprocess.run(
        [get_settings().ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x180:rate=10:duration=1",
         "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    roi = RoiSpec(name="r", x=0.25, y=0.25, w=0.5, h=0.5, fps=5, width_px=160)
    out = extract_rois(src, [roi], tmp_path / "out")

    probe = json.loads(subprocess.run(
        [get_settings().ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_streams", "-print_format", "json", str(out["r"])],
        capture_output=True, text=True, check=True,
    ).stdout)["streams"][0]

    # `color_space` (a matriz) e `color_range` sao os dois que mudam a conversao
    # YUV->RGB, e portanto a saturacao. Primaries e transfer o ffmpeg omite
    # quando nao acrescentam nada, entao nao ha o que afirmar sobre eles.
    assert probe.get("color_space") == "smpte170m"
    assert probe.get("color_range") == "tv"


def test_o_recorte_diz_por_onde_anda(tmp_path):
    """O recorte é ~3/4 do tempo de uma análise. Sem ele avisando por onde
    anda, a tela fica no mesmo número por minutos numa gravação de partida — e
    parada é como o usuário lê travada."""
    import subprocess

    from owcore.config import get_settings
    from owcore.ffmpeg import extract_rois
    from owcore.models import RoiSpec

    # pesado o bastante para o ffmpeg ter o que contar: ele so fala a cada meio
    # segundo de relogio, e um recorte que acaba antes disso avisaria uma vez so
    src = tmp_path / "src.mp4"
    subprocess.run(
        [get_settings().ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=1280x720:rate=30:duration=40",
         "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    rois = [
        RoiSpec(name=f"r{i}", x=0.0, y=0.0, w=1.0, h=1.0, fps=30, width_px=320)
        for i in range(5)
    ]

    vistos: list[float] = []
    out = extract_rois(src, rois, tmp_path / "out", on_progress=vistos.append)

    assert out["r0"].exists(), "o recorte tem de sair igual, com ou sem relator"
    assert vistos == sorted(vistos), "a barra nao pode andar para tras"
    assert all(0.0 <= v <= 1.0 for v in vistos), f"fracao fora de 0..1: {vistos}"
    # o que importa nao e quantas vezes avisou, e sim ter avisado ENQUANTO
    # trabalhava: um relator que so fala no fim nao move barra nenhuma
    assert any(v < 0.99 for v in vistos), f"so avisou no fim: {vistos}"
    assert vistos[-1] > 0.5, f"parou cedo demais em {vistos[-1]:.2f}"


# ── o ícone da HUD contra "qualquer coisa vermelha" ────────────────────────

MAGENTA = (115, 23, 235)  # BGR do magenta da caveira
FAIXA = [{"lo": [156, 185, 150], "hi": [178, 255, 255]}]


def _quadro(desenhar) -> np.ndarray:
    img = np.full((64, 102, 3), (70, 60, 55), np.uint8)  # cena neutra
    desenhar(img)
    return img


def _achar(img):
    from owcore.vision import find_icon

    return find_icon(img, FAIXA, min_area_frac=0.04, max_offset=0.30,
                     aspect_range=(0.55, 1.9))


def test_respingo_vermelho_pequeno_nao_e_icone():
    """A queixa que motivou o filtro de tamanho: um borrão de 20 pixels no meio
    da tela não é uma eliminação."""
    import cv2

    img = _quadro(lambda i: cv2.circle(i, (51, 32), 3, MAGENTA, -1))
    assert _achar(img) is None


def test_icone_do_tamanho_certo_e_encontrado():
    import cv2

    img = _quadro(lambda i: cv2.circle(i, (51, 32), 14, MAGENTA, -1))
    blob = _achar(img)
    assert blob is not None
    assert 0.04 < blob.area_frac < 0.30
    assert abs(blob.offset_x) < 0.1 and abs(blob.offset_y) < 0.1


def test_arco_de_dano_e_rejeitado_por_forma_e_posicao():
    """O indicador direcional de dano é largo, achatado e fica acima da mira."""
    import cv2

    img = _quadro(lambda i: cv2.ellipse(i, (51, 8), (34, 5), 0, 0, 360, MAGENTA, -1))
    assert _achar(img) is None


def test_buracos_internos_distinguem_caveira_de_mancha():
    """As órbitas são o que separa uma caveira de um borrão do mesmo tamanho."""
    import cv2

    solido = _quadro(lambda i: cv2.circle(i, (51, 32), 14, MAGENTA, -1))
    def com_orbitas(i):
        cv2.circle(i, (51, 32), 14, MAGENTA, -1)
        cv2.circle(i, (46, 29), 3, (20, 20, 20), -1)
        cv2.circle(i, (56, 29), 3, (20, 20, 20), -1)

    caveira = _quadro(com_orbitas)
    assert _achar(solido).hole_ratio == 0.0
    assert _achar(caveira).hole_ratio > 0.05


# ── schema que evolui sem perder o que ja estava la ─────────────────────────


def test_coluna_nova_chega_a_um_banco_que_ja_existia(isolated):
    """`create_all` ignora tabela que ja existe -- e isso quase custou caro.

    Quem ja tinha rodado o sistema tinha a tabela `renders` sem a coluna das
    montagens manuais. Sem reconciliar, a primeira montagem estouraria com
    "column renders.timelines does not exist", e a saida seria apagar o banco
    junto com as partidas ja analisadas.
    """
    from sqlalchemy import inspect, text

    from owcore.db import engine, init_db, session
    from owcore.models import Job, Render

    with session() as s:
        s.add(Job(id="j1", video_key="k", video_name="v.mp4"))
        s.add(Render(id="r1", job_id="j1", stage="na fila"))

    # volta o banco ao estado de antes desta funcionalidade
    eng = engine()
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE renders DROP COLUMN timelines"))
        conn.execute(text("DROP TABLE tracks"))
    assert "timelines" not in {
        c["name"] for c in inspect(eng).get_columns("renders")
    }

    init_db()

    colunas = {c["name"] for c in inspect(eng).get_columns("renders")}
    assert "timelines" in colunas, "a coluna nova nao chegou ao banco antigo"
    assert "tracks" in inspect(eng).get_table_names(), "a tabela nova nao foi criada"

    with session() as s:
        pedido = s.get(Render, "r1")
        # o pedido antigo continua inteiro, e a coluna nova le como vazia
        assert pedido.stage == "na fila"
        assert pedido.timelines == []


def test_coluna_nova_nao_deixa_as_linhas_antigas_com_NULL(isolated):
    """Backfill tambem nos tipos simples, e nao so no JSON.

    Uma coluna nova entra anulavel -- pôr NOT NULL numa tabela com linhas
    exigiria reescreve-la. Se as linhas antigas ficam com NULL, quem consome
    quebra longe daqui: foi `round(job.fps, 3)` derrubando a listagem inteira de
    partidas depois de `fps` entrar no modelo.
    """
    from sqlalchemy import text

    from owcore.db import engine, init_db, session
    from owcore.models import Job

    with session() as s:
        s.add(Job(id="j2", video_key="k", video_name="v.mp4"))

    eng = engine()
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE jobs DROP COLUMN fps"))
        conn.execute(text("ALTER TABLE jobs DROP COLUMN proxy_key"))

    init_db()

    with session() as s:
        job = s.get(Job, "j2")
        assert job.fps == 0.0, "a linha antiga ficou com NULL num Float"
        assert job.proxy_key == "", "a linha antiga ficou com NULL num String"


def test_repara_o_NULL_que_um_boot_anterior_deixou(isolated):
    """A coluna ja existe, mas com NULL onde o modelo promete um valor.

    Acontece quando ela foi criada por uma versao do reconciliador que ainda nao
    preenchia aquele tipo. Encontrar a coluna pronta e ir embora deixaria a base
    com NULL para sempre.
    """
    from sqlalchemy import text

    from owcore.db import engine, init_db, session
    from owcore.models import Job

    with session() as s:
        s.add(Job(id="j3", video_key="k", video_name="v.mp4"))

    eng = engine()
    with eng.begin() as conn:
        # o estado exato que um boot antigo deixaria: a coluna existe, anulavel
        # (pôr NOT NULL numa tabela com linhas exigiria reescreve-la), e sem
        # ninguem ter preenchido as linhas de antes
        conn.execute(text("ALTER TABLE jobs DROP COLUMN fps"))
        conn.execute(text("ALTER TABLE jobs ADD COLUMN fps FLOAT"))

    init_db()

    with session() as s:
        assert s.get(Job, "j3").fps == 0.0


def test_reconciliar_e_idempotente(isolated):
    """Todo worker chama `init_db` no boot; rodar de novo nao pode mexer em nada."""
    from owcore.db import _reconcile_columns, engine, init_db

    init_db()
    with engine().begin() as conn:
        assert _reconcile_columns(conn) == []
