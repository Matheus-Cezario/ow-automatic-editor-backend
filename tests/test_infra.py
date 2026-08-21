"""Barramento, storage e visao -- as pecas de infra que os microsservicos
assumem que funcionam."""

from __future__ import annotations

import numpy as np
import pytest

from owcore.bus import LocalBus
from owcore.storage import LocalStorage
from owcore.vision import TemplateBank, border_mask, find_pulses, hsv_ratio


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
        s.add(Render(id="r1", job_id="j1", selections=[{"proposal_id": "p"}]))

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
        assert pedido.selections == [{"proposal_id": "p"}]
        assert pedido.timelines == []


def test_reconciliar_e_idempotente(isolated):
    """Todo worker chama `init_db` no boot; rodar de novo nao pode mexer em nada."""
    from owcore.db import _reconcile_columns, engine, init_db

    init_db()
    with engine().begin() as conn:
        assert _reconcile_columns(conn) == []
