# OW Editor — backend

Microsserviços Python que recebem a gravação de uma partida, separam os
momentos importantes e — quando o usuário escolhe — montam os vídeos. Roda
sozinho: o frontend é opcional, e a API é navegável em
<http://localhost:8000/docs>.

> Todos os comandos abaixo assumem que você está **na raiz deste repositório**.

**A documentação do sistema mora aqui:**

| Documento | O que tem |
|---|---|
| [`docs/PRODUTO.md`](docs/PRODUTO.md) | o produto inteiro: as duas fases, os dois caminhos para virar vídeo, e o que está verificado e o que não está |
| [`docs/PLAN.md`](docs/PLAN.md) | arquitetura, detecção, regras dos melhores momentos, montagem manual e o contrato REST |
| [`docs/V2.md`](docs/V2.md) | o editor completo, em oito fases — **todas feitas**, cada uma com o que ficou de fora e por quê |

O app Flutter é um repositório à parte, e o `docker-compose.yml` (que orquestra
os dois) vive com ele. Em disco os dois ficam lado a lado — `ow_editor/backend`
e `ow_editor/frontend` —, e é assim que o compose os encontra.

---

## Rodando

### Sem Docker (só Python + ffmpeg)

```bash
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt

python tools/dev.py
```

Sobe os nove processos usando fila em disco, storage em pasta e SQLite —
nenhum servidor externo. Cada microsserviço continua sendo um processo à
parte falando pelo barramento; o que muda é só a implementação por trás das
interfaces.

### Com Docker

O `docker-compose.yml` fica na raiz do repositório, porque também monta o app
Flutter no gateway:

```bash
cd .. && docker compose up --build
```

---

## Testando

```bash
# gera o vídeo sintético que os testes de precisão usam (uma vez só)
python tools/make_sample.py --out data/sample/match.mp4 \
    --music data/sample/music.wav --ult-templates data/sample/ult_templates

pytest tests/ -q
```

São 223 testes em cinco camadas:

| Arquivo | O que cobre |
|---|---|
| `test_rules.py` | as regras de highlight e o encaixe da montagem na janela de música — função pura, sem tocar em vídeo |
| `test_infra.py` | barramento (fan-out entre grupos, competição dentro do grupo), storage, primitivas de visão, marcação de cor dos recortes |
| `test_detectors.py` | precisão de cada detector contra o gabarito do vídeo sintético, incluindo as duas habilidades do rodapé não se confundirem |
| `test_pipeline.py` | as duas fases atravessando todos os serviços — é o que verifica os *contratos* das mensagens, mais músicas diferentes por vídeo, a janela de música, o áudio original e o zip dos cortes |
| `test_timeline.py` | a montagem manual e o editor da V2: a matemática da linha do tempo (buraco vira preto, corte aparado não move o vizinho), camadas, efeitos e texto conferidos **no pixel** do mp4 que saiu, a exportação (dimensão, fps e trecho por `ffprobe`; `cover` contra `contain`), o reaproveitamento da imagem ao trocar de música, as várias montagens nomeadas com histórico e predefinições, mais as entregas com `Range` que o app usa para tocar a música e mostrar o preview |

Os testes que dependem do vídeo sintético se auto-pulam se ele não existir,
dizendo como gerá-lo.

---

## Arquitetura

O sistema tem **duas fases**, e elas não se misturam.

**Fase 1 — análise.** Roda uma vez por gravação, sozinha, e termina em `ready`:

```
gateway ──▶ preprocessor ──▶ detector_kills    ──┐
 (API)      (1 decode,       detector_survival   │
            N recortes)      detector_ults       ├──▶ planner ──▶ propostas
                             detector_banner   ──┘   (regras)
```

**Fase 2 — geração.** Disparada pelo usuário, quantas vezes ele quiser:

```
gateway ──▶ beats ──▶ editor ──▶ clipes
 (API)     (ritmo de   (cortes,
        cada música)    trilha)
```

Barramento com *consumer groups*: o preprocessor publica uma vez e cada
detector recebe a sua mensagem. O planner só age quando todos os esperados
reportaram — e a transição para `ready` é reivindicada atomicamente no banco,
para vários avisos chegando juntos não montarem a lista em dobro.

O `thumbs` extrai um quadro de cada momento para a barra lateral do editor.
Escuta o fim do planejamento num *consumer group* próprio — recebe o mesmo aviso
que o planejador e trabalha em paralelo, sem segurar o job em `ready`. A chave de
cada quadro sai do instante, então não há tabela nem coluna para eles.

Um detector que falha **não derruba o job**: ele registra o erro no relatório
e libera o planner, que propõe com o que os outros acharam.

O `beats` **não é um detector**: ele não olha a partida e não entra na análise.
Na fase 2 ele analisa cada música do pedido — uma grade de batidas por vídeo
escolhido — e só então o editor corta. Vídeo sem música pula essa etapa e sai
com o **áudio original** da partida.

O mesmo processo roda um **segundo laço**, em `ow.track`: a música que o usuário
sobe para montar à mão. Ela chega antes de existir vídeo nenhum — é ouvindo-a,
com as batidas e a forma de onda na tela, que ele decide onde cada corte cai —,
e volta com duração, BPM, batidas e um envelope reduzido a ~40 pontos por
segundo, que é o que o app desenha sem precisar baixar o áudio para isso. São
dois laços porque são dois momentos do fluxo; a análise por trás é a mesma, e
não valia um serviço a mais no compose.

### Os dois modos

Cada dependência de infra tem duas implementações atrás da mesma interface,
escolhidas por `OW_MODE`:

| | `local` (padrão) | `docker` |
|---|---|---|
| Fila | arquivos em `data/bus` | Redis Streams |
| Storage | pasta `data/blobs` | MinIO (S3) |
| Banco | SQLite | PostgreSQL |

### Estrutura

```
packages/owcore/   núcleo compartilhado, instalado em cada serviço
  bus.py           barramento (Redis Streams | fila em disco)
  storage.py       blobs (MinIO/S3 | pasta)
  db.py            SQLAlchemy (Postgres | SQLite)
  models.py        domínio + tabelas + mensagens do barramento
  rules.py         regras de highlight (função pura)
  timeline.py      montagem manual: blocos → pedaços a cortar (função pura)
  vision.py        primitivas de visão computacional
  ffmpeg.py        recorte, corte, concatenação, trilha
  audio.py         leitura de WAV e forma de onda (música e partida)
  compose.py       linha do tempo em camadas -> grafo de filtros (função pura)
  textfx.py        texto -> `drawtext`, com o escape que o filtergraph exige
  fonts.py         onde está a fonte; falha alto quando não há nenhuma
  detector.py      base dos microsserviços detectores
  worker.py        laço de consumo, ack, encerramento limpo
services/          um diretório por microsserviço
config/profiles/   posições e cores da HUD
templates/         ícones de referência (opcionais) — veja templates/README.md
tools/             gerador de exemplo, calibração, runner local
tests/
```

---

## Calibração — leia antes de usar com gameplay real

Os detectores procuram elementos da HUD em posições e cores definidas em
`config/profiles/ow2_default.json`. O perfil que vem no repositório foi
calibrado contra gameplay real de Overwatch 2 em 16:9 (medido em 360p; a HUD do
OW2 escala com a resolução, então os valores normalizados valem de 360p a 4K).

Ainda assim, **idioma, modo daltônico, proporção fora de 16:9 e patches do jogo
mudam posições e cores**. Se o sistema não achar nada na sua gravação, comece
por aqui.

> **Nota sobre cor.** O preprocessor grava os recortes com tags de cor BT.601
> explícitas, e isso não é cosmético: os detectores decidem por saturação, e a
> matriz YUV→RGB muda exatamente a saturação. Sem essas tags, o mesmo arquivo
> era lido com saturação 231 na máquina host e 205 dentro do container — e o
> detector achava 20 eliminações num lugar e 10 no outro, com o mesmo código.
> `tests/test_infra.py` trava essa marcação.

```bash
# 1. As regiões estão no lugar certo? Desenha os retângulos sobre quadros reais.
python tools/calibrate.py preview --video partida.mp4 --at 30 90 150

# 2. Que limiar usar? Mede a região quadro a quadro e sugere os números.
python tools/calibrate.py scan --video partida.mp4 --roi kills
```

Depois copie o perfil, ajuste e rode com `OW_PROFILE=meu_perfil`.

---

## Configuração

Tudo por variável de ambiente com prefixo `OW_` (veja `.env.example`):

| Variável | Padrão | Para quê |
|---|---|---|
| `OW_MODE` | `local` | `local` ou `docker` |
| `OW_PROFILE` | `ow2_default` | perfil da HUD |
| `OW_DATA_DIR` | `./data` | uploads, recortes, clipes, fila |
| `OW_WEB_DIR` | `../frontend/build/web` | app Flutter compilado, se houver |
| `OW_DATABASE_URL` | derivado do modo | SQLite ou Postgres |
| `OW_REDIS_URL` | `redis://localhost:6379/0` | barramento no modo docker |
| `OW_S3_*` | MinIO local | storage no modo docker |
| `OW_DETECTOR_TIMEOUT_S` | `900` | quando o planner desiste de um detector mudo |
