# OW Editor — Plano do Projeto

> **Onde mora cada coisa.** No Git são **dois repositórios**: este, o do
> backend, fica clonado dentro do repositório do app, na pasta `backend/` — que
> o de fora ignora. Os caminhos deste documento sem prefixo (`services/`,
> `packages/`) são deste repositório; os que começam com `frontend/` estão no
> outro, e o `docker-compose.yml` também.

**Editor de vídeo com detecção automática de eventos de Overwatch 2.** Recebe a
gravação de uma partida, assiste a ela e anota o que aconteceu; o usuário monta
o vídeo na linha do tempo — ouvindo a música no próprio app e pondo cada momento
no ponto dela que quiser, com a duração que quiser.

> **Descontinuado na Fase 11:** a geração automática. O sistema aplicava regras
> sobre os eventos e oferecia uma lista de vídeos prontos para escolher, cada um
> com a sua música. Esse caminho saiu inteiro — `services/planner` deixou de
> propor, `owcore/rules.py` ficou só com o cruzamento entre detectores, e a
> tabela `proposals`, o campo `selections` e a tela de geração do app foram
> removidos. O que restou é o que sempre foi o diferencial: um editor que sabe
> o que aconteceu no vídeo.

## 1. Princípios

1. **Tudo gratuito / open-source.** FastAPI, Redis, MinIO, PostgreSQL, OpenCV,
   ffmpeg, librosa, Flutter. Nenhum serviço pago, nenhuma API externa.
2. **Roda sem Docker também.** Cada dependência de infra tem duas
   implementações atrás da mesma interface, escolhidas por variável de ambiente:

   | Recurso | Modo `local` (sem Docker) | Modo `docker` |
   |---|---|---|
   | Fila / eventos | fila em disco (`LocalBus`) | Redis Streams |
   | Armazenamento  | pasta `data/` (`LocalStorage`) | MinIO (S3) |
   | Banco          | SQLite | PostgreSQL |

3. **Cada microsserviço recebe o mínimo de pixels possível.** O `preprocessor`
   faz *uma única* decodificação do vídeo e emite recortes (ROIs) pequenos, em
   baixo FPS, um por detector. Um detector nunca vê o vídeo inteiro.
4. **Calibrável.** Posições da HUD, cores e limiares ficam num *profile* JSON,
   não no código, porque mudam com resolução, idioma, modo daltônico e patch.
5. **O sistema detecta; quem monta é o usuário.** A análise entrega os
   *instantes* de cada acontecimento (eliminação, tiro na cabeça, morte por
   habilidade, dardo, pedrada) e nada mais. O caminho é a linha do tempo: o
   usuário ouve a música, vê as batidas e a forma de onda, e põe cada corte onde
   ele soa melhor. Nenhuma regra sabe onde é o refrão da música de alguém — e é
   por isso que as propostas saíram.

   Corolário operacional: **todo tipo de evento detectado tem de chegar à
   prateleira do editor.** Um detector cujo resultado não vira bloco é trabalho
   de CPU jogado fora. A lista vive em dois lugares que precisam concordar —
   `THUMB_KINDS` (`owcore/models.py`, de onde se extrai miniatura) e
   `_momentosUteis` (`frontend/lib/screens/timeline_screen.dart`, o que a tela
   mostra). Divergir dá cartão sem quadro, ou momento que não existe.
6. **Análise e geração são fases separadas.** A análise é cara e o resultado não
   muda — os momentos da partida são os que são. A escolha do que virar vídeo é
   barata, pessoal e mutável. Juntar as duas obrigaria a reanalisar o vídeo só
   para trocar de música.

## 2. Arquitetura

### Fase 1 — análise (roda uma vez, sozinha)

```
                 ┌────────────┐
  Flutter  ─────▶│  gateway   │  upload da gravação (só o vídeo)
  (mobile/web)   └─────┬──────┘
                       │ publica JobCreated
                       ▼
                 ┌────────────┐   1 decode → N recortes pequenos + áudio
                 │preprocessor│
                 └─────┬──────┘
     RoiReady(kills) │ (survival)   (ults)      (banner)    (killfeed)
       ┌───────────┬──┴────────┬────────────┬────────────┐
       ▼           ▼           ▼            ▼            ▼
 ┌──────────┐┌──────────┐┌──────────┐ ┌──────────┐ ┌──────────┐
 │detector_ ││detector_ ││detector_ │ │detector_ │ │detector_ │
 │kills     ││survival  ││ults      │ │banner    │ │killfeed  │
 └────┬─────┘└────┬─────┘└────┬─────┘ └────┬─────┘ └────┬─────┘
      │ eventos   │ eventos   │ eventos    │ eventos    │ eventos
      └───────────┴─────┬─────┴────────────┴────────────┘
                                  ▼
                          ┌───────────────┐
                          │    planner    │ espera todos, cruza o que
                          │  (fecha e     │ nenhum detector vê sozinho,
                          │   cruza)      │ nenhum pixel
                          └───────┬───────┘
                                  ▼
                    eventos cruzados + job em `ready`
```

O job para em `ready` e **espera**. Nenhuma música passou por aqui.

O `planner` já montava aqui a lista de propostas. Não monta mais; o que ele
ainda faz é o que ninguém mais pode: é o ponto onde todos os detectores se
encontram, e portanto o único lugar de onde se enxerga uma **ultimate anulada**
(ultimate inimiga + eliminação logo em seguida). Esses eventos derivados são
**gravados** como qualquer outro — antes nasciam dentro do gerador de propostas
e morriam ali, e o editor nunca chegava a vê-los.

### Fase 1½ — a biblioteca do editor (música, clipe, imagem)

```
  Flutter ──▶ gateway   POST /api/jobs/{id}/tracks   (só o arquivo de áudio)
                │ publica TrackUploaded
                ▼
          ┌───────────┐  duração, BPM, batidas e forma de onda reduzida
          │   beats   │  (o mesmo processo, num segundo laço)
          └─────┬─────┘
                ▼
     a música volta pronta para o app **desenhar** e **tocar**
```

A música sobe **antes** de existir vídeo nenhum, e essa é a inversão que a
montagem manual exige: não dá para decidir que um corte entra "na virada do
refrão" sem ouvir o refrão, nem encaixá-lo na batida sem saber onde as batidas
estão. Ela é do job, e não de um pedido — a mesma trilha serve a quantas
montagens o usuário quiser, sem subir de novo.

### Fase 2 — geração (sob demanda, quantas vezes o usuário quiser)

```
  Flutter ──▶ gateway   POST /api/jobs/{id}/renders
  (as montagens │       form: timelines JSON
   da régua)    │ publica RenderRequested → ow.render.ready
                ▼
          ┌───────────┐  corta e junta exatamente o que foi montado;
          │  editor   │  buraco vira preto, e sem trilha na régua o
          └─────┬─────┘  vídeo sai com o áudio original
                ▼
          clipes finais no storage, amarrados ao pedido
```

O pedido vai **direto** ao editor. Havia um passo do `beats` no meio, para
analisar a música de cada proposta escolhida; sem propostas, a única música que
existe é a da biblioteca, e ela já subiu analisada na fase 1½.

Barramento: streams com *consumer groups* (Redis Streams em produção,
equivalente em disco no modo local). Estado dos jobs: banco relacional via
SQLAlchemy 2.0, mesma model em SQLite e Postgres.

### Modelo de dados

| Tabela | O que guarda |
|---|---|
| `jobs` | a gravação, o andamento da **análise** (`pending → preprocessing → detecting → ready`) e a montagem em andamento (`draft`) |
| `events` | o que cada detector achou, com o instante — **inclusive** o que só se vê cruzando dois detectores (`ULT_NEGATED`), gravado pelo `planner` |
| `tracks` | uma música enviada para a partida, com duração, BPM, batidas e forma de onda — é o que a tela de montagem toca e desenha |
| `renders` | um **pedido** de geração: as montagens e o andamento (`pending → rendering → done`) |
| `clips` | o vídeo gerado, ligado ao pedido |
| `montages` | uma montagem **nomeada** de uma partida, com o histórico dela em `montage_versions`. Uma partida rende mais de um vídeo |
| `presets` | o **jeito** de montar, guardado para a próxima partida. Não pertence a job nenhum, de propósito |

> **Removidas na Fase 11:** a tabela `proposals` e a coluna `renders.selections`.
> O reconciliador de schema (`owcore/db.py`) acrescenta coluna, mas nunca remove
> — num banco que já rodou o sistema elas continuam lá, órfãs e ignoradas.
> Nenhuma migração é necessária.

O fluxo é repetível porque gerar um vídeo não consome nada: os `events` da
partida ficam onde estão, e o mesmo instante entra em quantas montagens se
quiser, com músicas diferentes.

A montagem não tem tabela própria: ela é uma lista de blocos guardada no
próprio pedido (`renders.timelines`). Cada bloco diz **o que** entra
(`start_s` + `duration_s`, na gravação) e **onde** entra (`at_s`, no vídeo que
vai sair) — duas coisas independentes, e é essa independência que deixa o mesmo
momento aparecer duas vezes, em pontos diferentes da música e com durações
diferentes.

Junto dos blocos vai o `export`: tamanho, fps, qualidade, enquadramento, trecho
e marca d'água. Ele fica **fora** das camadas de propósito — a mesma montagem
vira um 16:9 e um 9:16 sem que um bloco se mexa. E é ele, sozinho, que decide o
caminho de renderização: uma saída fora do padrão não existe no corte-e-emenda
da V1, só no grafo de filtros.

## 3. Detecção — como cada evento é reconhecido

| Detector | ROI enviada | Técnica | Evento emitido |
|---|---|---|---|
| `detector_kills` | ~16%×18% da tela em volta da mira, 12 fps | máscara HSV do magenta da HUD + **filtro de forma, posição e tamanho**: o blob tem de ser compacto, quase quadrado, centrado na mira e ocupar de 5% a 14% da região | `KILL` |
| `detector_kills` (2ª leitura) | a mesma ROI | o marcador de acerto **crítico** é vermelho (o normal é branco): quatro traços em X na mira. A decisão é por forma, não por cor — as quatro diagonais pintadas e as quatro direções retas limpas. É esse segundo teste que impede a caveira de eliminação, também vermelha e também na mira, de virar headshot | `HEADSHOT` |
| `detector_survival` | tira da barra de vida (canto inf. esq.), 6 fps | lê a fração preenchida pela *alternância* dos tracinhos; vida baixa sustentada, e vida zerando = interrupção | `LOW_HP`, `ESCAPE`, `DEATH` |
| `detector_ults` | **botão de ultimate** do rodapé, 5 fps | carregada, a ultimate do jogador é um disco branco com o ícone do herói e um anel ciano em volta; usá-la apaga as duas coisas. O evento é a **borda de descida**, e o ícone do disco diz de quem era a ultimate | `ULT_USED` (`side="self"`) |
| `detector_ults` | killfeed (canto sup. direito), 5 fps | `matchTemplate` multiescala contra ícones de ultimate fornecidos pelo usuário | `ULT_USED` (`side="enemy"`) |
| `detector_killfeed` | a mesma tira do killfeed, 5 fps | acha a linha pelas **cores dos times** (placa ciano à esquerda de uma vermelha = eliminação nossa) e lê o ícone no vão entre elas, comparando com `templates/abilities/`. Nenhum texto, nenhum OCR | `ABILITY_KILL` |
| `detector_banner` | faixa de avisos do rodapé, 4 fps | acha **todas** as faixas do quadro (uma máscara por cor, porque a HUD é ciano numa gravação e verde noutra) e identifica cada uma **pelo ícone** da ponta esquerda, com um limiar por habilidade. Num mesmo quadro só o molde vencedor pontua: uma faixa anuncia uma habilidade | `SLEEP`, `STUN` |

O `beats` não está nesta tabela de propósito: ele **não é um detector**. Não
olha a partida, não emite evento e não roda na análise. Ele ouve o que entra na
biblioteca do editor e produz, para a música, uma grade de batidas
(`librosa.beat.beat_track`, com estimador próprio em numpy de reserva).

> **O que a calibração em gameplay real mudou.** A primeira versão media
> apenas "quanto da região está vermelha". Contra 19 minutos de partida real
> isso não funciona: o mundo do jogo (iluminação quente dos mapas, contornos
> vermelhos de inimigos) e o *indicador direcional de dano* pintam a mesma cor
> quase o tempo todo — metade dos quadros passava de 3% de vermelho na região,
> e a precisão ficava em ~17%. O que separa a caveira do resto não é a cor:
> é ela ser um blob **compacto, quase quadrado e centrado na mira**, enquanto o
> indicador de dano é um arco largo desenhado num raio acima do centro. Com
> forma e posição no critério a precisão subiu, e o filtro que faltava era o
> **tamanho**: com o mínimo em 0.4% da região, qualquer respingo vermelho de 20
> pixels virava eliminação — eram esses respingos a maior parte dos falsos
> positivos restantes. A caveira ocupa de 5% a 14% da ROI. Com os quatro
> critérios, a precisão foi para ~91%.
>
> A vida baixa era inferida da vinheta vermelha das bordas — que é o aviso de
> *dano recebido*, presente em 32% dos quadros de uma partida real, e rendia 122
> "fugas" em 19 minutos. Hoje o detector lê a barra de vida direto (erro ≤ 0.05
> contra os valores na tela). A morte era inferida da killcam dessaturada, e
> encontrava zero mortes: a killcam do OW2 não é dessaturada.
>
> Ultimates continuam dependendo de ícones do jogo, que são assets e não vêm no
> repositório. A via alternativa por pico de áudio existe mas **vem desligada**:
> num vídeo sintético ela acerta, em partida real não distingue fala de ultimate
> de tiro e explosão, e não há gabarito para calibrá-la honestamente.

## 4. O que a análise entrega

Ela entrega **eventos**, e mais nada. Não agrupa, não pontua e não propõe.

Um único cruzamento acontece depois dos detectores, em
`packages/owcore/owcore/rules.py`, e é o `planner` que o roda ao fechar a
análise:

| Evento derivado | Regra |
|---|---|
| `ULT_NEGATED` | uma `ULT_USED` inimiga seguida de `KILL` dentro de `ult_negate_window_s` (6 s). Nenhum detector sozinho vê os dois tipos — correlação entre microsserviços é trabalho de quem agrega |

Ele fica aqui, e não num detector, por isso; e é **gravado como evento**, para
aparecer na prateleira do editor junto com os outros.

> **O motor de regras que saiu.** Este arquivo já era três vezes maior. Ele
> transformava eventos em *highlights*: `MULTIKILL` (≥3 `KILL` em 10 s),
> `SOLO_WIPE`, `ESCAPE`, e uma família de montagens no ritmo — uma por tipo de
> evento, mais uma por habilidade. O `planner` gravava tudo isso como
> `proposals`, o app mostrava a lista, e o usuário escolhia quais gerar dando
> uma música a cada uma.
>
> Saiu inteiro na Fase 11, junto com `MONTAGE_KINDS`, `ClipOptions`,
> `Selection`, `montage_segments` e `fit_to_window`. A razão não é técnica: uma
> regra sabe agrupar eliminações numa janela de 10 segundos, e não sabe onde a
> música vira. Ela produzia montagens medianas e, pior, funcionava como um
> **filtro** entre a detecção e o usuário — o que virava proposta era o que ele
> via. Foi assim que headshots e mortes por habilidade, detectados desde
> sempre, nunca chegaram à prateleira do editor.
>
> O que vale agora é o inverso: **tudo o que é detectado vira bloco possível**,
> e o julgamento é de quem monta.

O que sobrou de "como cortar" mora no editor, e está descrito abaixo.

### 4.1 A montagem — onde tudo é decidido

Uma regra sabe agrupar eliminações numa janela de 10 segundos. Não sabe que a
música vira aos 47 s, nem que aquela pedrada merece um segundo a mais de embalo.
Para isso existe a linha do tempo, e nela o sistema não decide nada:

| Decisão | Quem toma |
|---|---|
| que momento entra | o usuário, escolhendo da lista de instantes que a análise achou — clicando, ou arrastando o momento até o ponto da régua onde ele deve entrar |
| onde ele entra no vídeo | o usuário, arrastando o corpo do bloco; com o **ímã** ligado ele gruda na batida mais próxima (até 0,12 s) |
| quanto ele dura | o usuário, arrastando as bordas: a direita **estica** (cresce o rabo), a esquerda **apara** (come o começo sem mover o que está enquadrado) |
| onde a jogada cai dentro do corte | por padrão a **70%** do bloco — sobra embalo antes e o impacto cai perto do fim, que é onde ele funciona. Ajustável bloco a bloco, e **marcado dentro do bloco**: é a jogada que se alinha com a percussão, não a borda do corte |
| o que aparece nos espaços vazios | **tela preta, com a música tocando** |
| onde a música entra | o usuário, pondo o bloco de música onde ele deve começar. A régua é o tempo do **vídeo**: o instante zero é o primeiro quadro dele, e a música mora dentro dessa escala |

> Houve aqui uma regra automática que não sobreviveu ao uso: o primeiro bloco
> movia sozinho a entrada da música para o cursor, para evitar tela preta antes
> dele. O efeito era o vídeo passar a começar com a música já em 0:02, jogando
> fora o começo dela — e a tela anunciando isso como se fosse pedido. Espaço
> vazio no começo é uma escolha visível na régua e contada no resumo;
> retimar a música às escondidas não é. (A entrada da música, hoje, é onde o
> bloco dela começa: a régua inteira é tempo de vídeo.)

**A música é um bloco, e vem da biblioteca.** Houve uma faixa contínua que tocava
por baixo de tudo e não se cortava; ela saiu. Hoje a música entra pela biblioteca
de mídia, como vídeo e imagem, e vai para uma **camada de som**, onde é um clipe
como qualquer outro: corta, anda, se apara, se duplica. Uma camada desenha ou
toca — nunca as duas coisas —, e o servidor recusa conteúdo trocado de camada.
O ímã segue a grade da música que está tocando sob a cabeça de leitura: duas
faixas num vídeo são dois andamentos.

Montagens antigas não migram no banco: `track_id` e `music_start_s` continuam
sendo entrada válida e viram, **na leitura**, um bloco que começa onde a faixa
entrava e cobre o vídeo inteiro.

| Decisão | Quem toma |
|---|---|
| que música toca, e em que pedaço do vídeo | o usuário, arrastando da biblioteca para a régua ou clicando para pôr na cabeça de leitura |
| de que ponto da música o bloco sai | o usuário, pelo "trecho da música" — e a grade de batidas é medida a partir daí |
| o que se ouve onde não há bloco | o áudio da partida, na medida do `game_volume`: o silêncio é a falta de bloco |

Esticar e aparar não reenquadram o conteúdo: o começo do corte só se move quando
é a borda esquerda que anda, e na mesma medida que ela. Se a imagem se
reenquadrasse a cada pixel do arrasto, ela escorregaria debaixo do dedo. O
reenquadramento a 70% é palpite de **criação**; depois disso quem o muda é o
controle de enquadramento, não a duração.

### 4.2 As miniaturas dos momentos

A barra lateral do editor mostra cada momento com um quadro da partida —
escolher entre trinta eliminações sem imagem é escolher entre trinta relógios
iguais. Quem extrai é o serviço `thumbs`, que escuta o fim do planejamento por
um *consumer group* próprio: recebe o mesmo aviso que o planejador e trabalha em
paralelo, sem segurar o job em `ready`. Se as miniaturas demorarem, ou nem
saírem, o resto funciona igual.

Não há tabela nem coluna para elas: a chave de cada quadro sai do instante
(`frame_key(job_id, t)`), então quem escreve e quem lê chegam nela sozinhos. Uma
miniatura a mais não é uma migração a mais.

`-ss` **antes** do input faz o ffmpeg pular até o keyframe mais próximo em vez
de decodificar desde o começo: um quadro sai em dezenas de milissegundos, e por
isso extrair um por vez sai mais barato do que uma passagem única pelo vídeo
inteiro — ao contrário do que vale para os recortes da análise, onde a
decodificação única é o ponto.

### 4.3 A montagem não se perde

Meia hora de encaixe na batida sumia num F5 — a montagem só existia na memória
da aba. Agora ela é do **job**, e são **várias**: cada uma na tabela `montages`,
com nome, porque o corte de 30 s para o Shorts e a montagem longa são trabalhos
diferentes sobre o mesmo material. O app salva sozinho um segundo e meio depois
da última mexida (um arrasto inteiro vira um salvamento só), e a tela recupera a
mais recente ao abrir.

Cada montagem é uma `Timeline` sem a exigência de estar pronta: aceita zero
cortes, porque existe desde antes de o primeiro bloco entrar. Cada bloco, esse
sim, é validado — guardar lixo agora seria devolver lixo na próxima abertura.

Gerar o vídeo **não** apaga a montagem: depois de gerar, o normal é querer
ajustar e gerar de novo, e perder o trabalho nesse ponto seria o mesmo estrago.
O que gerar faz é tirar uma **foto** (`montage_versions`) — o que saiu foi
aquilo, e é o que torna o histórico útil sem guardar um estado por salvamento
automático.

A coluna `draft` do job, que guardava a montagem única, é lida uma última vez:
na primeira abertura ela vira a primeira montagem nomeada e se esvazia. Quem
sabe converter o formato velho é o código que lê.

### 4.4 O monitor

A tela tem um preview, e ele **não renderiza nada**. Abre a gravação original
(`GET /api/jobs/{id}/video`, com `Range`) e busca dentro dela o instante que a
cabeça de leitura pede: se ela está sobre um bloco que sai dos 3 min da partida,
é aos 3 min que a gravação é posicionada. Onde não há bloco, tela preta — o
mesmo que o servidor vai gerar ali.

Renderizar de verdade a cada ajuste custaria uma volta inteira pelo ffmpeg por
arrasto. Buscar dentro do arquivo que já existe é instantâneo, e a conta que
traduz "tempo do vídeo" em "instante da gravação" é a mesma dos dois lados:
`origemEm()` no app é a versão de leitura do `plan()` que o servidor usa para
cortar.

O elemento de vídeo do navegador às vezes **morre** — uma gravação de meio giga
entregue por `Range`, com dezenas de buscas por segundo enquanto se arrasta,
derruba ele. Antes o monitor ficava preto até a página ser recarregada, e
recarregar custava a montagem. Hoje ele é vigiado: ao detectar o erro, o player
é reaberto no mesmo ponto, até quatro vezes, e só então a tela oferece um botão
de tentar de novo. As buscas também são serializadas — uma por vez, com folga de
120 ms —, porque buscas sobrepostas eram justamente o que o derrubava.

O que o monitor **não** garante é sincronia de quadro com a música durante a
reprodução: são dois elementos de mídia independentes, e a emenda entre blocos é
feita por busca. Dentro de um bloco a imagem corre sozinha. O corte exato é o do
arquivo final.

O preto é a decisão que carrega o resto. Emendar os blocos para tapar o buraco
economizaria uma codificação e **moveria todos os cortes seguintes** — cada um
sairia de onde o usuário o encaixou. A promessa da tela é que o bloco cai no
ponto da música onde ele foi posto; espaço vazio é escolha, não sobra.

Pela mesma razão, um corte que passa do fim da gravação é **aparado** e o que
sobrou do lugar dele vira preto, em vez de o vídeo encolher. E quando o ímã
age, ele gruda pelo lado que estiver mais perto de uma batida — se é o fim do
bloco que está a um triz da percussão, é o fim que manda: numa montagem, o que
se ouve é a troca de cena.

As contas ficam em `owcore/timeline.py` (servidor) e `frontend/lib/montage.dart`
(app), separadas de ffmpeg e de widget justamente porque têm resposta certa e
dão para testar sozinhas. O ímã existe dos dois lados de propósito: o app gruda
enquanto o usuário arrasta, e quem confere depois tem de chegar no mesmo número.

## 5. Contrato REST

| Rota | O que faz |
|---|---|
| `POST /api/jobs` | multipart com `video` e `params` (JSON). Só a gravação — sem música |
| `GET /api/jobs/{id}` | estado da análise, **eventos**, montagens salvas, biblioteca e histórico de **pedidos** com os clipes de cada um |
| `POST /api/jobs/{id}/tracks` | multipart com `audio`. Manda o sistema ouvir uma música: volta na hora, `pending`, e a análise (duração, BPM, batidas, forma de onda) roda no worker |
| `GET /api/tracks/{id}` | a música analisada — é o que a tela de montagem desenha |
| `GET /api/tracks/{id}/audio` | o arquivo em si, com `Range`, para o player do app tocar e buscar |
| `GET /api/jobs/{id}/video` | a gravação original, com `Range` — é o que o monitor da tela de montagem mostra |
| `GET /api/jobs/{id}/frame?t=` | o quadro daquele instante, para a barra lateral. 404 = ainda não extraída |
| `POST /api/jobs/{id}/frames` | manda extrair as que faltam (jobs novos já saem com elas) |
| `GET /api/jobs/{id}/montages` | as montagens desta partida, da mais recente para a mais antiga |
| `POST /api/jobs/{id}/montages` | começa uma montagem, vazia ou com um conteúdo dado |
| `PUT /api/jobs/{id}/montages/{mid}` | guarda a montagem e/ou renomeia (o app chama sozinho) |
| `POST /api/jobs/{id}/montages/{mid}/duplicate` | uma cópia, sem o histórico da original |
| `DELETE /api/jobs/{id}/montages/{mid}` | apaga a montagem e as versões dela |
| `GET/POST /api/jobs/{id}/montages/{mid}/versions` | o histórico, e marcar uma foto. `409` = nada mudou desde a última |
| `POST .../versions/{vid}/restore` | volta para uma foto; a de agora vira foto antes |
| `GET/POST /api/presets`, `PUT/DELETE /api/presets/{id}` | as predefinições. Não pertencem a partida nenhuma |
| `PUT /api/jobs/{id}/draft` | **legado**: escreve na montagem mais recente |
| `DELETE /api/jobs/{id}/draft` | **legado**: descarta as montagens da partida |
| `DELETE /api/tracks/{id}` | tira a música do job; vídeos já gerados com ela ficam |
| `POST /api/jobs/{id}/renders` | `timelines` (JSON) com as montagens a gerar. Não leva arquivo: a música já subiu pela biblioteca. Exige o job em `ready` |
| `GET /api/renders/{id}` | andamento e clipes de um pedido |
| `DELETE /api/renders/{id}` | apaga o pedido e os vídeos dele; a montagem salva fica |
| `GET /api/jobs/{id}/cortes.zip` | pacote da partida inteira, todos os pedidos |

## 6. Frontend (Flutter, mobile-first, roda na web)

- `/` lista de jobs com status ao vivo (polling)
- `/new` escolher a gravação — só isso: nada de música e nenhum ajuste
- `/job/:id` linha do tempo dos eventos, relatório dos detectores, histórico de
  pedidos, player e downloads. A ação principal é **abrir o editor**
- **tela de montagem**: layout de editor — prateleira de momentos com miniatura
  na lateral, monitor redimensionável em cima, a música desenhada (forma de onda
  + batidas) e tocando, e os blocos posicionados em cima dela — arrastar o corpo
  move, as bordas esticam e aparam, o ímã gruda na batida. É **o** caminho:
  basta a análise ter achado momentos

> A tela de geração (`generate_screen.dart`) e o seletor de trecho de música
> (`music_window.dart`) foram removidos com as propostas.

## 7. Etapas de execução

Backend e frontend são projetos separados, cada um na sua pasta:

1. `packages/owcore` — config, models, db, bus, storage, ffmpeg, profiles, worker base
2. `services/gateway` — REST + upload
3. `services/preprocessor` — recortes ROI + áudio
4. `services/detector_*` — visão computacional, um por *pergunta sobre a tela*. Uma região pode responder duas (`detector_kills` lê eliminação e acerto crítico na mesma mira) e uma pergunta pode vir de duas regiões (`detector_ults` lê o botão do rodapé e o killfeed). O que **não** existe é um serviço por habilidade: `detector_banner` e `detector_killfeed` distinguem cada uma pelo ícone
5. `services/planner` — fecha a fase 1: espera todos os detectores, cruza o que nenhum vê sozinho (`ULT_NEGATED`) e grava; põe o job em `ready` e pede as miniaturas
6. `services/beats` — ouve o que entra na biblioteca do editor (música, clipe, imagem); `services/editor` — corta e junta a montagem (fase 2)
7. `packages/owcore/owcore/timeline.py` — a linha do tempo vira lista de pedaços a cortar, buracos pretos inclusive
8. `tools/make_sample.py` — gerador de vídeo sintético (permite testar tudo sem gameplay real)
9. `tests/` — unitários + end-to-end no vídeo sintético
10. `frontend/` — app Flutter
11. `docker-compose.yml` na raiz — orquestra os dois: constrói as imagens de
   `./backend` e monta o `build/web` do `./frontend` no gateway
