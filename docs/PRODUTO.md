# OW Editor

> **Visão geral do produto inteiro.** No Git são **dois repositórios**: este é o
> do backend, e ele fica clonado **dentro** do repositório do app, na pasta
> `backend/` — que o de fora ignora. O `docker-compose.yml`, que sobe os dois,
> mora lá.

Envie a gravação de uma partida de Overwatch 2, veja o que o sistema conseguiu
separar e faça o vídeo: aceitando um dos cortes que ele propõe, ou **montando
você mesmo** — ouvindo a música no app e pondo cada eliminação, dardo ou
pedrada no ponto dela que você quiser, com a duração que você quiser.

**100% software livre**: FastAPI, Redis, MinIO, PostgreSQL, OpenCV, ffmpeg,
librosa, Flutter. Nenhum serviço pago, nenhuma API externa, nada que precise
de conta.

---

## Dois projetos, dois repositórios

```
ow_editor/                 ← repositório do app
├── docker-compose.yml         orquestra os dois
├── frontend/                  app Flutter (mobile-first, roda na web)
└── backend/              ← ESTE repositório (o de cima o ignora)
    ├── docs/                  este documento e o PLAN.md
    └── README.md              como rodar e testar o backend
```

Cada um se desenvolve e testa sozinho. A única coisa que os liga é o contrato
REST do gateway (`/api/...`), documentado automaticamente em
<http://localhost:8000/docs> quando o backend está no ar — e é essa
independência que deixa cada um ter o seu próprio repositório.

Para rodar tudo junto, o repositório do backend tem de estar clonado **dentro**
da pasta do outro, com o nome `backend`: é lá que o compose procura por ele.

---

## Subindo tudo

A partir da pasta `ow_editor/`, onde está o compose:

```bash
# opcional: compila o app para o gateway servi-lo na mesma origem
cd frontend && flutter build web --dart-define=API_BASE= && cd ..

docker compose up --build
```

- App + API → <http://localhost:8000>
- Console do MinIO → <http://localhost:9001> (`minioadmin` / `minioadmin`)

O `docker-compose.yml` mora no repositório do app, mas orquestra os dois:
constrói as imagens a partir de `./backend` e monta o `build/web` do
`./frontend` no gateway. É por isso que o backend precisa estar clonado ali
dentro com esse nome. Se você não compilou o app, a pasta montada fica vazia e o
gateway serve só a API — nada quebra.

## Desenvolvendo os dois em paralelo

```bash
# terminal 1 — backend sem Docker nenhum (fila em disco + SQLite)
cd backend && python tools/dev.py

# terminal 2 — app com hot reload, apontando para o backend acima
cd frontend && flutter run -d chrome
```

Em desenvolvimento o app usa `API_BASE=http://localhost:8000` (o padrão) e o
gateway já libera CORS. Em produção, compilado com `API_BASE` vazio, ele chama
a API em caminho relativo e não precisa de CORS.

---

## Duas fases

O sistema é um **editor de vídeo com detecção automática de eventos**. Ele não
gera nada sozinho: **analisa** e depois **espera**:

```
1. você envia a gravação      → o sistema assiste e anota o que aconteceu
2. você abre o editor         → os momentos estão na prateleira, com o quadro
                                de cada um
3. você monta e manda gerar   → o vídeo sai exatamente como foi montado
   ↑__________________________________________|
   pode repetir quantas vezes quiser: usar um momento não o consome
```

> **O que saiu.** Até a Fase 11 havia um segundo caminho: o sistema aplicava
> regras sobre os eventos ("três eliminações em 10s viram uma rajada") e
> oferecia uma lista de vídeos prontos para escolher — bastava dar uma música a
> cada um. Esse caminho foi descontinuado.
>
> Ele resolvia o problema errado. O que um editor de Overwatch não tem não é
> alguém que corte por ele: é um editor que **saiba o que aconteceu no vídeo**.
> As regras davam palpites medianos sobre montagem e, ao mesmo tempo,
> escondiam momentos que o detector tinha achado — headshots e mortes por
> habilidade viravam proposta e não apareciam na prateleira do editor. O
> diferencial estava na detecção, e ela agora vai inteira para quem edita.

### A espera do passo 1 tem de parecer trabalho

Analisar 11 minutos de gravação leva ~8 minutos, e três quartos disso é uma
única coisa: o recorte das regiões da HUD. A barra antiga dava a essa fase a
faixa de 15% a 25% — ela ficava parada no mesmo número por seis minutos, e
parada é como qualquer pessoa lê "travou". Foi exatamente essa a queixa que
apareceu no uso real.

Duas mudanças, e nenhuma delas acelera coisa alguma:

* o ffmpeg passou a ser **perguntado** por onde anda (`-progress`), e o
  download do vídeo, a contar bytes. A barra agora se move o tempo todo;
* as fatias da barra passaram a ser proporcionais ao **tempo** de cada fase, e
  não ao número de fases — download 0–17%, recorte 17–90%, áudio até 94%,
  detecção o resto. Os números saem de cronometrar uma partida real.

Só com as duas é que a tela pode dizer **quanto falta**, e a conta mais simples
que existe basta: o que já andou, na velocidade com que andou. Medido numa
partida de 11 min, ela disse "~6,5 min" e errou por 8 segundos. Enquanto a barra
mentia sobre o formato do trabalho, nenhuma estimativa se sustentaria.

### A montagem, por dentro

```
a. você traz a música para a biblioteca → o sistema ouve e devolve a onda e as
   batidas
b. você põe a música na régua e cada momento onde quiser, do tamanho que
   quiser, vendo no monitor o que vai sair — arrastando o bloco e as bordas
c. o vídeo sai exatamente assim
```

A separação entre as duas fases existe porque as duas coisas têm ritmos diferentes. A análise é
cara (decodifica o vídeo, roda visão computacional) e o resultado não muda: os
momentos da partida são os que são. Já a escolha é barata, pessoal e mutável —
hoje você quer uma montagem com uma música, amanhã a mesma montagem com outra.
Rodar a análise de novo para trocar de música seria pagar caro por nada.

Disso saem duas consequências que valem dizer em voz alta:

- **cada vídeo tem a sua música.** A trilha é um bloco na régua da montagem,
  não uma propriedade da partida. Duas montagens da mesma gravação saem com
  músicas diferentes, e a mesma montagem troca de música sem reanalisar nada;
- **vídeo sem música fica com o áudio original da partida.** Não é silêncio nem
  um padrão qualquer: é o som do jogo, que costuma ser metade da graça de um
  trecho corrido.

## O que ele encontra

| Momento | Como é reconhecido |
|---|---|
| **Rajada de eliminações** | 3+ eliminações numa janela curta |
| **Sozinho contra todos** | 4+ eliminações seguidas sem morrer no meio |
| **Fuga por pouco** | episódios de vida baixa seguidos, sobrevividos |
| **Montagem no ritmo** | eliminações avulsas, uma por corte, na batida da música |
| **Dardos no alvo** | o tranquilizante da Ana acertando alguém — montagem própria, no ritmo |
| **Pedradas certeiras** | a Accretion do Sigma atordoando alguém — montagem própria, no ritmo |
| **Ultimates anuladas** | ultimate inimiga seguida de eliminação em poucos segundos |

> Um mesmo momento pode aparecer em mais de um vídeo. A rajada de eliminações
> vira um clipe próprio **e** entra na montagem no ritmo — cada vídeo é uma
> montagem independente, não uma partilha do material.

### Escolhendo o trecho da música

Na hora de gerar, para **cada** vídeo, dá para dizer onde a música entra e onde
termina; o vídeo é montado para caber nesse trecho:

- **repetindo trechos**: o vídeo sai com **exatamente** a duração escolhida,
  reaproveitando momentos quando eles acabam antes da música. A ordem é
  **sorteada**, e um momento só reaparece depois que todos já entraram — assim
  a montagem não vira a mesma sequência em loop;
- **sem repetir**: o vídeo vai até onde os momentos derem, **sem nunca passar**
  da duração escolhida, em ordem cronológica.

O início é encaixado na primeira batida a partir do ponto pedido, para o
primeiro corte cair no tempo em vez de entrar no meio de um compasso.

### Mais de uma música, e música que se corta

Na montagem à mão a música entra pela **Biblioteca**, no mesmo lugar em que
entram vídeo e imagem de fora da partida — clique para pô-la na cabeça de
leitura, ou arraste-a até o ponto da régua onde ela deve começar.

Na régua ela é um **bloco**: corta, anda e se apara como qualquer corte de vídeo.
É isso que deixa fazer o que uma trilha de fundo nunca deixou:

- **trocar de música no meio do vídeo**, uma faixa por trecho;
- **deixar um pedaço em silêncio**, para o tiro e o grito da jogada aparecerem
  sozinhos;
- **pôr o refrão só na virada**, e não do começo ao fim.

O ímã continua funcionando — com mais de uma faixa, ele gruda na batida **da
música que está tocando ali**, que é a única grade que faz sentido naquele ponto.

Montagens feitas antes disto não se perdem: a trilha de fundo que elas tinham
volta como um bloco que começa no mesmo ponto da música e cobre o vídeo inteiro.
O que se ouve é o mesmo; o que muda é que agora dá para pegar nas pontas.

### Várias montagens da mesma partida

Uma partida rende mais de um vídeo. O corte de 30 s para o Shorts e a montagem
longa são trabalhos diferentes sobre o mesmo material, e cada um tem o seu nome
— dá para trocar entre eles pelo alto da tela, duplicar um para experimentar sem
arriscar o que já está bom, e apagar o que não deu certo.

Cada montagem guarda um **histórico**. Não é o desfazer, que vale só enquanto a
aba está aberta: são marcos. Um é guardado a cada vídeo gerado — o que saiu foi
aquilo —, e dá para marcar um a qualquer momento ("estava bom assim"). Voltar
para um deles não apaga o que estava na frente: isso também vira um marco antes.

### Predefinições: a segunda partida sai pronta

Uma predefinição não guarda cortes — guarda o **jeito** de cortar. "Dois
segundos por eliminação, encaixado na batida, com zoom e contador" vale para
qualquer partida; uma lista de cortes só vale para aquela.

Monte um vídeo do jeito que gosta, salve como predefinição, e a próxima partida
sai montada num clique. O que sai é uma montagem comum: cada bloco continua se
movendo, se aparando e se apagando como qualquer outro — a predefinição é um
ponto de partida, não um molde do qual não se sai.

O tamanho pode ser em segundos ou em **batidas**. Em batidas é melhor: sobrevive
a uma música de outro andamento.

### Encaixando a jogada na batida

Um corte é um trecho; a jogada — a eliminação, o dardo, a pedrada — é um
instante dentro dele, e é ela que precisa cair na batida. O corte começa antes,
para dar embalo, então alinhar pela borda deixa o impacto atrasado.

Na régua, cada bloco mostra **onde a jogada acontece**. Ponha a cabeça de
leitura na batida, escolha o bloco e mande alinhar (botão no painel do bloco, ou
a tecla **M**): a jogada vai para debaixo da linha, e a marca acende para
confirmar. Se os blocos vizinhos não deixarem esse andar, o trecho é que desliza
dentro do bloco — o vídeo continua com a mesma duração, e a tela diz o que fez.

Arrastando com o ímã ligado, a jogada também gruda na batida: ela disputa com as
duas bordas do bloco, e vence quem estiver mais perto.

### Escrevendo na tela

O texto entra na régua como qualquer bloco e aparece **no monitor**, do tamanho
e no lugar em que vai sair. É lá que se escolhe onde ele fica: arraste a frase
pelo quadro. Tamanho, cor e contorno ficam no painel do bloco — o contorno não é
enfeite, é o que faz texto branco sobreviver a uma cena clara.

### Escolhendo o formato de saída

Uma montagem não tem formato: ela tem cortes, camadas e efeitos. O formato é a
janela por onde se olha para ela, escolhida na hora de gerar — e por isso o
mesmo trabalho vira um **16:9** para o YouTube e um **9:16** para os Shorts sem
que nada dele mude.

Dá para escolher tamanho, taxa de quadros e qualidade; exportar **só um trecho**
(ou só o que está selecionado, para conferir uma emenda sem esperar o vídeo
inteiro); e pôr uma **marca d'água** de qualquer imagem da biblioteca, num dos
quatro cantos.

Quando a proporção pedida não é a da gravação, há duas respostas e as duas
estão lá: **preencher**, que corta as sobras dos lados — numa montagem de
gameplay a ação está no meio —, ou **caber**, que mostra o quadro inteiro e
deixa barras, para quando o que importa está nos cantos.

Trocar a música e reexportar refaz o vídeo: o som é montado junto com a imagem,
e não por cima dela. (Houve um atalho aqui, quando a música era uma trilha de
fundo que não se cortava; ele foi embora com ela.)

### Levando os cortes para editar por fora

Cada partida oferece um **zip com tudo** — direto da lista, sem abrir vídeo
nenhum:

```
pedido_01/videos/01_beat_montage.mp4          os vídeos prontos
pedido_01/cortes/01_beat_montage/01_00m43.7s.mp4  cada corte, nomeado pelo
pedido_01/cortes/01_beat_montage/02_01m22.2s.mp4  instante de origem
pedido_02/videos/01_beat_montage.mp4          o mesmo momento, outra música
...
```

Cada pedido tem a sua pasta: gerar a mesma montagem duas vezes com músicas
diferentes dá dois vídeos, não um sobrescrevendo o outro.

Cada corte aparece **uma vez**, mesmo quando a montagem o repetiu. Dentro do
player também há o zip só daquela montagem, para quando você quer apenas ela.

## Como funciona

```
análise (uma vez, automática)
Flutter ──▶ gateway ──▶ preprocessor ──▶ detectores ──▶ planner ──▶ momentos
             (API)      (1 decode,        (kills,       (fecha e     + `ready`
                        N recortes)       survival,      cruza)
                                          ults, sleep)
                                                            ⌛ espera você

a biblioteca do editor (música, clipe, imagem)
Flutter ──▶ gateway ──▶ beats ──▶ onda + batidas de volta para o app
             (API)     (ouve a
                       música)

geração (sob demanda, repetível)
Flutter ──▶ gateway ──▶ editor ──▶ clipes
             (API)     (cortes,
                        camadas,
                        trilha)
```

O ponto central: **o vídeo pesado é decodificado uma única vez**. Dessa
passagem saem recortes minúsculos da HUD, em FPS baixo, um por detector.
Nenhum detector abre o arquivo original — só o editor volta a ele, e só nos
segundos que interessam. O ritmo da música não está na análise: a música existe
quando o usuário a traz para a biblioteca, e é ouvida ali. Numa gravação 640×360 a 30 fps, o detector de
eliminações recebe um recorte de 102×64 a 12 fps: **1,1% dos pixels**. Os três
recortes somados, mais o áudio, dão 21 MB de um vídeo de 100 MB.

Detalhes de arquitetura e das regras: [`PLAN.md`](PLAN.md).

---

## O que está verificado, e o que não está

Sendo direto sobre os limites.

**Calibrado contra gameplay real.** Os detectores foram medidos contra 19
minutos de partida de verdade (2 partidas, 360p), não só contra o vídeo
sintético dos testes. Isso mudou muita coisa:

| | antes | depois |
|---|---|---|
| Precisão das eliminações | ~17% | **~91%** |
| Eliminações em 19 min | 491 (quase tudo falso) | 11 |
| "Fugas" em 19 min | 122 (a vinheta de dano disparando) | 8 |
| Mortes detectadas | 0 | 15 |

As **habilidades anunciadas no rodapé** foram calibradas do mesmo jeito, em
duas gravações, com cada molde treinado só na primeira metade do respectivo
vídeo:

| | gravação | encontrados | falsos |
|---|---|---|---|
| Dardo da Ana | 16 min de Ana | **11 de 11** | 0 |
| Pedrada do Sigma | 11 min de Sigma | **23 de 23** (12 nunca vistas) | 0 |

O rodapé do OW2 empilha vários avisos com a mesma cor, forma e posição
(`SAVED …`, `ORB OF HARMONY …`, `… STUNNED BY ACCRETION`); quem separa é o
**ícone** da ponta esquerda, e não o texto — texto muda de idioma, ícone não.
Por isso é **um detector para a faixa inteira**, não um por habilidade:
acrescentar uma habilidade é acrescentar um molde ao perfil.

Duas coisas que só apareceram ao medir a segunda gravação:

- **a cor do aviso muda de partida para partida** — ciano numa, verde na outra.
  Pior: o cenário azulado da gravação de Sigma caía na faixa ciano, e somar as
  duas cores numa máscara só colava o cenário na faixa verde, que deixava de
  existir como retângulo. Procurar **uma cor de cada vez** recuperou 5 das 23
  pedradas;
- **cada molde separa num ponto diferente.** O ícone da pedrada é cheio e
  contrastado, e casa alto até com avisos vizinhos; o do dardo é de traços
  finos. Um limiar único obrigava a escolher entre perder dardos e aceitar
  pedradas falsas, então o limiar é **por habilidade** (0,85 e 0,92).

O detector também achou uma pedrada na gravação da *Ana* — conferida a olho,
era real: um Sigma aliado acertando uma Accretion.

**Ultimate do jogador, acerto crítico e morte por habilidade** foram calibrados
primeiro em gravações curtas de 2558×1438, feitas para mostrar exatamente esses
elementos da HUD, e depois **conferidos em 27 minutos de partida inteira** (16
min de Ana e 11 min de Sigma). Os dois materiais dizem coisas diferentes, e é a
partida que manda: veja "O que a partida inteira mostrou" logo abaixo.

| | material | resultado |
|---|---|---|
| Ultimate do jogador | 3 clipes curtos + 27 min de partida | 4 usos em 27 min, sem nenhum falso; o ícone do disco nomeou herói e habilidade com 0,81–0,98 de correlação, contra 0,49 do segundo colocado entre 270 ícones |
| Acerto crítico | 8 s de Ashe, luneta com filtro magenta | achou o X vermelho; numa gravação com eliminações e nenhum headshot, zero falsos |
| Morte por habilidade | 16 s de Domina + 27 min de partida | 7 de 11 eliminações nos 16 min de Ana, **sem nenhum falso** e todas nomeadas certo; as 4 perdidas tinham o ícone abaixo do limiar |

### O que a partida inteira mostrou

Os clipes curtos aprovaram os dois detectores; a partida inteira reprovou. É a
diferença mais cara que este projeto mediu, e vale escrita:

| | clipes curtos | partida inteira, antes | depois |
|---|---|---|---|
| Ultimate do jogador (Ana + Sigma) | 3/3 | 14 eventos, 4 verdadeiros | 4 eventos, 4 verdadeiros |
| Morte por habilidade (Ana, 11 reais) | 2/2 | 27 eventos, 11 linhas reais | 7 eventos, 7 verdadeiros |

O que os clipes curtos não continham:

- **ninguém morre neles.** Ao morrer, o OW2 mostra a *kill cam*: um disco claro
  com o rosto de quem matou, cercado de um anel — no mesmo lugar do botão de
  ultimate e com a mesma forma. Numa partida de Ana isso virou uma "ultimate do
  Roadhog"; numa de Sigma, Hanzo e Junkrat. O que separa os dois é o relógio:
  em 27 minutos, toda faixa curta (0,2–0,6 s) era falsa e toda ultimate de
  verdade ficou carregada 2,8 s ou mais;
- **o killfeed tem uma linha só, e isolada.** Numa partida ele empilha várias,
  elas deslizam quando chega uma nova, e o recorte do ícone falha por 3 a 5
  segundos seguidos no meio da vida de uma linha. A mesma eliminação saía duas,
  três, quatro vezes;
- **quase todo abate é com arma comum**, e o vão entre as placas continua tendo
  o `>` para casar com alguma coisa. Com o limiar em 0,55, `dva/light_gun` e
  `baptiste/exo_boots` — arma e passiva, que nem aparecem no killfeed — viravam
  eliminação. Em 0,65 isso some.

A lição, escrita para a próxima vez: **um clipe gravado para mostrar um elemento
da HUD mostra esse elemento e mais nada.** Ele não tem morte, não tem killfeed
cheio, não tem a partida em volta. Serve para achar a região e conferir o
casamento; não serve para calibrar limiar nenhum.

Três coisas que só apareceram ao medir:

- **decidir headshot por cor não funciona.** A luneta da Ashe tinge a tela
  inteira de magenta, e aí *tudo* cai na faixa de matiz do vermelho. O que não
  se move com o filtro é a **dominância** do canal vermelho sobre os outros
  dois: o marcador dá ~107 e o cenário tingido, ~20. E a forma decide o resto:
  as quatro diagonais pintadas com as quatro direções retas limpas — a caveira
  de eliminação, também vermelha e também na mira, preenche as oito;
- **a linha do killfeed fica segundos na tela**, então a presença dela não
  marca instante nenhum. O que marca é ela **aparecer**. Contar quantas linhas
  de cada habilidade há na tela quase resolve, e foi a primeira tentativa — mas
  não distingue "a mesma linha sumiu e voltou" de "apareceu uma linha nova com
  a mesma habilidade", e essas duas são exatamente os dois casos que importam.
  Hoje cada linha é **acompanhada** pelas suas bordas horizontais: as de dentro
  cercam o ícone e já estão paradas no primeiro quadro; as de fora são o
  comprimento dos nomes e distinguem uma linha da outra. A altura não entra —
  quando chega uma eliminação nova a pilha inteira desliza, e uma identidade
  presa a ela trocaria de linha justamente aí;
- **descartar pedaços pequenos do desenho parecia limpeza barata** e não é:
  metade dos ícones do jogo é feita de partes soltas, e cortá-las muda o
  enquadramento de um quadro para o outro. Numa gravação real isso transformou
  uma eliminação em quatro.

**Limites conhecidos:**

- um headshot que *mata* pode passar despercebido. A caveira de eliminação
  nasce ~0,1 s depois do marcador crítico e cobre as mesmas diagonais; a 12 fps
  nem sempre sobra um quadro entre as duas. A eliminação continua sendo
  detectada — o que se perde é o rótulo de "na cabeça";
- **morte por habilidade acha 7 de cada 11**, e essa troca é deliberada: o
  limiar do ícone está onde a precisão é 100%. Uma prateleira com um momento a
  menos é melhor que uma que oferece um corte que não é do que diz ser: quem
  monta confia no rótulo e não volta a conferir a gravação;
- **duas eliminações do mesmo jogador sobre a mesma vítima, com a mesma
  habilidade e dentro de ~7 s uma da outra, contam como uma.** As duas linhas
  são idênticas em tudo que o detector usa para reconhecê-las. Exige a vítima
  renascer e ser morta de novo no mesmo lugar; é raro, e o preço de errar para
  o outro lado — repetir uma eliminação — é bem mais caro;
- **a ultimate exige 2 s carregada.** No campo de treino a carga é instantânea,
  então um clipe gravado lá pode não render evento. Em partida isso não
  acontece, e foi justamente por deixar dois clipes de campo de treino mandarem
  na calibração que 10 falsos passaram antes.

O que estava errado: medir "quanto da região está vermelha" não distingue a
caveira de eliminação do cenário do jogo nem do indicador direcional de dano.
A decisão agora usa **forma, posição e tamanho** — a caveira é um blob
compacto, centrado na mira, que ocupa de 5% a 14% da região. Esse último
filtro é o que impede um respingo vermelho de 20 pixels de virar eliminação.
E a vida baixa vinha da vinheta vermelha das bordas, que na verdade é o aviso
de *dano recebido*; hoje o sistema lê a barra de vida direto.

### Se o vídeo não sair, os cortes saem

Os trechos são cortados **antes** de serem juntados, e o zip é fechado nesse
ponto. Se a junção ou a trilha falharem, o clipe aparece como "vídeo não gerado
— cortes disponíveis" e o download continua lá. Material já cortado não se perde
por causa da etapa seguinte.

**Montagem manual verificada de ponta a ponta.** O teste sobe a música pela
API, roda quem a ouve, monta dois blocos com um buraco de 1 s entre eles, pede
a geração e **abre o mp4 que saiu**: ele dura os 4 s certos (os 3 s de corte
mais 1 s de preto) e tem faixa de áudio. O que não dá para testar assim é o
player de áudio dentro do app — se a música não tocar no seu navegador, a
montagem continua possível pela onda e pelas batidas desenhadas, e a tela diz
isso em vez de travar.

**Suíte de testes**: 223 no backend, 241 no frontend. Cobrem as duas fases, os
contratos entre os microsserviços, o corte no ritmo, músicas diferentes no
mesmo pedido, a montagem manual (o buraco que vira preto, o corte aparado que
não move os vizinhos, o ímã da batida), a resiliência (um detector que falha
não derruba o job; uma música ilegível falha sozinha) e a precisão dos
detectores contra o gabarito do vídeo sintético. O caso "sem
música fica com o áudio original" é verificado no arquivo gerado: o teste baixa
o mp4 e checa que ele tem faixa de áudio.

**O que não está resolvido:**

- **Ultimates inimigas** precisam de ícones do jogo em `templates/ults/`
  — são assets do Overwatch e não vêm no repositório. Sem eles o detector emite
  zero, de propósito. A via alternativa por pico de áudio existe mas vem
  desligada: em partida real ela não distingue fala de ultimate de tiro e
  explosão, e não há gabarito para calibrá-la honestamente. (A ultimate **do
  jogador** é outra história: essa sai do botão do rodapé e não depende de
  nenhum asset.)
- **Nomes de habilidade em inglês.** O rótulo de uma montagem por habilidade sai
  do arquivo do ícone, que veio da Blizzard: "Orisa: Energy Javelin". Traduzir
  exigiria uma tabela de 270 linhas para envelhecer a cada herói novo, e o nome
  original é o que aparece na tela de herói do jogo.
- **Duas ultimates encadeadas contam como uma.** D.Va e Dmon liberam a segunda
  ultimate logo depois da primeira; quando as duas caem dentro da mesma janela
  — no campo de treino, onde a carga é instantânea — só a última vira evento.
  Em partida, onde recarregar leva dezenas de segundos, as duas saem separadas.
- **`DEATH` não é exatamente "morte"**. O sinal é a vida zerar ou a HUD sumir, o
  que cobre morte, killcam, troca de round e seleção de herói. Para as regras é
  o que importa (a sequência do jogador foi interrompida), e o app rotula isso
  como "Interrupção" em vez de prometer mais do que entrega.
- **Um só material de referência.** Os limiares foram calibrados numa gravação.
  Idioma, modo daltônico, proporção de tela fora de 16:9 e patches do jogo podem
  exigir recalibrar — `tools/calibrate.py` existe para isso.
- **Recall das eliminações não foi medido.** Sei que ~94% do que ele aponta é
  eliminação de verdade; não sei quantas ele deixa passar, porque isso exigiria
  rotular manualmente as 19 minutos de killfeed.
