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

O sistema não gera nada sozinho. Ele **analisa** e depois **espera**:

```
1. você envia a gravação          → o sistema separa os momentos importantes
2. o sistema diz o que dá gerar   → você escolhe quais quer
3. você dá a música de cada vídeo → os vídeos são gerados
   ↑______________________________________________|
   pode repetir quantas vezes quiser: as propostas continuam lá
```

Ou, no lugar do passo 2, você monta o vídeo à mão:

```
2'. você traz a música para a biblioteca → o sistema ouve e devolve a onda e as
    batidas
3'. você põe a música na régua e cada momento onde quiser, do tamanho que
    quiser, vendo no monitor o que vai sair — arrastando o bloco e as bordas
4'. o vídeo sai exatamente assim
```

Os dois convivem. As propostas são um palpite bom para quando você só quer o
vídeo pronto; a montagem é para quando você já sabe onde a música vira e quer o
corte caindo *ali*.

A separação existe porque as duas coisas têm ritmos diferentes. A análise é
cara (decodifica o vídeo, roda visão computacional) e o resultado não muda: os
momentos da partida são os que são. Já a escolha é barata, pessoal e mutável —
hoje você quer uma montagem com uma música, amanhã a mesma montagem com outra.
Rodar a análise de novo para trocar de música seria pagar caro por nada.

Disso saem duas consequências que valem dizer em voz alta:

- **cada vídeo tem a sua música.** As opções (trilha, trecho, repetição) são
  por vídeo, não por partida. Dois vídeos do mesmo pedido podem sair com
  músicas diferentes;
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
Flutter ──▶ gateway ──▶ preprocessor ──▶ detectores ──▶ planner ──▶ propostas
             (API)      (1 decode,        (kills,       (regras)
                        N recortes)       survival,
                                          ults, sleep)
                                                            ⌛ espera você

a música, quando você vai montar à mão
Flutter ──▶ gateway ──▶ beats ──▶ onda + batidas de volta para o app
             (API)     (ouve a
                       música)

geração (sob demanda, repetível)
Flutter ──▶ gateway ──▶ beats ──▶ editor ──▶ clipes
             (API)     (ritmo de  (cortes,
                     cada música)  trilha)
```

O ponto central: **o vídeo pesado é decodificado uma única vez**. Dessa
passagem saem recortes minúsculos da HUD, em FPS baixo, um por detector.
Nenhum detector abre o arquivo original — só o editor volta a ele, e só nos
segundos que interessam. O ritmo da música saiu da análise e foi para a
geração: é lá que a música existe. Numa gravação 640×360 a 30 fps, o detector de
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
  explosão, e não há gabarito para calibrá-la honestamente.
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
