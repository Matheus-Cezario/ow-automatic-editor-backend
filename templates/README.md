# Templates

Recortes PNG de ícones da HUD do Overwatch, usados como referência no
casamento de template. São **assets do jogo**, então não acompanham o
repositório.

```
templates/
├── kills/      caveira de eliminação (opcional, recortada à mão)
├── ults/       ícones de ultimate inimiga no killfeed (opcional, à mão)
└── abilities/  ícones oficiais de TODAS as habilidades — baixados
```

## `abilities/` — baixado, não recortado

```bash
python tools/fetch_ability_icons.py
```

São ~270 arquivos em `abilities/<herói>/<habilidade>.png`, um por habilidade de
cada herói do jogo. A lista sai da página oficial de heróis da Blizzard (via a
[OverFast API](https://overfast-api.tekrop.fr)), então **herói novo entra
rodando o comando de novo** — não há lista escrita no repositório para
envelhecer a cada patch.

Recortar 270 ícones à mão da própria gravação não é razoável, e é por isso que
esta pasta é a exceção. Quem usa esses ícones:

* o **detector de ultimates**, para dizer de qual herói era a ultimate que o
  jogador usou (o desenho preto dentro do disco branco do botão do rodapé);
* o **detector de killfeed**, para dizer com que habilidade cada eliminação foi
  feita (o desenho na caixinha entre as duas placas coloridas).

Os dois comparam a **marca** — os pixels do desenho, recortados do fundo,
enquadrados num quadrado e normalizados de tamanho. Por isso o mesmo arquivo
serve para as duas formas em que o jogo desenha o ícone: preto sobre disco
branco (ultimate) e branco sobre caixa escura (habilidade comum).

O arquivo é gravado como marca **preta sobre fundo branco**, que é o que o
`IconBank` espera. O ícone da Blizzard vem branco com fundo transparente — o
desenho vive no canal alfa —, e o baixador converte.

## `kills/` e `ults/` — recortados da sua gravação

```bash
# gera imagens das regiões a partir da sua gravação
python tools/calibrate.py preview --video partida.mp4 --at 30 90 150
```

Abra `data/calib/roi_*.png`, recorte o ícone bem justo (sem margem de cenário
em volta) e salve em `templates/ults/nome_do_heroi.png`. O nome do arquivo vira
o rótulo do evento detectado.

Tamanho ideal: até ~96px de largura. Imagens maiores são reduzidas
automaticamente.

## Sem templates o sistema funciona

- **Eliminações**, **acertos críticos** e **sobrevivência** não usam template
  nenhum.
- A **ultimate do jogador** continua sendo detectada sem `abilities/`: o botão
  do rodapé descarregando é o evento, e os ícones só dizem de quem ela era.
- **Ultimates inimigas** continuam saindo pelo pico de áudio; o que fica
  desligado sem `ults/` é só o reconhecimento de qual ultimate apareceu no
  killfeed. O serviço avisa isso no log em vez de fingir que detectou.
- **Eliminações com habilidade** ficam desligadas sem `abilities/` — de
  propósito. Uma eliminação sem saber com o que foi já é o que o detector da
  mira reporta; repetir isso aqui só duplicaria evento.
