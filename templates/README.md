# Templates

Recortes PNG de ícones da HUD do Overwatch, usados como referência no
casamento de template. São **assets do jogo**, então não acompanham o
repositório — recorte da sua própria gravação.

```
templates/
├── kills/    caveira de eliminação (opcional)
└── ults/     ícones de ultimate no killfeed (opcional)
```

## Como recortar

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

- **Eliminações** e **sobrevivência** não usam template nenhum.
- **Ultimates** continuam sendo detectadas pelo pico de áudio; o que fica
  desligado é só o reconhecimento de *qual* ultimate apareceu no killfeed.
  O serviço avisa isso no log em vez de fingir que detectou.

Com templates, quando as duas pistas (áudio e killfeed) apontam o mesmo
instante, a confiança do evento sobe bem acima do que qualquer uma daria
sozinha.
