# Uma imagem parametrizada serve todos os microsservicos: eles compartilham o
# mesmo pacote `owcore` e mudam so nas dependencias e no processo de entrada.
FROM python:3.12-slim

ARG SERVICE
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OW_ROOT=/app \
    OW_MODE=docker

# ffmpeg faz todo o corte e recorte; libgl/libglib sao do OpenCV headless
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencias primeiro, para o cache de camadas sobreviver a mudanca de codigo
COPY requirements-base.txt /app/requirements-base.txt
COPY services/${SERVICE}/requirements.txt /app/services/${SERVICE}/requirements.txt
RUN pip install -r /app/services/${SERVICE}/requirements.txt

COPY packages/owcore /app/packages/owcore
RUN pip install /app/packages/owcore

COPY config /app/config
COPY templates /app/templates
COPY services/${SERVICE} /app/service

RUN useradd --create-home --uid 1000 ow && mkdir -p /app/data && chown -R ow /app/data
USER ow

WORKDIR /app/service
CMD ["python", "main.py"]
