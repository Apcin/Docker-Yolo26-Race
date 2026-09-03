FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics-config \
    YOLO_OFFLINE=true

COPY environment.yml /tmp/environment.yml
RUN conda env create --file /tmp/environment.yml \
    && conda clean --all --yes

COPY app/ /app/
WORKDIR /app

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pytorch_base", "python", "/app/main.py"]
