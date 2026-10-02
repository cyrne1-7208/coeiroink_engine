# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /opt/coeiroink_engine

# The setup script builds pyopenjtalk and installs the CPU PyTorch wheel.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        git \
        libsndfile1 \
        libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Build with the repository workspace as the context:
# docker build -f coeiroink_engine/Dockerfile -t coeiroink-engine:cpu .
COPY coeiroink_engine/ /opt/coeiroink_engine/
COPY coeiroink_core/ /opt/coeiroink_core/

RUN bash build_util/setup_mycoeiroink_linux_cpu.bash /opt/venv /opt/coeiroink_core \
    && rm -rf /root/.cache/pip

VOLUME ["/speaker_info"]
EXPOSE 50032

ENTRYPOINT ["/opt/venv/bin/python", "/opt/coeiroink_engine/run.py"]
CMD ["--host", "0.0.0.0", "--speaker_info_dir", "/speaker_info", "--cpu_num_threads", "1"]
