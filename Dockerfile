# CoreとEngineの両ディレクトリを含む親ディレクトリをビルドコンテキストにする。
ARG PYTHON_IMAGE=python:3.14-slim-bookworm
ARG UV_VERSION=0.12.5
FROM ${PYTHON_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/coeiroink/coeiroink_engine/.venv

WORKDIR /opt/coeiroink

# pyopenjtalkとPyWorldのビルド、ESPnetの取得、音声出力に必要なLinux依存。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY coeiroink_core /opt/coeiroink/coeiroink_core
COPY coeiroink_engine /opt/coeiroink/coeiroink_engine

WORKDIR /opt/coeiroink/coeiroink_engine

# 汎用GPU runtimeを暗黙に持ち込まない再現可能な配布物とするため、DockerはLinux CPU profileを既定に固定する。
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.4.1 \
    uv sync --locked --extra cpu --group build --no-dev \
    && torch_cpu_library="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/torch/lib/libtorch_cpu.so" \
    && test -f "$torch_cpu_library" \
    && .venv/bin/patchelf --clear-execstack "$torch_cpu_library" \
    && .venv/bin/python -c "import importlib.metadata, espnet2, pyopenjtalk, torch; print(f'torch={torch.__version__}'); print(f'espnet={importlib.metadata.version(\"espnet\")}'); print(f'pyopenjtalk={pyopenjtalk.__version__}'); pyopenjtalk.g2p('コンテナ確認')"

RUN useradd --create-home --uid 10001 coeiroink \
    && mkdir -p /opt/coeiroink/speaker_info \
    && chown -R coeiroink:coeiroink /opt/coeiroink

USER coeiroink

EXPOSE 50032

CMD ["uv", "run", "--locked", "--extra", "cpu", "--no-sync", "python", "run.py", "--host", "0.0.0.0", "--speaker_info_dir", "/opt/coeiroink/speaker_info"]
