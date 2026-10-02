# CoreとEngineの両ディレクトリを含む親ディレクトリをビルドコンテキストにする。
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG UV_VERSION=0.12.5
ARG COEIROINK_BACKEND=cpu
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM ${PYTHON_IMAGE}

ARG COEIROINK_BACKEND=cpu

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/coeiroink/coeiroink_engine/.venv \
    COEIROINK_BACKEND=${COEIROINK_BACKEND} \
    COEIROINK_DEVICE=${COEIROINK_BACKEND}

WORKDIR /opt/coeiroink

RUN useradd --create-home --uid 10001 coeiroink \
    && install -d -o coeiroink -g coeiroink /opt/coeiroink/speaker_info

# pyopenjtalkとPyWorldのビルド、ESPnetの取得、音声出力に必要なLinux依存。
# OpenCLだけはホスト側のICDを利用するため、コンテナにはloaderのみ入れる。
RUN set -eux; \
    apt-get update; \
    case "$COEIROINK_BACKEND" in \
        cpu|cuda) \
            apt-get install -y --no-install-recommends \
                build-essential ca-certificates git libgomp1 libsndfile1 libsqlite3-dev ;; \
        opencl) \
            apt-get install -y --no-install-recommends \
                build-essential ca-certificates git libgomp1 libsndfile1 libsqlite3-dev \
                ocl-icd-libopencl1 ocl-icd-opencl-dev opencl-headers ;; \
        *) \
            echo "Unsupported Linux container backend: $COEIROINK_BACKEND" >&2; exit 2 ;; \
    esac; \
    rm -rf /var/lib/apt/lists/*

COPY --chown=coeiroink:coeiroink coeiroink_core /opt/coeiroink/coeiroink_core
COPY --chown=coeiroink:coeiroink coeiroink_engine /opt/coeiroink/coeiroink_engine

WORKDIR /opt/coeiroink/coeiroink_engine

# バックエンドごとに依存profileを分離し、同じイメージへ異なるTorch wheelを混在させない。
# 公式PythonイメージのC++共有ライブラリ用リンカーはgccのため、pyopenjtalkをlibstdc++へ正しくリンクする。
RUN CXX=g++ LDCXXSHARED="g++ -shared" \
    uv sync --locked --extra "$COEIROINK_BACKEND" --group build --no-dev \
    && torch_cpu_library="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/torch/lib/libtorch_cpu.so" \
    && test -f "$torch_cpu_library" \
    && .venv/bin/patchelf --clear-execstack "$torch_cpu_library" \
    && .venv/bin/python -c "import importlib.metadata, espnet2, pyopenjtalk, torch; print(f'torch={torch.__version__}'); print(f'espnet={importlib.metadata.version(\"espnet\")}'); print(f'pyopenjtalk={pyopenjtalk.__version__}'); pyopenjtalk.g2p('コンテナ確認')"

USER coeiroink

EXPOSE 50032

CMD ["sh", "-c", "exec uv run --locked --extra \"$COEIROINK_BACKEND\" --no-sync python run.py --host 0.0.0.0 --device \"$COEIROINK_DEVICE\" --speaker_info_dir /opt/coeiroink/speaker_info"]
