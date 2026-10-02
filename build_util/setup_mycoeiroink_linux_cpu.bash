#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_dir="$(cd "$script_dir/.." && pwd)"
venv_dir="$(realpath -m "${1:-$engine_dir/.venv}")"
core_dir="$(realpath -m "${2:-$engine_dir/../coeiroink_core}")"
python_bin="${PYTHON_BIN:-python3.12}"
uv_bin="${UV_BIN:-uv}"

"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
test -f "$engine_dir/pyproject.toml"
test -f "$engine_dir/uv.lock"
test -f "$core_dir/pyproject.toml"

expected_core_dir="$(realpath -m "$engine_dir/../coeiroink_core")"
if [ "$core_dir" != "$expected_core_dir" ]; then
  echo "CoreはEngineの隣接するcoeiroink_coreでなければなりません: $expected_core_dir" >&2
  exit 1
fi

if [ -x "$venv_dir/bin/python" ]; then
    "$venv_dir/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
else
    "$uv_bin" venv --allow-existing --python "$python_bin" "$venv_dir"
fi

venv_python="$venv_dir/bin/python"

# CPU版の依存とビルドツールはEngineのlockから一度に同期する。
(
  cd "$engine_dir"
  UV_PROJECT_ENVIRONMENT="$venv_dir" \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.4.1 \
    PATH="$venv_dir/bin:/usr/bin:/bin" \
    "$uv_bin" sync --locked --extra cpu --group build --no-dev
)

# 一部のLinuxカーネルがPyTorch共有ライブラリの実行スタック属性を拒否するため、
# wheelに含まれる共有ライブラリだけ属性を明示的に解除する。
purelib="$("$venv_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
torch_cpu_library="$purelib/torch/lib/libtorch_cpu.so"
test -f "$torch_cpu_library"
"$venv_dir/bin/patchelf" --clear-execstack "$torch_cpu_library"

# importと標準辞書の初期化をビルド時に検証し、初回HTTP要求へ遅延させない。
"$venv_python" - <<'PY'
import importlib.metadata
import espnet2
import pyopenjtalk
import torch

print(f"torch={torch.__version__}")
print(f"espnet={importlib.metadata.version('espnet')}")
print(f"pyopenjtalk={pyopenjtalk.__version__}")
pyopenjtalk.g2p("ビルド確認")
PY
