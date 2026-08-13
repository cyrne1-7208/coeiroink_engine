#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_dir="$(cd "$script_dir/.." && pwd)"
venv_dir="${1:-$engine_dir/.venv}"
core_dir="${2:-$engine_dir/../coeiroink_core}"
python_bin="${PYTHON_BIN:-python3.12}"

"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
test -f "$core_dir/requirements.txt"

"$python_bin" -m venv "$venv_dir"
venv_python="$venv_dir/bin/python"

"$venv_python" -m pip install --upgrade \
  'pip>=25,<27' \
  'setuptools>=75,<80' \
  'wheel>=0.45,<1'
"$venv_python" -m pip install \
  'Cython>=3.0.11,<3.1' \
  'numpy>=1.26.4,<2' \
  'cmake==3.31.6' \
  ninja
"$venv_python" -m pip install \
  'torch==2.3.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu

# Some hardened Linux kernels reject the executable-stack flag in native
# PyTorch libraries. Clearing the flag does not change executable code or model
# weights and is harmless on hosts that do not enforce it.
"$venv_python" -m pip install 'patchelf==0.19.1.0'
purelib="$($venv_python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$venv_dir/bin/patchelf" --clear-execstack "$purelib/torch/lib/libtorch_cpu.so"

# Build pyopenjtalk with the current Open JTalk dictionary package.  The
# explicit no-isolation mode keeps the native build on the Python 3.12 tools
# installed above.
env PATH="$venv_dir/bin:/usr/bin:/bin" "$venv_python" -m pip install \
  --no-build-isolation \
  -r "$core_dir/requirements-pyopenjtalk.txt"
"$venv_python" -m pip install --no-deps \
  -r "$core_dir/requirements-espnet.txt"

env PATH="$venv_dir/bin:/usr/bin:/bin" "$venv_python" -m pip install \
  --no-build-isolation \
  -r "$core_dir/requirements.txt"
"$venv_python" -m pip install --editable "$core_dir"
env PATH="$venv_dir/bin:/usr/bin:/bin" "$venv_python" -m pip install \
  --no-build-isolation \
  -r "$engine_dir/requirements.txt"

"$venv_python" -c 'import coeirocore, espnet, torch; print(torch.__version__)'
