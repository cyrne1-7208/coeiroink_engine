#!/usr/bin/env bash
# Engineのlockから一時仮想環境を作り、ライセンス一覧を生成する。

set -euxo pipefail

if [ ! -v OUTPUT_LICENSE_JSON_PATH ]; then
    echo "OUTPUT_LICENSE_JSON_PATHが未定義です"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_dir="$(cd "$script_dir/.." && pwd)"
core_dir="$(cd "$engine_dir/../coeiroink_core" && pwd)"
venv_path="$engine_dir/licenses_venv"

python_bin="${PYTHON_BIN:-python3.14}"
uv_bin="${UV_BIN:-uv}"
if [[ "$uv_bin" != /* ]]; then
    uv_bin="$(command -v "$uv_bin")"
fi
"$python_bin" -c 'import sys; assert (3, 12) <= sys.version_info[:2] < (3, 15), sys.version'
test -f "$engine_dir/pyproject.toml"
test -f "$engine_dir/uv.lock"
test -f "$core_dir/pyproject.toml"

"$uv_bin" venv --allow-existing --python "$python_bin" "$venv_path"
(
    cd "$engine_dir"
    UV_PROJECT_ENVIRONMENT="$venv_path" \
        SETUPTOOLS_SCM_PRETEND_VERSION=0.4.1 \
        PATH="$venv_path/bin:/usr/bin:/bin" \
        "$uv_bin" sync --locked --extra cpu --group licenses --no-dev
)

output_license_path="$(realpath -m "$OUTPUT_LICENSE_JSON_PATH")"
(
    cd "$engine_dir"
    "$venv_path/bin/python" generate_licenses.py > "$output_license_path"
)

rm -rf "$venv_path"
