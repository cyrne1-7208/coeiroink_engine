#!/usr/bin/env bash
set -euo pipefail

image="${1:?An image is required}"
backend="${2:?A backend is required}"

if [ "$backend" = cpu ]; then
  docker run --rm "$image" .venv/bin/python -c "import parselmouth"
fi

# イメージ内のPythonでテスト用の話者を作る。コンテナの実行ユーザーが書き込めるように権限を広げる。
speaker_info_dir="$(mktemp -d)"
chmod 0777 "$speaker_info_dir"
style_id="$(docker run --rm --volume "$speaker_info_dir:/tmp/speaker_info" "$image" \
  .venv/bin/python build_util/make_ci_speaker.py /tmp/speaker_info | tail -n 1)"

# GitHub-hosted runnerにはGPUがないため、GPU版もCPUで起動して確かめる。
container_id="$(docker run --detach --env COEIROINK_DEVICE=cpu \
  --volume "$speaker_info_dir:/opt/coeiroink/speaker_info" \
  --publish 127.0.0.1:50032:50032 "$image")"
cleanup() {
  local status="$?"
  if [ "$status" -ne 0 ]; then
    docker logs "$container_id" || true
  fi
  docker rm --force "$container_id" >/dev/null
}
trap cleanup EXIT

ready=false
for attempt in $(seq 1 12); do
  status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
    http://127.0.0.1:50032/voicevox/version || true)"
  if [ "$status" = 200 ]; then
    ready=true
    break
  fi
  echo "Attempt $attempt/12: HTTP status ${status:-connection failure}"
  sleep 5
done
if [ "$ready" != true ]; then
  exit 1
fi
echo "$backend container is ready"

# テスト用の話者で、モデルの読み込みから合成までを確かめる。
query="$(curl --silent --show-error --fail --max-time 300 --request POST --get \
  --data "speaker=$style_id" --data-urlencode "text=こんにちは" \
  http://127.0.0.1:50032/voicevox/audio_query)"
wave_path="$(mktemp)"
curl --silent --show-error --fail --max-time 300 --request POST \
  --header "Content-Type: application/json" --data-binary "$query" \
  --output "$wave_path" "http://127.0.0.1:50032/voicevox/synthesis?speaker=$style_id"
if [ "$(head -c 4 "$wave_path")" != RIFF ]; then
  echo "$backend container did not return a WAV file" >&2
  exit 1
fi
echo "$backend container synthesized speech"
