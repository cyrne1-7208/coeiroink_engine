#!/usr/bin/env bash
set -euo pipefail

image="${1:?An image is required}"
backend="${2:?A backend is required}"

if [ "$backend" = cpu ]; then
  docker run --rm "$image" .venv/bin/python -c "import parselmouth"
fi

# GitHub-hosted runnerにはGPUがないため、GPU版もCPUで起動し、HTTP応答だけを確認する。
container_id="$(docker run --detach --env COEIROINK_DEVICE=cpu --publish 127.0.0.1:50032:50032 "$image")"
cleanup() {
  local status="$?"
  if [ "$status" -ne 0 ]; then
    docker logs "$container_id" || true
  fi
  docker rm --force "$container_id" >/dev/null
}
trap cleanup EXIT

for attempt in $(seq 1 12); do
  status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
    http://127.0.0.1:50032/voicevox/version || true)"
  if [ "$status" = 200 ]; then
    echo "$backend container is ready"
    exit 0
  fi
  echo "Attempt $attempt/12: HTTP status ${status:-connection failure}"
  sleep 5
done
exit 1
