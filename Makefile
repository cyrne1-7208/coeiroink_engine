IMAGE_NAME ?= coeiroink-engine
TAG ?= latest
CONTEXT ?= ..
UV ?= uv

.PHONY: lock sync sync-cpu sync-cuda sync-directml sync-opencl test build-docker

lock:
	$(UV) lock

# backend profileは一つだけ同期し、引数なしの開発環境はLinux CPU版を既定にする。
sync: sync-cpu

sync-cpu:
	$(UV) sync --locked --extra cpu

sync-cuda:
	$(UV) sync --locked --extra cuda

sync-directml:
	$(UV) sync --locked --extra directml

sync-opencl:
	$(UV) sync --locked --extra opencl

test:
	$(UV) run --locked --extra cpu --group dev pytest -q

build-docker:
	docker build -f Dockerfile -t $(IMAGE_NAME):$(TAG) $(CONTEXT)

.PHONY: run-docker
run-docker:
	docker run --rm -it -p 127.0.0.1:50032:50032 \
		-v "$(PWD)/speaker_info:/opt/coeiroink/speaker_info:ro" \
		$(IMAGE_NAME):$(TAG)
