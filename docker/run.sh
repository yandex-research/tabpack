#!/usr/bin/env bash
# Run a command in the TabPack image with the repo mounted at /workspace.
#
#   docker/run.sh                                   # interactive shell
#   docker/run.sh python -m common.run ...          # one command
#   docker/run.sh --root ncu ... python -m ...      # ncu needs root + SYS_ADMIN when the host driver restricts profiling to admins
#
# GPUs: set CUDA_VISIBLE_DEVICES on the host (indices or UUIDs); default is all GPUs.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-tabpack:cu132}"

user_args=(--user "$(id -u):$(id -g)")
if [ "${1:-}" = "--root" ]; then
    shift
    user_args=(-e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)")
fi

gpus="all"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    gpus="\"device=${CUDA_VISIBLE_DEVICES}\""
fi

# Network: host networking + the same proxy as the docker daemon (if any), so that
# `uv pip install ...` etc. work inside the container on proxied hosts.
net_args=(--network=host)
proxy="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}"
if [ -z "$proxy" ]; then
    proxy="$(docker info 2>/dev/null | sed -n 's/^ *HTTPS\? Proxy: *//p' | head -1)"
fi
if [ -n "$proxy" ]; then
    for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do net_args+=(-e "$v=$proxy"); done
fi

tty_args=()
if [ -t 0 ] && [ -t 1 ]; then tty_args=(-it); fi
if [ $# -eq 0 ]; then set -- bash; fi

mkdir -p "$REPO/.cache"
exec docker run --rm "${tty_args[@]}" "${user_args[@]}" "${net_args[@]}" \
    --gpus "$gpus" --ipc=host --cap-add=SYS_ADMIN \
    -v "$REPO":/workspace -w /workspace \
    "$IMAGE" "$@"
