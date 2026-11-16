#!/usr/bin/env bash
# Build the TabPack image. Usage: docker/build.sh [extra docker build args]
# Override the tag with IMAGE=..., e.g. IMAGE=abusagit/tabpack:cu132 docker/build.sh
#
# Behind a proxy: the proxy is taken from $HTTPS_PROXY/$HTTP_PROXY or, if unset, from the
# docker daemon config (`docker info`), and forwarded to the build steps.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-tabpack:cu132}"

proxy="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}"
if [ -z "$proxy" ]; then
    proxy="$(docker info 2>/dev/null | sed -n 's/^ *HTTPS\? Proxy: *//p' | head -1)"
fi
proxy_args=()
if [ -n "$proxy" ]; then
    echo "Using proxy for build steps: $proxy"
    for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do
        proxy_args+=(--build-arg "$v=$proxy")
    done
fi

# Pre-pull bases through the daemon (BuildKit's own registry auth may bypass the proxy).
base="$(sed -n 's/^ARG CUDA_IMAGE=//p' docker/Dockerfile)"
docker image inspect "$base" >/dev/null 2>&1 || docker pull "$base"

docker build --network=host "${proxy_args[@]}" -f docker/Dockerfile -t "$IMAGE" "$@" .
echo "Built $IMAGE"
