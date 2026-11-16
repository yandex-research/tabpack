# TabPack dev/profiling image.
# One image for A4000 (sm_86), Hopper (sm_90), Blackwell DC (sm_100) and RTX Blackwell (sm_120).
# Requires host NVIDIA driver >= 580 (CUDA 13.x).
#
# Build:  docker/build.sh            Run:  docker/run.sh <cmd>
# The repo is NOT copied into the image: it is mounted at /workspace at run time and put
# on PYTHONPATH, so edits on the host are picked up without rebuilding.

ARG CUDA_IMAGE=nvidia/cuda:13.2.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_APT_VER=13-2

# System deps + Nsight Systems / Nsight Compute CLIs (from the CUDA apt repo that the base
# image already configures).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        git curl ca-certificates build-essential ninja-build \
        cuda-nsight-systems-${CUDA_APT_VER} cuda-nsight-compute-${CUDA_APT_VER} \
    && rm -rf /var/lib/apt/lists/* \
    && for tool in nsys ncu; do \
         if ! command -v "$tool" >/dev/null; then \
           p="$(find /usr/local/cuda* /opt/nvidia -name "$tool" -type f -perm -u+x 2>/dev/null | grep -v host- | head -1)"; \
           [ -n "$p" ] && ln -sf "$p" "/usr/local/bin/$tool"; \
         fi; \
       done \
    && nsys --version && ncu --version

# Python env (uv only for fast installs).
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
RUN uv venv /opt/venv --python /usr/bin/python3.12

COPY docker/requirements.txt /tmp/requirements.txt
RUN uv pip install --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu132 \
        -r /tmp/requirements.txt \
    && python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"

# Dev tools (ONNX export in examples/, ruff/ty lint) in their own layer.
COPY docker/requirements-dev.txt /tmp/requirements-dev.txt
RUN uv pip install -r /tmp/requirements-dev.txt \
    && uv pip freeze > /opt/pip-freeze.txt

# Arch list for any CUDA/C++ extension built inside the container (Triton JITs at runtime
# and needs nothing). +PTX keeps forward compatibility with future GPUs.
ENV TORCH_CUDA_ARCH_LIST="8.6;9.0;10.0;12.0+PTX" \
    TRITON_DEFAULT_FP_FUSION=1 \
    PYTHONPATH=/workspace/src:/workspace/dev/pepilica/src:/workspace \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp/home \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor \
    RUFF_CACHE_DIR=/workspace/.cache/ruff

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh && mkdir -p /tmp/home && chmod 1777 /tmp/home

# Dev-container support:
# - sudo: the image's `ubuntu` user (UID 1000, remapped to the host UID by dev containers)
#   gets passwordless sudo inside the container, e.g. `sudo -E ncu ...` on hosts that
#   restrict GPU profiling counters to root. Unrelated to sudo on the host.
# - openssh-client: git over SSH (agent forwarding) from inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends sudo openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
