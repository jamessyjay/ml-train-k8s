# syntax=docker/dockerfile:1.7-labs
ARG BASE_IMAGE=ghcr.io/jamessyjay/gpu-cluster-acceptance:gpu

############################
# STAGE 1: build wheel
############################
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS fa-build
ENV DEBIAN_FRONTEND=noninteractive MAMBA_ROOT_PREFIX=/opt/conda
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends curl bzip2 ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# micromamba (минимально)
ARG MAMBA_VERSION=1.5.10
RUN curl -L -o /tmp/micromamba.tar.bz2 \
      https://micromamba.snakepit.net/api/micromamba/linux-64/${MAMBA_VERSION} \
 && mkdir -p ${MAMBA_ROOT_PREFIX}/bin \
 && tar -xjf /tmp/micromamba.tar.bz2 -C ${MAMBA_ROOT_PREFIX}/bin --strip-components=1 bin/micromamba \
 && rm -f /tmp/micromamba.tar.bz2

SHELL ["/bin/bash","-lc"]

# ВАЖНО: повторяем стек как в твоей базе: py311 + torch 2.5.1 + CUDA 12.4
RUN ${MAMBA_ROOT_PREFIX}/bin/micromamba create -y -n buildapp \
      -c pytorch -c nvidia -c conda-forge \
      python=3.11 pytorch=2.5.1 pytorch-cuda=12.4 && \
    ${MAMBA_ROOT_PREFIX}/bin/micromamba run -n buildapp \
      python -m pip install -U pip wheel setuptools

# Сборка flash-attn в wheel (без установки)
ENV TORCH_CUDA_ARCH_LIST="80;86;89;90" MAX_JOBS=4
RUN ${MAMBA_ROOT_PREFIX}/bin/micromamba run -n buildapp \
      pip wheel --no-build-isolation --wheel-dir /wheels "flash-attn==2.6.*"

############################
# STAGE 2: extend your base
############################
FROM ${BASE_IMAGE} AS final
# ставим колесо в уже существующий conda env "app"
COPY --from=fa-build /wheels /tmp/wheels
RUN /opt/conda/envs/app/bin/pip install --no-cache-dir /tmp/wheels/flash_attn-*.whl && \
    rm -rf /tmp/wheels

WORKDIR /app
COPY src/ /app/src/
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src
EXPOSE 29500
CMD ["/opt/conda/envs/app/bin/python","/app/src/train.py"]