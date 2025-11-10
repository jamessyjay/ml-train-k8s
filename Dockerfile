# ===== builder =====
FROM nvidia/cuda:12.6.1-devel-ubuntu24.04 AS builder
ARG PYVER=3.12
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
      python${PYVER} python${PYVER}-venv python${PYVER}-dev \
      build-essential git cmake ninja-build \
  && ln -s /usr/bin/python${PYVER} /usr/bin/python \
  && python -m venv /venv \
  && . /venv/bin/activate && python -m pip install --upgrade pip

# Torch CUDA12.6 + стек HF
RUN . /venv/bin/activate && \
    pip install --index-url https://download.pytorch.org/whl/cu126 \
      "torch==2.6.*" "torchvision==0.21.*" "torchaudio==2.6.*" && \
    pip install \
      "transformers==4.45.*" \
      "peft>=0.13" \
      "accelerate>=0.34" \
      "datasets>=2.19" \
      "sentencepiece" \
      "protobuf<6"

# Sanity-check
RUN . /venv/bin/activate && python - <<'PY'
import sys
import torch, transformers, peft
print("[builder] torch:", torch.__version__)
print("[builder] transformers:", transformers.__version__)
print("[builder] peft:", peft.__version__)
PY

# Очистка
RUN . /venv/bin/activate && \
    find /venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find /venv -type f -name "*.pyc" -delete && \
    (command -v strip >/dev/null 2>&1 && find /venv -type f -name "*.so*" -exec strip --strip-unneeded {} + || true)

# ===== runtime =====
FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04
ARG PYVER=3.12
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python${PYVER} python${PYVER}-venv \
  && ln -s /usr/bin/python${PYVER} /usr/bin/python \
  && rm -rf /var/lib/apt/lists/*

ENTRYPOINT []

COPY --from=builder /venv /opt/venv
RUN ln -s /opt/venv /venv

WORKDIR /app
COPY src/ ./src/

# Light preflight on start
CMD ["bash","-lc","python - <<'PY'\nimport torch, transformers;print('OK', torch.__version__, transformers.__version__)\nPY\n"]