# ===== builder =====
FROM nvidia/cuda:12.6.1-devel-ubuntu24.04 AS builder
ARG PYVER=3.12
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# Build tools: cmake/ninja are mandatory for flash-attn
RUN apt-get update && apt-get install -y \
      python${PYVER} python${PYVER}-venv python${PYVER}-dev \
      build-essential git cmake ninja-build \
  && ln -s /usr/bin/python${PYVER} /usr/bin/python \
  && python -m venv /venv \
  && . /venv/bin/activate && python -m pip install --upgrade pip

# Torch with CUDA 12.6 + dependencies
RUN . /venv/bin/activate && \
    pip install --index-url https://download.pytorch.org/whl/cu126 \
      "torch==2.6.*" "torchvision==0.21.*" "torchaudio==2.6.*" && \
    pip install pydantic==2.9.2 torchmetrics==1.4.0.post0 peft && \
# ===== builder =====
    python - <<'PY'
import sys
try:
    import torch, transformers  # verify install
    print("[builder] imports ok:", torch.__version__, transformers.__version__)
except Exception as e:
    print("[builder] import failed:", e, file=sys.stderr)
    sys.exit(1)
PY

# Clean up inside venv
RUN . /venv/bin/activate && \
    find /venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find /venv -type f -name "*.pyc" -delete && \
    (command -v strip >/dev/null 2>&1 && find /venv -type f -name "*.so*" -exec strip --strip-unneeded {} + || true)

# ===== runtime =====
FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04
ARG PYVER=3.12
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python${PYVER} python${PYVER}-venv \
  && ln -s /usr/bin/python${PYVER} /usr/bin/python \
  && rm -rf /var/lib/apt/lists/*

# Disable default NVIDIA entrypoint banner/license notice
ENTRYPOINT []

COPY --from=builder /venv /opt/venv
RUN ln -s /opt/venv /venv

WORKDIR /app
COPY src/ ./src/

CMD ["python", "/app/src/train.py"]
