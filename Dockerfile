ARG BASE_IMAGE=ghcr.io/jamessyjay/gpu-cluster-acceptance:gpu
FROM ${BASE_IMAGE}

WORKDIR /app

# Basic envs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy only project sources
COPY src/ /app/src/

# Port for torchrun rendezvous (optional)
EXPOSE 29500

# Default command; can be overridden by K8s manifest
CMD ["python", "/app/src/train.py"]
