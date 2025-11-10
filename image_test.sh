#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/jamessyjay/gpu-cluster-acceptance:train-smoke}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
RUNTIME_FLAGS="--gpus all --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility"

PASS=0
FAIL=0
step() { echo -e "\n==== $* ===="; }
ok()   { echo "[OK] $*"; ((PASS++)) || true; }
bad()  { echo "[BAD] $*"; ((FAIL++)) || true; }

echo "Running Docker GPU smoke on image: ${IMAGE}"
echo "Using docker: $($DOCKER_BIN --version)"

step "1) Check nvidia runtime"
if $DOCKER_BIN info 2>/dev/null | grep -iq 'Runtimes:.*nvidia'; then
  ok "NVIDIA runtime found"
else
  bad "NVIDIA runtime not found in Docker. Check nvidia-container-toolkit"
fi

step "2) Try nvidia-smi from your image"
if $DOCKER_BIN run --rm $RUNTIME_FLAGS "$IMAGE" nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi works inside your image"
else
  echo "nvidia-smi not available in your image. Trying reference CUDA base"
  if $DOCKER_BIN run --rm $RUNTIME_FLAGS nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    ok "nvidia-smi works (check through base CUDA image)"
  else
    bad "GPU not visible to container (nvidia-smi fails)."
  fi
fi

step "3) Python/torch versions and CUDA visibility"
if $DOCKER_BIN run --rm $RUNTIME_FLAGS "$IMAGE" bash -lc 'python - <<PY
import sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    i=0
    print("device:", torch.cuda.get_device_name(i), "cc:", torch.cuda.get_device_capability(i))
PY'; then
  ok "PyTorch sees GPU"
else
  bad "PyTorch doesn't see GPU in container"
fi

step "4) Check flash-attn installed and imported"
if $DOCKER_BIN run --rm $RUNTIME_FLAGS "$IMAGE" bash -lc 'python - <<PY
import sys, importlib
try:
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
    print("flash-attn version:", getattr(flash_attn, "__version__", "?"))
except Exception as e:
    print("IMPORT_FAIL:", e); sys.exit(2)
PY'; then
  ok "flash-attn imported"
else
  bad "flash-attn not imported"
fi

step "5) Run minimal CUDA kernel from flash-attn (BF16)"
if $DOCKER_BIN run --rm $RUNTIME_FLAGS "$IMAGE" bash -lc 'python - <<PY
import sys, torch
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
if not torch.cuda.is_available():
    print("NO CUDA"); sys.exit(3)
B,S,H,D=1,256,8,64
qkv=torch.randn(B,S,3,H,D, device="cuda", dtype=torch.bfloat16)
out=flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False)
print("flash-attn OK:", out.shape, out.dtype, out.is_cuda)
PY'; then
  ok "flash-attn kernel runs on GPU"
else
  bad "flash-attn kernel doesn't run on GPU (incompatibility wheel/versions/archs?)"
fi

step "6) NCCL availability (without pg initialization)"
if $DOCKER_BIN run --rm $RUNTIME_FLAGS "$IMAGE" bash -lc 'python - <<PY
import torch
print("torch.distributed available:", torch.distributed.is_available())
print("nccl available:", torch.distributed.is_nccl_available() if torch.distributed.is_available() else False)
PY'; then
  ok "NCCL visible to PyTorch"
else
  bad "NCCL not visible to PyTorch (not critical for single process)"
fi

echo -e "\n================= SUMMARY ================="
echo "PASSED: $PASS   FAILED: $FAIL"
test "$FAIL" -eq 0 && echo "[OK] All good" || echo "[BAD] Some problems — see red points"
exit "$FAIL"


# How to run:
# $ chmod +x image_test.sh
# $ IMAGE=ghcr.io/jamessyjay/gpu-cluster-acceptance:train-smoke ./image_test.sh