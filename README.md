<<<<<<< HEAD
# ml-images-docker-k8s
Collection of different kinds of Machine Learning templates, useful automation images for docker and k8s.
=======
# k8s-training/k8s — Training Guide

This guide covers two entry points:

- train_framework.py — LoRA fine-tuning framework for HF Transformers models (JSONL datasets, DDP via torchrun, offline/online).
- train.py — a minimal DDP smoke test on synthetic data (quickly validates CUDA/NCCL/DDP setup).


# Organization Guidelines (Source, Checkpoints, Topology, Runtime, Inference)

## Model source of truth
- Online (pull once, cache globally):
  - Set caches to shared storage
    ```bash
    export HF_HOME=/mnt/filesystem-o2/.cache/hf
    export TRANSFORMERS_CACHE=/mnt/filesystem-o2/.cache/hf
    ```
  - First run pulls from HF into the shared cache, subsequent runs read from it.
- Offline/local (pre-fetched on a jumper):
  - Place snapshot under
    ```
    /mnt/filesystem-o2/models/Qwen2.5-3B-Instruct/
    ```
  - Train with:
    ```bash
    --model-local-dir /mnt/filesystem-o2/models/Qwen2.5-3B-Instruct \
    --tokenizer-local-dir /mnt/filesystem-o2/models/Qwen2.5-3B-Instruct \
    --local-files-only
    ```

## Best model tracking (validation + checkpoints)
- Prepare a validation JSONL split (e.g., `/mnt/filesystem-o2/datasets/val.jsonl`).
- Save checkpoints per epoch and rank:
  ```
  /mnt/filesystem-o2/checkpoints/<run>/epoch-*/rank-*/
  ```
- After each epoch:
  - run validation; compute val loss/perplexity
  - keep best as a symlink/copy at `/mnt/filesystem-o2/checkpoints/<run>/best`.
- Track logs & metrics in MLflow:
  ```bash
  export MLFLOW_TRACKING_URI=/mnt/filesystem-o2/mlruns
  # log metrics/params/artifacts via mlflow in training loop (optional)
  ```

## Cluster topology for distributed runs
- K8s StatefulSet with `replicas=4`, one GPU per Pod.
- Pod anti-affinity + topology spread to land each Pod on a different node.
- Headless Service for rendezvous; use `pod-0` as rendezvous target.
- Launch command across nodes:
  ```bash
  torchrun --nnodes=4 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=<headless-svc>:29500 \
    k8s-training/k8s/train_framework.py ...
  ```

## Runtime settings (recommended)
```bash
export HF_HOME=/mnt/filesystem-o2/.cache/hf
export TRANSFORMERS_CACHE=/mnt/filesystem-o2/.cache/hf
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
# data/checkpoints roots
DATA_ROOT=/mnt/filesystem-o2/datasets
CKPT_ROOT=/mnt/filesystem-o2/checkpoints
```

## Inference after training
- Option A: Base model + LoRA adapters
  - Serve with vLLM/TGI that supports PEFT adapters.
- Option B: Merge LoRA into a single weight
  - Produce merged weights and deploy as a standard model via Nebius ml-containers.

---

## 1) train_framework.py (LoRA fine-tuning for LLMs)

### What it does
- Fine-tunes a Causal LM with LoRA (PEFT).
- Runs single-GPU or multi-GPU (DDP via torchrun).
- Accepts simple JSONL datasets (messages, prompt+completion, or text).
- Dynamic padding, gradient accumulation, linear warmup.
- Works online (Hugging Face) or fully offline from a local model directory.
- Rich logging with adjustable verbosity and batch-level debugging.

### Requirements
- Python 3.9+ (or 3.10+)
- GPU with recent CUDA + PyTorch with CUDA
- Packages: transformers, peft, torch (and optionally huggingface_hub for online download)

### Dataset format (JSONL)
Each line is a JSON object. Supported shapes:

- messages
  ```json
  {"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"World"}]}
  ```
- prompt + completion
  ```json
  {"prompt":"Hello","completion":" World"}
  ```
- text
  ```json
  {"text":"Hello World"}
  ```

Quick sample:
```bash
cat >/tmp/ds.jsonl <<EOF
{"prompt":"Hello","completion":" World"}
{"messages":[{"role":"user","content":"Ping"},{"role":"assistant","content":"Pong"}]}
EOF
```

### Key CLI arguments
- Model and data
  - --model <repo_or_name> (e.g. Qwen/Qwen2.5-3B-Instruct)
  - --data /path/to/data.jsonl
  - --output /path/to/output_dir
- Training
  - --epochs INT, --seq-len INT
  - --batch INT (per GPU), --grad-accum INT
  - --bf16 (enable bfloat16 autocast)
- LoRA
  - --lora (enable LoRA)
  - --lora-r INT, --lora-alpha INT, --lora-dropout FLOAT
  - --target-modules q_proj k_proj v_proj o_proj (defaults)
- Logging and debugging
  - --log-level DEBUG|INFO|WARNING|ERROR (default: INFO)
  - --log-batches all|first-last|none (default: first-last)
- Offline/local model
  - --model-local-dir /path/to/local_model
  - --tokenizer-local-dir /path/to/local_tokenizer (optional; defaults to model path)
  - --local-files-only (disable network; load only from disk)

### Quick start (single GPU, open model)
```bash
python k8s-training/k8s/train_framework.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data /tmp/ds.jsonl \
  --output /tmp/out \
  --epochs 1 --seq-len 512 --batch 1 --grad-accum 1 \
  --log-level DEBUG --log-batches all
```

### Online use with gated models (Hugging Face)
If your model is gated (e.g., meta-llama/*):
1) Accept license/terms on the model page.
2) Create an access token: https://huggingface.co/settings/tokens
3) Authenticate:
   - CLI: `pip install huggingface_hub && huggingface-cli login --token <HF_TOKEN>`
   - or ENV: `export HUGGINGFACE_HUB_TOKEN=<HF_TOKEN>` (optionally `export HF_TOKEN=$HUGGINGFACE_HUB_TOKEN`)

Then run normally with `--model <gated_model_id>`.

### Fully offline from a local directory
Prepare a local model folder in HF format (e.g., copied from another machine):
- config.json
- model.safetensors (or sharded files model-00001-of-000xx.safetensors)
- tokenizer.json / tokenizer.model / tokenizer_config.json
- special_tokens_map.json / vocab.json / merges.txt (depending on model)
- generation_config.json (optional)

Run offline:
```bash
export TRANSFORMERS_OFFLINE=1
python k8s-training/k8s/train_framework.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --model-local-dir /models/Qwen2.5-3B-Instruct \
  --tokenizer-local-dir /models/Qwen2.5-3B-Instruct \
  --local-files-only \
  --data /tmp/ds.jsonl \
  --output /tmp/out \
  --lora --bf16 --epochs 1 --seq-len 512 --batch 1 --grad-accum 1 \
  --log-level DEBUG --log-batches all
```

### Multi-GPU (DDP via torchrun)
```bash
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
  k8s-training/k8s/train_framework.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data /tmp/ds.jsonl \
  --output /tmp/out \
  --epochs 1 --seq-len 512 --batch 1 --grad-accum 1 \
  --log-level INFO --log-batches first-last
```

Notes:
- `--batch` is per-rank (per GPU).
- Use `sampler.set_epoch(epoch)` (already handled) for proper shuffling.

### Kubernetes (high level sketch)
- Container command: torchrun … train_framework.py …
- Mount volumes for data/model/output.
- Set GPU resources.
- Optionally propagate env (e.g., HUGGINGFACE_HUB_TOKEN) if using online loading.
- See train-ddp.yaml for a reference skeleton and adapt to your cluster.

### Logging: what to expect
- INFO level:
  - [STEP] pipeline steps (parse args, init DDP, build loader/model, start training, finalize)
  - [DDP] device binding, process group init/destroy
  - [DATA] dataset size, sampler usage, number of batches
  - [MODEL] base params, optimizer, scheduler
  - [TRAIN] tokens/s periodically
  - [EPOCH] per-epoch summary
  - [RESULT]/[DONE] final artifacts and throughput
- DEBUG level adds:
  - [COLLATE] shapes of input_ids/attention_mask
  - [DATA] sampler details, DataLoader settings
  - [MODEL] trainable parameter counts
  - [BATCH] batch-level loss/time (configurable with --log-batches)

### Troubleshooting
- 401 Gated repo / Unauthorized
  - Accept license on HF model page, login with token, or use offline local model.
- CUDA OOM
  - Reduce --batch or --seq-len; increase --grad-accum.
- DDP hangs
  - Ensure identical env per rank and proper torchrun args; NCCL must be available; network ok.
- Missing local files (offline)
  - Ensure required files listed above exist in model directory.

---

## 2) train.py (synthetic DDP smoke test)

### What it does
- Generates a synthetic, linearly-separable dataset.
- Trains a tiny MLP to verify that CUDA, NCCL, and DDP are functioning.
- Logs step-by-step initialization and per-epoch metrics.

### CLI (key)
- --epochs INT (default 2)
- --batch INT (per process, default 256)
- --samples INT (default 4096)
- --features INT (default 128)
- --classes INT (default 4)
- --lr FLOAT (default 3e-3)
- --seed INT (default 42)

### Single-GPU example
```bash
python k8s-training/k8s/train.py \
  --epochs 2 --batch 256 --samples 4096 --features 128 --classes 4 --lr 3e-3 --seed 42
```

### Multi-GPU (DDP via torchrun)
```bash
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
  k8s-training/k8s/train.py \
  --epochs 2 --batch 256 --samples 4096 --features 128 --classes 4 --lr 3e-3 --seed 42
```

### What to look for
- [STEP] init, seed, loader/model/DDP
- [TRAIN] per-epoch loss
- [BATCH] (DEBUG) first/last batch loss and tensor shapes
- [RESULT] sanity check of improvement and timing

### Common issues
- NCCL init failure: verify CUDA/NCCL availability and correct torchrun setup.
- Rank/GPU mismatch: torchrun provides LOCAL_RANK/… env; avoid overriding.

---

## Handy snippets

- Minimal framework smoke
  ```bash
  python k8s-training/k8s/train_framework.py --model Qwen/Qwen2.5-3B-Instruct --data /tmp/ds.jsonl --output /tmp/out \
    --epochs 1 --seq-len 512 --batch 1 --grad-accum 1
  ```
- Maximum debug
  ```bash
  python k8s-training/k8s/train_framework.py --model Qwen/Qwen2.5-3B-Instruct --data /tmp/ds.jsonl --output /tmp/out \
    --epochs 1 --seq-len 512 --batch 1 --grad-accum 1 --log-level DEBUG --log-batches all
  ```
- Fully offline
  ```bash
  TRANSFORMERS_OFFLINE=1 python k8s-training/k8s/train_framework.py --model <name> \
    --model-local-dir /models/<name> --tokenizer-local-dir /models/<name> --local-files-only \
    --data /tmp/ds.jsonl --output /tmp/out --lora --bf16 --epochs 1 --seq-len 512 --batch 1 --grad-accum 1
  ```
- DDP
  ```bash
  torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    k8s-training/k8s/train_framework.py --model Qwen/Qwen2.5-3B-Instruct --data /tmp/ds.jsonl --output /tmp/out \
    --epochs 1 --seq-len 512 --batch 1 --grad-accum 1
  ```

---

## Notes
- Batch size flags are per-rank (per GPU). Effective global batch = batch * world_size * grad_accum.
- BF16 works best on recent GPUs (A100/H100). If unstable, try FP16 (omit --bf16).
- For K8s, ensure you mount the model/data/output volumes and set GPU resources.

---

# Environment Design (Actionable)

This section is a prescriptive setup you can hand to ops/ML engineers.

## Single-node, multi-GPU (baseline)
- Hardware: 1 node with N GPUs (e.g., 4×A100)
  - /models (optional; local model snapshots)
  - /data (JSONL)
  - /out (training outputs)
- Run with torchrun:
  ```bash
  torchrun --nproc_per_node=${NUM_GPUS} --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    k8s-training/k8s/train_framework.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data /data/dataset.jsonl --output /out/run-$(date +%Y%m%d-%H%M) \
    --epochs 1 --seq-len 2048 --batch 2 --grad-accum 8 \
    --log-level INFO --log-batches first-last
  ```

## Multi-node, multi-GPU (DDP)
- One rank per GPU; torchrun across nodes.
- Required env per node:
  - MASTER_ADDR=<ip-or-host-of-rank0>
  - MASTER_PORT=29500
  - NODE_RANK=0..(num_nodes-1)
  - WORLD_SIZE=num_nodes * gpus_per_node
- Launcher (on each node):
  ```bash
  torchrun --nnodes=${NNODES} --node_rank=${NODE_RANK} \
    --nproc_per_node=${NGPU} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    k8s-training/k8s/train_framework.py ...
  ```
- NCCL tips (often helpful):
  ```bash
  export NCCL_DEBUG=INFO
  export NCCL_SOCKET_IFNAME=eth0   # or your data interface
  export NCCL_IB_HCA=^mlx5_1       # if you need to exclude/choose IB cards
  ```

## Kubernetes reference design
- Pod spec:
  - resources: requests/limits for nvidia.com/gpu: N
  - volumes:
    - pvc:/data (dataset JSONL)
    - pvc:/out (outputs)
    - (optional) pvc:/models (offline models)
  - env:
    - HUGGINGFACE_HUB_TOKEN (Secret) if using gated models online
    - TRANSFORMERS_OFFLINE=1 when fully offline
    - NCCL_DEBUG=INFO (debugging)
  - command:
    ```bash
    torchrun --nproc_per_node=$NPROC --rdzv_backend=c10d --rdzv_endpoint=$HOST_IP:29500 \
      k8s-training/k8s/train_framework.py --model <id-or-local> --data /data/ds.jsonl \
      --output /out/<run-id> --epochs 1 --seq-len 4096 --batch 2 --grad-accum 8
    ```
- Multi-node on K8s:
  - Use a StatefulSet or Job with stable hostnames; pick rank0 Pod as rendezvous host.
  - Headless Service to resolve MASTER_ADDR.
  - Set NODE_RANK per Pod index.

---

# Models: How to Plug Them In

## Open models from HF (no auth)
```bash
--model Qwen/Qwen2.5-3B-Instruct
```
Use as-is. No token needed.

## Gated models from HF (auth needed)
```bash
export HUGGINGFACE_HUB_TOKEN=<token>
--model meta-llama/Llama-3.2-3B-Instruct
```
Prereq: accept model license; ensure token in env or via `huggingface-cli login`.

## Offline local directory (no network)
```bash
export TRANSFORMERS_OFFLINE=1
--model <anything> \
--model-local-dir /models/llama \
--tokenizer-local-dir /models/llama \
--local-files-only
```
Folder must contain HF files (config.json, model.safetensors, tokenizer.*, etc.).

## Customizing LoRA
- Enable: `--lora`
- R/alpha/dropout: `--lora-r 16 --lora-alpha 16 --lora-dropout 0.05`
- Target modules:
  ```bash
  --target-modules q_proj k_proj v_proj o_proj
  # or for different architectures adjust to module names present in the model
  ```

---

# Outputs: What Gets Saved and Where

## Default outputs (to --output)
- Model adapters/weights (safe_serialization where applicable)
- Tokenizer files
- JSON report (if `--report-json` provided) with:
  - world_size, total tokens, elapsed seconds, tokens/sec

## Recommended run directory convention
```bash
--output /out/${MODEL_SHORT}/lora-r${R}-a${ALPHA}-len${LEN}-$(date +%Y%m%d-%H%M)
```
Keeps runs tidy and sortable.

## Persisting to object storage (patterns)
- Option 1: mount S3/GCS bucket via CSI driver to `/out` and write directly
- Option 2: write to local `/out` then sync:
  ```bash
  s5cmd cp -r /out s3://my-bucket/experiments/
  # or: aws s3 sync /out s3://bucket/path
  ```

---

# Datasets: Where to Get and How to Feed

## Hand-crafted JSONL (quick)
```bash
cat > /data/ds.jsonl <<EOF
{"prompt":"Hello","completion":" World"}
{"messages":[{"role":"user","content":"Ping"},{"role":"assistant","content":"Pong"}]}
EOF
```

## From Hugging Face datasets (export to JSONL)
Python snippet:
```python
from datasets import load_dataset
import json

ds = load_dataset("OpenAssistant/oasst1", split="train[:1%]")
with open("/data/oasst.jsonl", "w", encoding="utf-8") as f:
    for ex in ds:
        messages = ex.get("messages") or ex.get("conversations")
        if messages:
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
```

## Converting generic CSV to JSONL (prompt+completion)
```bash
python - <<'PY'
import csv, json, sys
inp, out = sys.argv[1], sys.argv[2]
with open(inp, newline='', encoding='utf-8') as f, open(out, 'w', encoding='utf-8') as g:
    r = csv.DictReader(f)
    for row in r:
        prompt = row.get('prompt') or row.get('input') or ''
        completion = row.get('completion') or row.get('output') or ''
        g.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
PY
# usage: python script.py data.csv /data/ds.jsonl
```

## Validating the JSONL
```bash
python - <<'PY'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    for i,line in enumerate(f,1):
        try:
            json.loads(line)
        except Exception as e:
            print(f"Bad JSON at line {i}: {e}")
            sys.exit(1)
print("OK")
PY
# usage: python script.py /data/ds.jsonl
```

---

# Minimal, Focused Use-cases

## (A) Quick sanity on one GPU (open model)
```bash
python k8s-training/k8s/train_framework.py \
  --model distilgpt2 --data /data/ds.jsonl --output /out/run \
  --epochs 1 --seq-len 1024 --batch 2 --grad-accum 8
```

## (B) Full debug in container
```bash
python k8s-training/k8s/train_framework.py \
  --model distilgpt2 --data /data/ds.jsonl --output /out/run \
  --epochs 1 --seq-len 1024 --batch 1 --grad-accum 1 \
  --log-level DEBUG --log-batches all
```

## (C) Offline run with local LLaMA snapshot
```bash
export TRANSFORMERS_OFFLINE=1
python k8s-training/k8s/train_framework.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --model-local-dir /models/llama3.2-3b \
  --tokenizer-local-dir /models/llama3.2-3b \
  --local-files-only \
  --data /data/ds.jsonl --output /out/run --lora --bf16 \
  --epochs 1 --seq-len 4096 --batch 2 --grad-accum 8
```

## (D) DDP across 4 GPUs on one node
```bash
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
  k8s-training/k8s/train_framework.py \
  --model distilgpt2 --data /data/ds.jsonl --output /out/run \
  --epochs 1 --seq-len 2048 --batch 2 --grad-accum 8
```

---

# Appendix: train.py quick-reference
- Use this to validate CUDA/NCCL/DDP; not for real LLM training.
```bash
python k8s-training/k8s/train.py \
  --epochs 2 --batch 256 --samples 4096 --features 128 --classes 4 --lr 3e-3 --seed 42

```

>>>>>>> 6bd8a6c (Initial commit)
