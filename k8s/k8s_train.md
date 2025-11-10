# Kubernetes Training Guide (torchrun + DDP)

Practical recipes to run `train_framework.py` and `train.py` on Kubernetes with GPUs.

Current implementation is for one model training distributedly, all pods — participants of one torchrun world.
We have this pattern: nnodes=$NNODES, nproc_per_node=$NPROC_PER_NODE, RANK = pod index. 
This is 1 model on all pods. We just move this to Job and make rendezvous through etcd.

## Preparation (Before you run)
- **Image**
  - Build/push your image that contains the project (Dockerfile provided) so the script path inside the container is `/app/src/train_framework.py`.
  - If your registry is private, create an `imagePullSecret` and reference it in the StatefulSet.
- **Headless Service (rendezvous)**
  - Create a headless Service named `train-rdzv` (no ClusterIP) with port `29500` used by `torchrun` rendezvous.
- **StatefulSet**
  - Use `replicas = NNODES` (one pod per GPU is simplest; set `nvidia.com/gpu: 1`).
  - Derive `RANK` from the pod ordinal or `apps.kubernetes.io/pod-index`.
  - Set env: `NNODES`, `NPROC_PER_NODE=1`, `MASTER_ADDR=train-rdzv`, `MASTER_PORT=29500`.
- **Volumes**
  - PVCs you need to have beforehand:
    - `datasets-pvc` → `/data`
    - `outputs-pvc`  → `/out`
    - `models-pvc`   → `/models` (optional; for offline/local models)
- **Secret (optional)**
  - `hf-token` with key `HUGGINGFACE_HUB_TOKEN` if you train from gated HF models.
- **DDP/NCCL env quick list**
  - `NNODES`, `RANK` (pod index), `NPROC_PER_NODE=1`, `MASTER_ADDR=train-rdzv`, `MASTER_PORT=29500`
  - `NCCL_DEBUG=INFO`, `NCCL_SOCKET_IFNAME=eth0` (or your interface), `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
- **Offline training (optional)**
  - Mount model snapshot to `/models/<name>` and run with `--model-local-dir /models/<name> --tokenizer-local-dir /models/<name> --local-files-only`.

### Scaling
	•	Now: 4 nodes × 1 GPU → NNODES=4, NPROC_PER_NODE=1 → WORLD_SIZE=4.
	•	Later: 64 nodes × 8 GPUs → NNODES=64, NPROC_PER_NODE=8 → WORLD_SIZE=512.

### Quick deploy
  - Save the Service and StatefulSet manifests and apply:
    ```bash
    kubectl apply -f train-rdzv.yaml
    kubectl apply -f stateful-set.yaml
    ```
  - Watch and view logs:
    ```bash
    kubectl get pods -l app=train-llm -w
    kubectl logs -f train-llm-0 -c trainer
    ```
  - Scale (remember to sync `NNODES` env with `replicas`):
    ```bash
    kubectl scale statefulset/train-llm --replicas=3
    # then update the StatefulSet env NNODES="3" and re-apply
    ```
  - Re-create StatefulSet:
    ```bash
    # gracefully stop sts (statefulset)
    kubectl -n train scale sts/train-llm --replicas=0
    kubectl -n train rollout status sts/train-llm --timeout=10m

    # delete only sts (pods are already gone)
    kubectl -n train delete sts train-llm

    # create again with your stateful-set.yaml
    kubectl -n train apply -f stateful-set.yaml
    ```


## TL;DR
- One pod per GPU is simplest. Use StatefulSet with `replicas = NUM_GPUS`.
- Rendezvous via Headless Service and `pod-0`.
- Mount three volumes: `/data` (datasets), `/out` (outputs), `/models` (optional offline models).
- Use `torchrun --nnodes=<pods> --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=<headless-svc>:29500`.

---

## 1) Objects overview
- Namespace: your project (e.g., `ml-train`)
- PVCs:
  - `datasets-pvc` → mounted at `/data`
  - `outputs-pvc` → mounted at `/out`
  - `models-pvc`  → mounted at `/models` (optional, for offline local models)
- Secret (optional): `hf-token` with key `HUGGINGFACE_HUB_TOKEN`
- Headless Service: `train-rdzv` (no ClusterIP)
- StatefulSet: `train-llm` with `replicas = N`, 1 GPU per pod

---

## 2) Headless Service (rendezvous)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: train-rdzv
  labels:
    app: train-llm
spec:
  clusterIP: None   # headless
  selector:
    app: train-llm
  ports:
    - name: rdzv
      port: 29500
      targetPort: 29500
```

---

## 3) StatefulSet (1 GPU per Pod)
see `stateful-set.yaml`
Notes:
- On some clusters, `apps.kubernetes.io/pod-index` is provided automatically for StatefulSets. If not, you can derive rank via hostname (`${HOSTNAME##*-}`) and set `NNODES` via an env or ConfigMap.
- Use topology spread constraints if your cluster supports them for even distribution.

---

## 4) Online vs Offline models
- Online (gated): add Secret `hf-token` and set `HUGGINGFACE_HUB_TOKEN`. Accept license first.
- Offline: mount `/models/<model>` and run with:
  ```bash
  --model-local-dir /models/<model> --tokenizer-local-dir /models/<model> --local-files-only
  # plus optionally: export TRANSFORMERS_OFFLINE=1
  ```

---

## 5) Outputs & Checkpoints
- Write to `/out/<run-id>` per run.
- Recommended structure:
  - `/out/<run>/epoch-*/rank-*/` per-epoch, per-rank
  - `/out/<run>/best` symlink/copy of best (lowest val loss/perplexity)
- Sync to object storage after completion (e.g., s5cmd/awscli) or mount an object store via CSI.

---

## 6) Validation split & metrics
- Provide a validation JSONL on `/data/val.jsonl`.
- After each epoch:
  - compute val loss/perplexity
  - update `best`
- Optional: track in MLflow
  ```bash
  export MLFLOW_TRACKING_URI=/mnt/filesystem-o2/mlruns
  # log metrics/params/artifacts from the training loop
  ```

---

## 7) Debugging DDP/NCCL
- Add ENV:
  ```bash
  NCCL_DEBUG=INFO
  NCCL_SOCKET_IFNAME=eth0    # or your data interface
  ```
- Check rendezvous reachability: `nc -vz train-rdzv 29500` from inside pod.
- If hanging on barriers, ensure identical images and torch/transformers versions across pods.

---

## 8) Example: train.py smoke test on K8s
Use the same StatefulSet but replace the `torchrun ... train_framework.py` command with:
```bash
torchrun --nnodes=$NNODES --node_rank=$RANK \
  --nproc_per_node=$NPROC_PER_NODE --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  k8s-training/k8s/train.py --epochs 2 --batch 256 --samples 4096 --features 128 --classes 4 --lr 3e-3 --seed 42
```

## 9) Viewing training logs:
```bash
kubectl -n train logs -f statefulset/train-llm | egrep "\[DDP\]|\[TRAIN\]|\[EPOCH\]|\[DONE\]"
```
