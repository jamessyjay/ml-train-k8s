#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
( - AI-generated code -)
train_smoke.py — DDP smoke-training on synthetic data.

Goal:
- Quickly prove that training is running and distributed training works on all 4 nodes.
- No external datasets/libraries (only PyTorch).

Output:
- Logs with rank/world, CUDA device name, loss, and time.

Input/launch (1 node × 4 GPU):
    torchrun --nproc_per_node=4 train_smoke.py --epochs 2

Input/launch (4 nodes × 1 GPU; rendezvous via master):
    # on all nodes:
    torchrun --nnodes=4 --nproc_per_node=1 \
      --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
      train_smoke.py --epochs 2

Tips:
- Add env NODE_NAME: fieldRef: spec.nodeName to K8s manifest to see node name.
"""

# Plan (what happens when you run the script):
# 1) Parse CLI args and init logging
# 2) Read DDP env (torchrun exports ranks/world) and init process group if needed
# 3) Fix seeds for reproducibility across ranks
# 4) Build synthetic dataset + (optionally) DistributedSampler + DataLoader
# 5) Create model/optimizer/loss on CUDA and wrap with DDP if world>1
# 6) Train for N epochs: forward → CE loss → backward → step; log loss per-rank
# 7) Print final summary per rank (to ensure all ranks/nodes ran)
# 8) Gracefully finalize DDP

from __future__ import annotations

import argparse
import os
import random
import socket
from logging import INFO, basicConfig, getLogger
from time import time
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler


LOGGER = getLogger("smoke")

# ------------------------ CLI ------------------------
def parse_arguments() -> argparse.Namespace:
    """Parsing command-line arguments."""
    parser = argparse.ArgumentParser(description="DDP smoke training on synthetic data")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=256, help="Batch size per process (per-rank)")
    parser.add_argument("--samples", type=int, default=4096, help="Size of synthetic dataset")
    parser.add_argument("--features", type=int, default=128, help="Number of input features")
    parser.add_argument("--classes", type=int, default=4, help="Number of classes")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def init_logging() -> None:
    """Logging setup."""
    basicConfig(level=INFO, format="%(levelname)s:%(name)s:%(message)s")


# ------------------------ DDP utils ------------------------
def get_dist_env() -> Tuple[int, int, int, int]:
    """LOCAL_RANK, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE from env (torchrun sets them)."""
    # torchrun sets these variables; defaults allow single-process debug without torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return local_rank, rank, world, local_world


def init_dist_if_needed(local_rank: int, world: int) -> None:
    """Initializes NCCL process group if world>1 and binds CUDA to local_rank."""
    # Pin current process to its GPU to avoid accidental cross-GPU usage
    LOGGER.info("[DDP] set_device local_rank=%d", local_rank)
    torch.cuda.set_device(local_rank)
    if world > 1:
        # Default rendezvous comes from torchrun; NCCL is the backend for multi-GPU
        LOGGER.info("[DDP] init_process_group backend=nccl world>1")
        dist.init_process_group(backend="nccl")


def finalize_dist(world: int) -> None:
    """Finalizes DDP process group."""
    if world > 1 and dist.is_initialized():
        LOGGER.info("[DDP] barrier before destroy")
        dist.barrier()
        LOGGER.info("[DDP] destroy_process_group")
        dist.destroy_process_group()


# ------------------------ Data/Model ------------------------
class SyntheticDataset(Dataset):
    """Synthetic linearly separable dataset. Generated deterministically."""

    def __init__(self, n_samples: int, n_features: int, n_classes: int, seed: int) -> None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        # Random features generated with a fixed CPU generator → reproducible across ranks
        self.inputs = torch.randn(n_samples, n_features, generator=g)
        # Linear weights to make classes linearly separable; argmax of logits becomes label
        weights = torch.randn(n_features, n_classes, generator=g)
        logits = self.inputs @ weights
        self.labels = logits.argmax(dim=1)
        self.n = n_samples

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        return self.inputs[index], self.labels[index]


class TinyMLP(nn.Module):
    """Tiny MLP: enough to make loss drop quickly."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------ Train ------------------------
def set_global_seed(seed: int) -> None:
    """Fix seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(
    args: argparse.Namespace, rank: int, world: int
) -> DataLoader:
    """Creates DataLoader; adds DistributedSampler for DDP."""
    dataset = SyntheticDataset(args.samples, args.features, args.classes, args.seed)
    # DistributedSampler shards dataset across ranks so each GPU sees unique slices
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    # When sampler is set, DataLoader must not shuffle to avoid conflict
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler)
    try:
        num_batches = len(loader)
    except TypeError:
        num_batches = -1  # should not happen for standard DataLoader
    LOGGER.info(
        "[DATA] rank=%d samples=%d features=%d classes=%d batch=%d sampler=%s num_batches=%s",
        rank, dataset.n, args.features, args.classes, args.batch, "DDP" if sampler else "None", str(num_batches)
    )
    return loader


def build_model_and_opt(args: argparse.Namespace) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """Creates model, optimizer, and loss."""
    model = TinyMLP(args.features, args.classes).cuda()  # move model to current CUDA device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # AdamW is robust for this toy task
    criterion = nn.CrossEntropyLoss().cuda()  # standard multi-class objective
    try:
        n_params = sum(p.numel() for p in model.parameters())
    except Exception:
        n_params = -1
    LOGGER.info(
        "[MODEL] type=TinyMLP params=%s optimizer=AdamW lr=%.2e criterion=CrossEntropyLoss",
        str(n_params), args.lr
    )
    return model, optimizer, criterion


def wrap_ddp(model: nn.Module, local_rank: int, world: int) -> nn.Module:
    """Wraps model in DDP if world>1."""
    if world > 1:
        # device_ids ensures gradients sync on the correct GPU per process
        LOGGER.info("[DDP] wrapping model with DDP on device_ids=[%d]", local_rank)
        return DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """One training epoch. Returns average loss."""
    model.train()
    total_loss, total_count = 0.0, 0
    try:
        total_batches = len(loader)
    except Exception:
        total_batches = -1
    batch_idx = 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        try:
            # Transfer batch to GPU; non_blocking works if pinned memory is used
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # set_to_none=True is a tiny perf/mem win over zeroing to 0
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * inputs.size(0)
            total_count += inputs.size(0)

            if batch_idx == 0 or (total_batches != -1 and batch_idx == total_batches - 1):
                LOGGER.debug(
                    "[BATCH] idx=%d/%s size=%d loss=%.4f inputs=%s labels=%s",
                    batch_idx,
                    str(total_batches - 1) if total_batches != -1 else "?",
                    inputs.size(0),
                    float(loss.item()),
                    tuple(inputs.shape),
                    tuple(labels.shape),
                )
        except Exception as e:
            LOGGER.exception("[BATCH-ERROR] idx=%d: %s", batch_idx, str(e))
            raise
    return total_loss / max(1, total_count)


def main() -> int:
    """Main entry point: initialization → training → logging → finalization."""
    init_logging()
    LOGGER.info("[STEP] parse CLI args")
    args = parse_arguments()

    LOGGER.info("[STEP] read DDP env and init")
    local_rank, rank, world, local_world = get_dist_env()
    init_dist_if_needed(local_rank, world)
    LOGGER.info("[STEP] set global seed=%d", args.seed)
    set_global_seed(args.seed)

    node_name = os.environ.get("NODE_NAME", "unknown-node")
    pod_name = socket.gethostname()
    cuda_name = torch.cuda.get_device_name(local_rank)  # useful to verify GPU visibility per rank

    if rank == 0:
        # Log global settings once from rank 0 to avoid noise
        LOGGER.info("[ENV] world=%d local_world=%d", world, local_world)
        LOGGER.info(
            "[ENV] MASTER_ADDR=%s MASTER_PORT=%s",
            os.environ.get("MASTER_ADDR", "?"), os.environ.get("MASTER_PORT", os.environ.get("TORCHELASTIC_RESTART_COUNT", "?"))
        )
    LOGGER.info("[RANK] rank=%d node=%s pod=%s cuda=%s", rank, node_name, pod_name, cuda_name)

    try:
        LOGGER.info("[STEP] build loaders")
        loader = build_loaders(args, rank, world)
        LOGGER.info("[STEP] build model/optimizer/criterion")
        model, optimizer, criterion = build_model_and_opt(args)
        LOGGER.info("[STEP] wrap model with DDP if needed")
        model = wrap_ddp(model, local_rank, world)

        start = time()
        first_loss = None
        try:
            total_batches = len(loader)
        except Exception:
            total_batches = -1
        LOGGER.info("[STEP] start training epochs=%d batches_per_epoch=%s", args.epochs, str(total_batches))
        for epoch in range(args.epochs):
            t0 = time()
            epoch_loss = train_epoch(model, loader, optimizer, criterion)
            dt = time() - t0
            if first_loss is None:
                first_loss = epoch_loss
            LOGGER.info("[TRAIN] rank=%d epoch=%d loss=%.4f time=%.2fs", rank, epoch + 1, epoch_loss, dt)

        elapsed = time() - start
        # Expect solid drop on synthetic linear data; 30%+ improvement is a crude sanity check
        improved = first_loss is not None and epoch_loss < first_loss * 0.7

        # Result on each rank (convenient to see that all 4 ranks reached)
        LOGGER.info(
            "[RESULT] rank=%d world=%d improved=%s time=%.2fs first=%.4f last=%.4f",
            rank, world, str(improved), elapsed, (first_loss or 0.0), epoch_loss
        )
    except Exception as e:
        LOGGER.exception("[FATAL] training failed: %s", str(e))
        raise
    finally:
        LOGGER.info("[STEP] finalize DDP")
        finalize_dist(world)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())