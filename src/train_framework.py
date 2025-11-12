#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py — minimalistic and readable framework for distributed LoRA training of LLMs
on PyTorch + Hugging Face Transformers + PEFT with DDP (torchrun).

GOAL:
- Launch fine-tuning (LoRA) of one LLM model in Kubernetes/cluster.
- DDP through torchrun on N nodes/GPUs (mechanism: rendezvous, RANK/WORLD_SIZE).
- Simple dataset format (JSONL) with support for messages / prompt+completion / text.
- Accurate logging, memory control (dynamic padding, non_blocking).

OUTPUT:
- Saved artifacts of the model (LoRA adapters) and tokenizer in the specified directory.
- Optionally — JSON report with metrics (tokens per second, steps, world_size).

INPUT:
- Command-line arguments (see USAGE below).
- JSONL dataset (each line — object: {"messages":[...] } or {"prompt":..., "completion":...} or {"text":...}).

SPECIAL IMPORTS (and why):
- transformers: model and tokenizer (AutoModelForCausalLM, AutoTokenizer).
- peft: LoRA (get_peft_model, LoraConfig) — дешёвая, быстрая настройка головы модели.
- torch.distributed: DDP — распределённый тренинг без внешних фреймворков.

USAGE (locally, 1 GPU):
    python -m train \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data /mnt/filesystem-o2/datasets/fc.jsonl \
        --output /mnt/filesystem-o2/checkpoints/llama3b-lora \
        --lora --bf16 --epochs 2 --seq-len 4096 --batch 2 --grad-accum 8

USAGE (DDP, 4 processes per GPU, through torchrun):
    torchrun --nnodes=1 --nproc_per_node=4 \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
        train.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data /mnt/filesystem-o2/datasets/fc.jsonl \
        --output /mnt/filesystem-o2/checkpoints/llama3b-lora \
        --lora --bf16 --epochs 2 --seq-len 4096 --batch 2 --grad-accum 8

NOTES:
- For large datasets use common HF cache (HF_HOME) on shared storage.
- By default functions are short (under 20 lines); between functions — two empty lines.
- Variable names are self-explanatory; we try to keep strings under ~110 characters.
"""

# Plan (high-level execution order):
# 1) Parse CLI args and init logging
# 2) Read torchrun env and init DDP (if world>1), set global seed
# 3) Build tokenizer and dataloader (DistributedSampler if DDP)
# 4) Compute total steps for scheduler
# 5) Load base model, optionally apply LoRA, wrap with DDP if needed
# 6) Build optimizer and LR scheduler
# 7) Train for N epochs with grad accumulation, log tokens/s
# 8) On rank 0 save artifacts and optional JSON report, finalize DDP

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from logging import INFO, DEBUG, basicConfig, getLogger
from time import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model


# ============================
# Global constants/settings
# ============================
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_SEQ_LEN = 4096
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_PER_GPU = 2
DEFAULT_GRAD_ACCUM = 8
DEFAULT_LR = 2e-4
DEFAULT_WD = 0.05
DEFAULT_WARMUP_FRAC = 0.05
DEFAULT_SEED = 42
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

LOGGER = getLogger("train")


# ============================
# Arguments and initialization
# ============================
def parse_args() -> argparse.Namespace:
    """Parses CLI arguments. Sets safe defaults."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning with DDP")

    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, type=str)
    parser.add_argument("--data", required=True, type=str, help="Path to JSONL")
    parser.add_argument("--output", required=True, type=str, help="Directory for artifacts")
    parser.add_argument("--report-json", default="", type=str, help="Path for JSON report")

    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--batch", default=DEFAULT_BATCH_PER_GPU, type=int, help="Per-GPU batch size")
    parser.add_argument("--grad-accum", default=DEFAULT_GRAD_ACCUM, type=int)
    parser.add_argument("--seq-len", default=DEFAULT_SEQ_LEN, type=int)
    parser.add_argument("--lr", default=DEFAULT_LR, type=float)
    parser.add_argument("--wd", default=DEFAULT_WD, type=float)
    parser.add_argument("--warmup", default=DEFAULT_WARMUP_FRAC, type=float, help="Fraction of total steps")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 autocast")
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)

    parser.add_argument("--lora", action="store_true", help="Enable LoRA adaptation")
    parser.add_argument("--lora-r", default=DEFAULT_LORA_R, type=int)
    parser.add_argument("--lora-alpha", default=DEFAULT_LORA_ALPHA, type=int)
    parser.add_argument("--lora-dropout", default=DEFAULT_LORA_DROPOUT, type=float)
    parser.add_argument("--target-modules", nargs="*", default=list(DEFAULT_TARGET_MODULES))

    parser.add_argument("--log-level", default="INFO", type=str, help="Logging level: DEBUG/INFO/WARNING/ERROR")
    # Control batch log detail: DEBUG → all; otherwise by --log-batches
    parser.add_argument("--log-batches", default="first-last", type=str, choices=["all", "first-last", "none"])

    # Local paths (offline/HF token)
    parser.add_argument("--model-local-dir", default="", type=str, help="Path to local model dir (HF format)")
    parser.add_argument("--tokenizer-local-dir", default="", type=str, help="Path to local tokenizer dir (optional)")
    parser.add_argument("--local-files-only", action="store_true", help="Disable network; only use local files")

    return parser.parse_args()


def init_logging(level_name: str = "INFO") -> None:
    """Sets up logger with selected level."""
    level_name = str(level_name).upper()
    level_map = {"DEBUG": DEBUG, "INFO": INFO}
    level = level_map.get(level_name, INFO)
    basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def get_dist_env() -> Tuple[int, int, int, int]:
    """
    Returns (local_rank, global_rank, world_size, nproc_per_node) from env.
    These variables are set by torchrun. If they are not present, we assume single-GPU.
    """
    # Current variables set by torchrun. If they are not present — means single-GPU run
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # current GPU on this machine
    global_rank = int(os.environ.get("RANK", "0"))  # process number in the global world
    world_size = int(os.environ.get("WORLD_SIZE", "1"))  # total number of processes (GPUs)
    nproc_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))  # number of processes on one node
    return local_rank, global_rank, world_size, nproc_per_node


def init_distributed(local_rank: int, world_size: int) -> None:
    """Initializes the NCCL process group if we have more than one GPU."""
    if world_size > 1:
        LOGGER.info("[DDP] set_device local_rank=%d", local_rank)
        torch.cuda.set_device(local_rank)  # bind process to its GPU
        LOGGER.info("[DDP] init_process_group backend=nccl world=%d", world_size)
        dist.init_process_group(backend="nccl")  # start communication between processes (NCCL — standard for CUDA)


def finalize_distributed(world_size: int) -> None:
    """Closes the DDP process group correctly."""
    if world_size > 1 and dist.is_initialized():
        LOGGER.info("[DDP] barrier before destroy")
        dist.barrier()  # wait for all processes to reach this point
        LOGGER.info("[DDP] destroy_process_group")
        dist.destroy_process_group()  # safely close the group


def set_global_seed(seed: int) -> None:
    """Fixes seeds for reproducibility."""
    # Fix all random number generators to ensure reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================
# Dataset processing
# ============================
def _example_to_text(example: Dict[str, Any]) -> str:
    """
    Converts a JSON record to raw text for tokenization.
    Supports:
      - {"messages":[{"role":"user","content":"..."}, ...]}
      - {"prompt":"...", "completion":"..."}
      - {"text":"..."}
    """
    if "messages" in example:
        lines = []
        for msg in example["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"<<{role}>> {content}")
        return "\n".join(lines)

    if "prompt" in example and "completion" in example:
        prompt = str(example["prompt"]).rstrip()
        completion = str(example["completion"])
        return f"{prompt}\n{completion}"

    return str(example.get("text", ""))


class IndexedJsonlDataset(Dataset):
    """
    Lazy selection of JSONL without loading the entire file into memory.
    Idea: index the byte offsets of lines once, then read by seek.
    """

    def __init__(self, jsonl_path: str) -> None:
        self.jsonl_path = jsonl_path
        # Build index of byte offsets once, then read by seek
        self.offsets: List[int] = self._build_index(jsonl_path)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Read specific line by pre-saved offset
        with open(self.jsonl_path, "rb") as handle:
            handle.seek(self.offsets[index])
            line = handle.readline().decode("utf-8").strip()
            if not line:
                return {"text": ""}
            return json.loads(line)

    @staticmethod
    def _build_index(jsonl_path: str) -> List[int]:
        offsets: List[int] = []  # here we will store byte positions of non-empty lines
        cursor = 0
        with open(jsonl_path, "rb") as handle:
            while True:
                pos = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)
                cursor += len(line)
        return offsets


@dataclass
class CollateOutput:
    """Structure of the output batch after collation."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def build_tokenizer(model_name: str, tokenizer_dir: str = "", local_files_only: bool = False) -> Any:
    """Loads tokenizer (from local dir if provided) and ensures pad_token."""
    source = tokenizer_dir if tokenizer_dir else model_name
    LOGGER.info("[DATA] loading tokenizer from %s (local_files_only=%s)", source, str(local_files_only))
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # if no padding token — use EOS as padding
    LOGGER.info("[DATA] tokenizer loaded for model=%s pad_token=%s", model_name, str(tokenizer.pad_token))
    try:
        LOGGER.debug("[DATA] tokenizer vocab_size=%s model_max_length=%s padding_side=%s",
                     str(len(tokenizer)), str(getattr(tokenizer, "model_max_length", "?")), str(tokenizer.padding_side))
    except Exception:
        pass
    return tokenizer


def collate_dynamic_padding(
    batch: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> CollateOutput:
    """
    Dynamic padding (memory economy).
    Tokenize already at the collator level, to avoid storing large tensors in Dataset.
    """
    # Convert each JSON object to plain text (considering different formats)
    texts = [_example_to_text(example) for example in batch]
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]  # token indices
    attention_mask = encoded["attention_mask"]  # 1 — real tokens, 0 — padding
    labels = input_ids.clone()  # target tokens are the same as input (causal LM)
    labels[attention_mask == 0] = -100  # ignore padding in loss to not punish the model
    LOGGER.debug("[COLLATE] batch=%d max_len=%d input_ids=%s attn_mask=%s",
                 len(batch), max_length, tuple(input_ids.shape), tuple(attention_mask.shape))
    return CollateOutput(input_ids, attention_mask, labels)


def build_dataloader(
    jsonl_path: str,
    tokenizer: Any,
    max_length: int,
    world_size: int,
    global_rank: int,
    per_gpu_batch: int,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Creates DataLoader with dynamic padding; with DistributedSampler for DDP.
    """
    dataset = IndexedJsonlDataset(jsonl_path)
    sampler = None  # in DDP we distribute different chunks of dataset to different processes
    if world_size > 1:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            drop_last=False,
        )
        LOGGER.debug("[DATA] DistributedSampler rank=%d replicas=%d", global_rank, world_size)

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        out = collate_dynamic_padding(batch, tokenizer, max_length)
        return {
            "input_ids": out.input_ids,
            "attention_mask": out.attention_mask,
            "labels": out.labels,
        }

    # DataLoader gathers batches: dynamic padding saves memory
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=per_gpu_batch,
        num_workers=8,
        pin_memory=True,
        collate_fn=_collate,
    )
    try:
        num_batches = len(dataloader)
    except Exception:
        num_batches = -1
    LOGGER.info(
        "[DATA] path=%s size=%d per_gpu_batch=%d sampler=%s num_batches=%s",
        jsonl_path, len(dataset), per_gpu_batch, "DDP" if sampler else "None", str(num_batches)
    )
    LOGGER.debug("[DATA] loader kwargs num_workers=%d pin_memory=%s shuffle=%s",
                 2, str(True), str(sampler is None))
    return dataloader, sampler


# ============================
# Model, optimizer, loss
# ============================
def autocast_dtype(use_bf16: bool) -> torch.dtype:
    """Selects type for autocast: BF16 (preferred on H100/H200) or FP16."""
    return torch.bfloat16 if use_bf16 else torch.float16


def load_base_model(model_name_or_dir: str, use_bf16: bool, local_files_only: bool = False) -> torch.nn.Module:
    """Loads base CausalLM (from local dir if provided) and enables grad checkpointing."""
    dtype = torch.bfloat16 if use_bf16 else torch.float16  # select precision: BF16 (better on new GPUs) or FP16
    device_map = "auto" if use_bf16 else None
    try:
        import flash_attn  # noqa: F401
        _use_fa2 = True
    except Exception:
        _use_fa2 = False
    # SDPA or Flash Attention 2. (SDPA - Self-Attention with Dot Product - is the default implementation)
    attention_implementation = "flash_attention_2" if _use_fa2 else "sdpa"
    # Build config explicitly and force the attention implementation to avoid HF auto-toggling
    cfg = AutoConfig.from_pretrained(model_name_or_dir, local_files_only=local_files_only)
    setattr(cfg, "attn_implementation", attention_implementation)
    LOGGER.info("[MODEL] attn_impl=%s (flash_attn_installed=%s)", attention_implementation, str(_use_fa2))
    LOGGER.info("[MODEL] loading base model from %s (local_files_only=%s)", model_name_or_dir, str(local_files_only))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_dir,
        config=cfg,
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
        attn_implementation=attention_implementation
    )
    #model.gradient_checkpointing_enable()  # checkpointing saves memory by re-computing forward
    try:
        n_params = sum(p.numel() for p in model.parameters())
    except Exception:
        n_params = -1
    LOGGER.info("[MODEL] base loaded name=%s dtype=%s params=%s", model_name_or_dir, str(dtype), str(n_params))
    LOGGER.debug("[MODEL] gradient_checkpointing=%s device_map=%s", str(True), str(None))
    return model


def apply_lora_if_needed(
    model: torch.nn.Module,
    enabled: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
) -> torch.nn.Module:
    """Wraps model in PEFT-LoRA if enabled=True."""
    if not enabled:
        LOGGER.info("[MODEL] LoRA disabled; using base model")
        return model
    # LoRA teaches small adapters instead of all weights — it's fast and cheap
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    LOGGER.info(
        "[MODEL] applying LoRA r=%d alpha=%d dropout=%.3f targets=%s",
        lora_r, lora_alpha, lora_dropout, ",".join(target_modules)
    )
    try:
        before = sum(p.numel() for p in model.parameters())
    except Exception:
        before = -1
    return get_peft_model(model, cfg)


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Creates AdamW with careful eps/betas for LLM."""
    # AdamW — standard optimizer for LLM; betas/eps are tuned for transformers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    LOGGER.info("[MODEL] optimizer=AdamW lr=%.2e wd=%.2e", learning_rate, weight_decay)
    try:
        n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOGGER.debug("[MODEL] trainable_params=%s", str(n_params_trainable))
    except Exception:
        pass
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_frac: float,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear LR with warmup by fraction of steps."""
    warmup_steps = max(1, int(total_steps * warmup_frac))  # first we slowly warm up the LR (warmup)
    LOGGER.info("[MODEL] scheduler=Linear warmup_frac=%.3f warmup_steps=%d total_steps=%d", warmup_frac, warmup_steps, total_steps)
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


# ============================
# Training: one step/epoch/save
# ============================
def forward_loss(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    use_bf16: bool,
) -> torch.Tensor:
    """Computes loss for one step with autocast."""
    # autocast enables mixed precision — faster and cheaper by memory
    with torch.autocast(
        device_type="cuda",
        dtype=autocast_dtype(use_bf16),
        enabled=True,
    ):
        # Send tensors to GPU and compute loss inside the model
        out = model(
            input_ids=batch["input_ids"].cuda(non_blocking=True),
            attention_mask=batch["attention_mask"].cuda(non_blocking=True),
            labels=batch["labels"].cuda(non_blocking=True),
        )
    return out.loss


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_accum: int,
    epoch_index: int,
    total_epochs: int,
    global_rank: int,
    use_bf16: bool,
    log_batches: str,
) -> Tuple[int, float]:
    """
    One epoch pass. Returns (visible_tokens, total_time_sec).
    """
    model.train()  # enable training mode (dropout and etc.)
    tokens_seen = 0
    epoch_start = time()

    try:
        total_batches = len(dataloader)
    except Exception:
        total_batches = -1

    for step_index, batch in enumerate(dataloader):  # go through batches inside epoch
        try:
            t0 = time()
            loss = forward_loss(model, batch, use_bf16) / grad_accum  # divide loss for gradient accumulation
            loss.backward()  # compute gradients

            if (step_index + 1) % grad_accum == 0:
                optimizer.step()  # update weights once every grad_accum steps
                scheduler.step()  # update LR according to schedule
                optimizer.zero_grad(set_to_none=True)  # zero gradients efficiently

            tokens_seen += int(batch["attention_mask"].sum().item())  # count only «visible» (non-padding) tokens

            if global_rank == 0:
                # Control batch log detail: DEBUG → all; otherwise by --log-batches
                if LOGGER.isEnabledFor(DEBUG):
                    do_log = True
                elif log_batches == "all":
                    do_log = True
                elif log_batches == "first-last":
                    do_log = (step_index == 0) or (total_batches != -1 and step_index == total_batches - 1)
                else:  # none
                    do_log = False

                if do_log:
                    LOGGER.debug(
                        "[BATCH] epoch=%d step=%d/%s loss=%.4f tokens_seen=%d dt=%.3fs",
                        epoch_index + 1,
                        step_index,
                        str(total_batches - 1) if total_batches != -1 else "?",
                        float(loss.item() * grad_accum),
                        tokens_seen,
                        time() - t0,
                    )

            if global_rank == 0 and step_index % 50 == 0:
                elapsed = max(1e-6, time() - epoch_start)
                tps = int(tokens_seen / elapsed)
                LOGGER.info(
                    "[TRAIN] epoch=%d/%d step=%d tokens/s=%d",
                    epoch_index + 1,
                    total_epochs,
                    step_index,
                    tps,
                )
        except Exception as e:
            LOGGER.exception("[BATCH-ERROR] epoch=%d step=%d: %s", epoch_index + 1, step_index, str(e))
            raise

    return tokens_seen, time() - epoch_start


def compute_total_steps(
    dataset_size: int,
    per_gpu_batch: int,
    world_size: int,
    grad_accum: int,
    epochs: int,
) -> int:
    """Estimates the number of optimization steps (for LR scheduling)."""
    # How many optimizer steps will be in the entire training with size, batch, and world
    steps_per_epoch = math.ceil(dataset_size / max(1, per_gpu_batch * world_size))
    return max(1, (steps_per_epoch * epochs) // grad_accum)


def wrap_ddp_if_needed(
    model: torch.nn.Module,
    world_size: int,
    local_rank: int,
) -> torch.nn.Module:
    """Wraps in DDP if world_size>1."""
    model = model.cuda()  # move model to current GPU
    if world_size > 1:
        LOGGER.info("[DDP] wrapping model with DDP device_ids=[%d]", local_rank)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)  # wrap for gradient synchronization
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Returns the original module from DDP-wrapper (if any)."""
    return model.module if hasattr(model, "module") else model  # extract real model from DDP-wrapper


def save_artifacts(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: str,
    report_path: str,
    world_size: int,
    tokens_total: int,
    total_time_sec: float,
) -> None:
    """
    Saves weights and tokenizer on rank==0. Also writes JSON report.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_to_save = unwrap_model(model)  # save the clean model (without DDP-wrapper)
    model_to_save.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    if report_path:
        payload = {
            "world_size": world_size,
            "tokens_total": tokens_total,
            "elapsed_sec": round(total_time_sec, 3),
            "tokens_per_sec": int(tokens_total / max(1e-6, total_time_sec)),
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    LOGGER.info("[RESULT] artifacts saved → %s", output_dir)


# ============================
# Main scenario
# ============================
def main() -> int:
    """Orchestrates all steps: arguments → DDP → data → training → save."""
    # First parse arguments to know which logging level to enable
    LOGGER.info("[STEP] parse CLI args")
    args = parse_args()  # take all launch parameters
    init_logging(args.log_level)

    LOGGER.info("[STEP] read DDP env and init")
    local_rank, global_rank, world_size, _ = get_dist_env()  # ranks and world size from torchrun
    init_distributed(local_rank, world_size)
    LOGGER.info("[STEP] set global seed=%d", args.seed)
    set_global_seed(args.seed)  # fix randomness to repeat results

    if global_rank == 0:
        LOGGER.info("[DDP] rank=%d world=%d device=%d", global_rank, world_size, local_rank)
        LOGGER.info("[CFG] model=%s lora=%s bf16=%s seq_len=%d batch=%d grad_accum=%d",
                    args.model, args.lora, args.bf16, args.seq_len, args.batch, args.grad_accum)

    try:
        LOGGER.info("[STEP] build tokenizer and dataloader")
        tokenizer_source = args.tokenizer_local_dir if args.tokenizer_local_dir else args.model
        tokenizer = build_tokenizer(
            model_name=args.model,
            tokenizer_dir=args.tokenizer_local_dir,
            local_files_only=args.local_files_only,
        )  # can cut strings into tokens, adds padding
        dataloader, sampler = build_dataloader(
            jsonl_path=args.data,
            tokenizer=tokenizer,
            max_length=args.seq_len,
            world_size=world_size,
            global_rank=global_rank,
            per_gpu_batch=args.batch,
        )

        dataset_size = len(dataloader.dataset)  # type: ignore[attr-defined]
        total_steps = compute_total_steps(  # needed for LR scheduler
            dataset_size=dataset_size,
            per_gpu_batch=args.batch,
            world_size=world_size,
            grad_accum=args.grad_accum,
            epochs=args.epochs,
        )
        if global_rank == 0:
            LOGGER.info("[DATA] dataset_size=%d total_steps=%d", dataset_size, total_steps)

        LOGGER.info("[STEP] load base model and apply LoRA if enabled")
        model_source = args.model_local_dir if args.model_local_dir else args.model
        model = load_base_model(model_source, args.bf16, local_files_only=args.local_files_only)  # load base LLM
        model = apply_lora_if_needed(
            model=model,
            enabled=args.lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )  # apply LoRA if enabled
        LOGGER.info("[STEP] wrap with DDP if needed")  # wrap with DDP if needed
        model = wrap_ddp_if_needed(model, world_size, local_rank)  # distributed training on multiple GPUs

        LOGGER.info("[STEP] build optimizer and scheduler")
        optimizer = build_optimizer(unwrap_model(model), args.lr, args.wd)  # configure optimizer for real weights
        scheduler = build_scheduler(optimizer, args.warmup, total_steps)  # and learning rate scheduler

        tokens_total = 0
        train_start = time()
        LOGGER.info("[STEP] start training epochs=%d", args.epochs)

        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)  # important step for DistributedSampler: different shuffling each epoch

            epoch_t0 = time()
            tokens_seen, _ = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_accum=args.grad_accum,
                epoch_index=epoch,
                total_epochs=args.epochs,
                global_rank=global_rank,
                use_bf16=args.bf16,
                log_batches=args.log_batches,
            )
            tokens_total += int(tokens_seen)
            if global_rank == 0:
                LOGGER.info("[EPOCH] %d/%d tokens_seen=%d time=%.2fs",
                            epoch + 1, args.epochs, tokens_seen, time() - epoch_t0)

        total_time = time() - train_start

        if global_rank == 0:
            save_artifacts(
                model=unwrap_model(model),
                tokenizer=tokenizer,
                output_dir=args.output,
                report_path=args.report_json,
                world_size=world_size,
                tokens_total=tokens_total,
                total_time_sec=total_time,
            )
            LOGGER.info("[DONE] training finished: epochs=%d total_tokens=%d elapsed=%.2fs (~%d tok/s)",
                        args.epochs, tokens_total, total_time, int(tokens_total / max(1e-6, total_time)))
    except Exception as e:
        LOGGER.exception("[FATAL] training failed: %s", str(e))
        raise
    finally:
        LOGGER.info("[STEP] finalize DDP")
        finalize_distributed(world_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())