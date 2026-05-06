"""Multi-stage contrastive training pipeline for ColBERT-RU.

Phase 1 — easy negatives, shorter sequences.
Phase 2 — hard negatives, longer sequences.

Uses InfoNCE (contrastive) loss with temperature scaling and optional
in-batch negative mining.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from config import ModelConfig, DataConfig, TrainConfig, PipelineConfig
from modeling import ColBERTModel, build_model_and_tokenizer
from data import (
    load_training_data,
    enrich_with_hard_negatives,
    _load_mmarco_examples,
    _load_miracl_examples,
    build_dataloader,
    RetrievalExample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with temperature scaling.

    Given a batch of query embeddings and their positive-document scores,
    treats all other documents in the batch as negatives (in-batch negatives).
    Optionally adds an explicit hard-negative score column.

    ``L = -log( exp(s_pos / τ) / Σ_j exp(s_j / τ) )``
    """

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        in_batch_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss.

        Parameters
        ----------
        pos_scores : (B,)
            Similarity of each query with its gold positive document.
        neg_scores : (B,)
            Similarity of each query with its explicit hard negative.
        in_batch_scores : (B, B), optional
            Full cross-similarity matrix (query_i, doc_j) for in-batch negatives.
            Diagonal entries are the positive scores.

        Returns
        -------
        loss : scalar tensor.
        """
        if in_batch_scores is not None:
            logits = in_batch_scores / self.temperature  # (B, B)

            neg_col = (neg_scores / self.temperature).unsqueeze(1)  # (B, 1)
            logits = torch.cat([logits, neg_col], dim=1)            # (B, B+1)

            labels = torch.arange(logits.size(0), device=logits.device)
            return F.cross_entropy(logits, labels)

        # Fallback: pairwise loss (no in-batch negatives)
        scores = torch.stack([pos_scores, neg_scores], dim=1) / self.temperature
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        return F.cross_entropy(scores, labels)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: ColBERTModel,
    val_loader: DataLoader,
    loss_fn: InfoNCELoss,
    device: torch.device,
    max_batches: int = 100,
) -> Dict[str, float]:
    """Run a quick validation pass and return average loss + accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)

        loss = loss_fn(out["pos_scores"], out["neg_scores"], out["in_batch_scores"])
        total_loss += loss.item()

        correct += (out["pos_scores"] > out["neg_scores"]).sum().item()
        total += out["pos_scores"].size(0)

    n = min(i + 1, max_batches)
    return {
        "val_loss": total_loss / max(n, 1),
        "val_accuracy": correct / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Single training phase
# ---------------------------------------------------------------------------

def train_one_phase(
    model: ColBERTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    loss_fn: InfoNCELoss,
    device: torch.device,
    train_cfg: TrainConfig,
    phase_epochs: int,
    phase_name: str = "phase1",
    scaler: Optional[GradScaler] = None,
) -> None:
    """Run a single training phase (epochs loop with logging)."""
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(phase_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if scaler is not None:
                with autocast():
                    out = model(**batch)
                    loss = loss_fn(out["pos_scores"], out["neg_scores"], out["in_batch_scores"])
                    loss = loss / train_cfg.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            else:
                out = model(**batch)
                loss = loss_fn(out["pos_scores"], out["neg_scores"], out["in_batch_scores"])
                loss = loss / train_cfg.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            epoch_loss += loss.item() * train_cfg.gradient_accumulation_steps

            if global_step % train_cfg.log_every_n_steps == 0 and global_step > 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "[%s] epoch=%d  step=%d  loss=%.4f  lr=%.2e",
                    phase_name, epoch, global_step, loss.item() * train_cfg.gradient_accumulation_steps, lr,
                )

            if global_step % train_cfg.eval_every_n_steps == 0 and global_step > 0:
                val_metrics = validate(model, val_loader, loss_fn, device)
                logger.info(
                    "[%s] VALIDATION  step=%d  val_loss=%.4f  val_acc=%.4f",
                    phase_name, global_step, val_metrics["val_loss"], val_metrics["val_accuracy"],
                )
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    _save_checkpoint(model, optimizer, scheduler, global_step, train_cfg, tag=f"{phase_name}_best")
                model.train()

            if global_step % train_cfg.save_every_n_steps == 0 and global_step > 0:
                _save_checkpoint(model, optimizer, scheduler, global_step, train_cfg, tag=f"{phase_name}_step{global_step}")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(step + 1, 1)
        logger.info(
            "[%s] Epoch %d done — avg_loss=%.4f  time=%.1fs",
            phase_name, epoch, avg_loss, epoch_time,
        )

    _save_checkpoint(model, optimizer, scheduler, global_step, train_cfg, tag=f"{phase_name}_final")


def _save_checkpoint(
    model: ColBERTModel,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    global_step: int,
    train_cfg: TrainConfig,
    tag: str = "latest",
) -> None:
    path = os.path.join(train_cfg.output_dir, f"checkpoint_{tag}.pt")
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
        },
        path,
    )
    logger.info("Checkpoint saved → %s", path)


# ---------------------------------------------------------------------------
# Full multi-stage pipeline
# ---------------------------------------------------------------------------

def run_training(cfg: PipelineConfig) -> None:
    """Execute the two-phase ColBERT-RU training pipeline.

    Phase 1: random / easy negatives, shorter max lengths.
    Phase 2: hard negatives, longer max lengths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Phase 1 setup ----
    phase1_model_cfg = ModelConfig(
        **{**cfg.model.__dict__, "query_max_len": cfg.train.phase1_max_len, "doc_max_len": cfg.train.phase1_max_len}
    )
    model, tokenizer = build_model_and_tokenizer(phase1_model_cfg)
    model.to(device)

    train_loader, val_loader = load_training_data(
        tokenizer, phase1_model_cfg, cfg.data,
        train_batch_size=cfg.train.per_device_batch_size,
        num_workers=cfg.train.num_workers,
    )

    total_steps_p1 = (len(train_loader) // cfg.train.gradient_accumulation_steps) * cfg.train.num_epochs_phase1
    optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.train.learning_rate,
        total_steps=max(total_steps_p1, 1),
        pct_start=cfg.train.warmup_ratio,
    )
    loss_fn = InfoNCELoss(temperature=cfg.train.temperature)
    scaler = GradScaler() if cfg.train.fp16 and device.type == "cuda" else None

    logger.info("=== Phase 1: Easy negatives, max_len=%d ===", cfg.train.phase1_max_len)
    train_one_phase(
        model, train_loader, val_loader,
        optimizer, scheduler, loss_fn, device,
        cfg.train, cfg.train.num_epochs_phase1,
        phase_name="phase1", scaler=scaler,
    )

    # ---- Phase 2 setup ----
    logger.info("=== Phase 2: Hard negatives, max_len=%d ===", cfg.train.phase2_max_len)

    phase2_model_cfg = ModelConfig(
        **{**cfg.model.__dict__, "query_max_len": cfg.train.phase2_max_len, "doc_max_len": cfg.train.phase2_max_len}
    )
    model.cfg = phase2_model_cfg

    ru_examples = _load_mmarco_examples(cfg.data, split="train", lang="russian")
    miracl_examples = _load_miracl_examples(cfg.data, split="train")
    all_examples = ru_examples + miracl_examples

    corpus_texts = list({ex.positive for ex in all_examples})
    logger.info("Mining hard negatives over %d corpus documents…", len(corpus_texts))
    all_examples = enrich_with_hard_negatives(
        all_examples, corpus_texts, cfg.data, device=str(device),
    )

    train_loader_p2 = build_dataloader(
        all_examples, tokenizer, phase2_model_cfg, cfg.data,
        train_cfg_batch_size=cfg.train.per_device_batch_size,
        num_workers=cfg.train.num_workers,
        is_train=True,
    )

    total_steps_p2 = (len(train_loader_p2) // cfg.train.gradient_accumulation_steps) * cfg.train.num_epochs_phase2
    optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate * 0.5, weight_decay=cfg.train.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.train.learning_rate * 0.5,
        total_steps=max(total_steps_p2, 1),
        pct_start=cfg.train.warmup_ratio,
    )
    scaler = GradScaler() if cfg.train.fp16 and device.type == "cuda" else None

    train_one_phase(
        model, train_loader_p2, val_loader,
        optimizer, scheduler, loss_fn, device,
        cfg.train, cfg.train.num_epochs_phase2,
        phase_name="phase2", scaler=scaler,
    )

    logger.info("Training complete. Final checkpoints in %s", cfg.train.output_dir)
