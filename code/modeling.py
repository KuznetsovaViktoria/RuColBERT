"""ColBERT late-interaction model with XLM-RoBERTa backbone.

Implements:
  * Multi-vector query / document encoding via XLM-RoBERTa + linear projection.
  * MaxSim relevance scoring.
  * Embedding compression: linear dimensionality reduction and token pruning.
  * Punctuation-mask filtering.
"""

from __future__ import annotations

import string
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from config import ModelConfig

_PUNCTUATION = set(string.punctuation)


class ColBERTEncoder(nn.Module):
    """Shared encoder that produces per-token embeddings for either queries or
    documents.

    Architecture
    ------------
    XLM-RoBERTa  →  Linear(hidden_size, embedding_dim)  →  L2-normalise

    For documents, an optional **token-pruning** step can discard tokens whose
    importance score (L2 norm before normalisation) falls below a threshold.
    This reduces index size without measurably hurting quality (ColBERTer idea).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.encoder_name)
        hidden = self.backbone.config.hidden_size
        self.linear = nn.Linear(hidden, cfg.embedding_dim, bias=False)
        self.normalize = cfg.normalize_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode input tokens into per-token embeddings.

        Parameters
        ----------
        input_ids : (B, L)
        attention_mask : (B, L)

        Returns
        -------
        embs : (B, L, D)  — optionally L2-normalised.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, H)
        embs = self.linear(hidden_states)           # (B, L, D)

        if self.normalize:
            embs = F.normalize(embs, p=2, dim=-1)

        return embs


class TokenPruner(nn.Module):
    """Drop document-token embeddings whose pre-norm magnitude is below a
    learned or fixed threshold.

    This reduces the average number of vectors stored per document — a key
    system-level metric for ColBERT.

    If ``threshold == 0`` the module is a no-op (all tokens kept).
    """

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        embs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prune low-importance tokens.

        Parameters
        ----------
        embs : (B, L, D)  — embeddings **before** L2 normalisation.
        attention_mask : (B, L)

        Returns
        -------
        pruned_embs : (B, L, D)  — zeros in pruned positions.
        pruned_mask : (B, L)     — updated mask.
        """
        if self.threshold <= 0:
            return embs, attention_mask

        norms = embs.norm(dim=-1)  # (B, L)
        keep = (norms > self.threshold) & attention_mask.bool()
        pruned_mask = keep.long()
        pruned_embs = embs * pruned_mask.unsqueeze(-1).float()
        return pruned_embs, pruned_mask


def _build_punctuation_mask(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
) -> torch.Tensor:
    """Return a boolean mask that is ``False`` for punctuation-only tokens."""
    batch_size, seq_len = input_ids.shape
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)

    for b in range(batch_size):
        tokens = tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
        for t, tok in enumerate(tokens):
            cleaned = tok.lstrip("▁").strip()
            if cleaned and all(c in _PUNCTUATION for c in cleaned):
                mask[b, t] = False
    return mask


class ColBERTModel(nn.Module):
    """Full ColBERT late-interaction model.

    Provides separate ``encode_query`` / ``encode_doc`` methods and
    a ``score`` method that computes MaxSim relevance.

    Compression mechanisms:
      * **Linear projection** (always active): reduces hidden dim → embedding_dim.
      * **Token pruning** (optional): drops low-importance doc-token vectors.
      * **Punctuation masking**: ignores punctuation tokens during scoring.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ColBERTEncoder(cfg)
        self.pruner = TokenPruner(cfg.pruning_threshold)
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of queries.

        Returns
        -------
        q_embs : (B, Lq, D)
        q_mask : (B, Lq)
        """
        q_embs = self.encoder(input_ids, attention_mask)
        return q_embs, attention_mask

    def encode_doc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        apply_pruning: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of documents.

        Applies token pruning and punctuation masking when enabled.

        Returns
        -------
        d_embs : (B, Ld, D)
        d_mask : (B, Ld)
        """
        raw_embs = self.encoder.backbone(input_ids=input_ids, attention_mask=attention_mask)
        raw_hidden = raw_embs.last_hidden_state

        if apply_pruning and self.cfg.compression_strategy == "token_pruning":
            projected = self.encoder.linear(raw_hidden)
            projected, attention_mask = self.pruner(projected, attention_mask)
            if self.encoder.normalize:
                norms = projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                projected = projected / norms
            d_embs = projected
        else:
            d_embs = self.encoder.linear(raw_hidden)
            if self.encoder.normalize:
                d_embs = F.normalize(d_embs, p=2, dim=-1)

        if self.cfg.mask_punctuation and self.tokenizer is not None:
            punct_mask = _build_punctuation_mask(input_ids, self.tokenizer)
            attention_mask = attention_mask * punct_mask.long().to(attention_mask.device)

        return d_embs, attention_mask

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def maxsim(
        q_embs: torch.Tensor,
        q_mask: torch.Tensor,
        d_embs: torch.Tensor,
        d_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MaxSim relevance between queries and documents.

        For each query-token embedding, find its maximum cosine similarity to
        any document-token embedding, then sum over query tokens.

        Parameters
        ----------
        q_embs : (Bq, Lq, D)
        q_mask : (Bq, Lq)
        d_embs : (Bd, Ld, D)
        d_mask : (Bd, Ld)

        Returns
        -------
        scores : (Bq, Bd)
        """
        # (Bq, Lq, D) x (Bd, Ld, D)^T  ->  (Bq, Bd, Lq, Ld)
        sim = torch.einsum("iqd,jkd->ijqk", q_embs, d_embs)

        d_mask_expanded = d_mask.unsqueeze(0).unsqueeze(2).bool()  # (1, Bd, 1, Ld)
        sim = sim.masked_fill(~d_mask_expanded, float("-inf"))

        max_sim, _ = sim.max(dim=-1)  # (Bq, Bd, Lq)

        q_mask_expanded = q_mask.unsqueeze(1).bool()  # (Bq, 1, Lq)
        max_sim = max_sim.masked_fill(~q_mask_expanded, 0.0)

        scores = max_sim.sum(dim=-1)  # (Bq, Bd)
        return scores

    def score(
        self,
        q_input_ids: torch.Tensor,
        q_attention_mask: torch.Tensor,
        d_input_ids: torch.Tensor,
        d_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end scoring: encode + MaxSim."""
        q_embs, q_mask = self.encode_query(q_input_ids, q_attention_mask)
        d_embs, d_mask = self.encode_doc(d_input_ids, d_attention_mask)
        return self.maxsim(q_embs, q_mask, d_embs, d_mask)

    # ------------------------------------------------------------------
    # Forward (training): returns per-query scores for pos + neg docs
    # ------------------------------------------------------------------

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        pos_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        neg_input_ids: torch.Tensor,
        neg_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Returns dict with ``pos_scores`` and ``neg_scores`` (both shape (B,)),
        plus ``in_batch_scores`` of shape (B, B) for in-batch negative loss.
        """
        q_embs, q_mask = self.encode_query(query_input_ids, query_attention_mask)
        p_embs, p_mask = self.encode_doc(pos_input_ids, pos_attention_mask)
        n_embs, n_mask = self.encode_doc(neg_input_ids, neg_attention_mask)

        # Paired scores
        pos_scores = self.maxsim(q_embs, q_mask, p_embs, p_mask)
        neg_scores = self.maxsim(q_embs, q_mask, n_embs, n_mask)

        pos_diag = pos_scores.diag()  # (B,)
        neg_diag = neg_scores.diag()  # (B,)

        return {
            "pos_scores": pos_diag,
            "neg_scores": neg_diag,
            "in_batch_scores": pos_scores,  # (B, B) for in-batch negatives
        }


def build_model_and_tokenizer(cfg: ModelConfig) -> Tuple[ColBERTModel, PreTrainedTokenizerFast]:
    """Instantiate model + tokenizer from config."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
    model = ColBERTModel(cfg)
    model.set_tokenizer(tokenizer)
    return model, tokenizer
