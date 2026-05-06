from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from config import IndexConfig, ModelConfig, PipelineConfig
from modeling import ColBERTModel, build_model_and_tokenizer

logger = logging.getLogger(__name__)


class _CorpusDataset(Dataset):
    """Tokenises raw texts on the fly for batch encoding."""

    def __init__(
        self,
        texts: List[str],
        doc_ids: List[str],
        tokenizer: PreTrainedTokenizerFast,
        max_len: int,
    ) -> None:
        self.texts = texts
        self.doc_ids = doc_ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "doc_idx": idx,
        }


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class ColBERTIndexer:
    """Build and persist a ColBERT index over a document corpus.

    The index stores:
      * ``embeddings.npy``  — (total_tokens, D) float16/float32 tensor.
      * ``doc_offsets.npy``  — start offset for each document in the flat array.
      * ``doc_lengths.npy``  — number of retained token vectors per document.
      * ``metadata.json``    — corpus size, embedding dim, dtype.
    """

    def __init__(
        self,
        model: ColBERTModel,
        tokenizer: PreTrainedTokenizerFast,
        index_cfg: IndexConfig,
        model_cfg: ModelConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = index_cfg
        self.model_cfg = model_cfg
        self.device = torch.device("cuda" if index_cfg.use_gpu and torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def build(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> Path:
        """Encode all documents and write the index to ``self.cfg.index_dir``.

        Returns the index directory path.
        """
        self.model.to(self.device)
        self.model.eval()

        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(texts))]

        dataset = _CorpusDataset(texts, doc_ids, self.tokenizer, self.model_cfg.doc_max_len)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        all_embs: List[np.ndarray] = []
        doc_lengths: List[int] = []

        for batch in tqdm(loader, desc="Indexing"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            d_embs, d_mask = self.model.encode_doc(input_ids, attention_mask)

            for b in range(d_embs.size(0)):
                mask_b = d_mask[b].bool()
                kept = d_embs[b][mask_b].cpu().numpy()
                if self.cfg.save_fp16:
                    kept = kept.astype(np.float16)
                all_embs.append(kept)
                doc_lengths.append(kept.shape[0])

        flat_embs = np.concatenate(all_embs, axis=0)
        offsets = np.cumsum([0] + doc_lengths[:-1])

        index_dir = Path(self.cfg.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        np.save(index_dir / "embeddings.npy", flat_embs)
        np.save(index_dir / "doc_offsets.npy", offsets)
        np.save(index_dir / "doc_lengths.npy", np.array(doc_lengths))

        with open(index_dir / "doc_ids.json", "w") as f:
            json.dump(doc_ids, f)

        metadata = {
            "num_docs": len(texts),
            "total_vectors": int(flat_embs.shape[0]),
            "embedding_dim": int(flat_embs.shape[1]),
            "dtype": str(flat_embs.dtype),
            "avg_vectors_per_doc": float(np.mean(doc_lengths)),
            "index_size_mb": flat_embs.nbytes / (1024 ** 2),
        }
        with open(index_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Index built: %d docs, %d vectors, %.1f MB, avg %.1f vecs/doc",
            metadata["num_docs"],
            metadata["total_vectors"],
            metadata["index_size_mb"],
            metadata["avg_vectors_per_doc"],
        )
        return index_dir


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class ColBERTRetriever:
    """Load a pre-built ColBERT index and retrieve documents for queries."""

    def __init__(
        self,
        model: ColBERTModel,
        tokenizer: PreTrainedTokenizerFast,
        index_dir: str,
        model_cfg: ModelConfig,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        idx = Path(index_dir)
        self.flat_embs = torch.from_numpy(np.load(idx / "embeddings.npy").astype(np.float32)).to(self.device)
        self.doc_offsets = np.load(idx / "doc_offsets.npy")
        self.doc_lengths = np.load(idx / "doc_lengths.npy")
        with open(idx / "doc_ids.json") as f:
            self.doc_ids = json.load(f)
        with open(idx / "metadata.json") as f:
            self.metadata = json.load(f)

        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded index: %d docs, %d vectors", len(self.doc_ids), self.flat_embs.size(0))

    def _get_doc_embs(self, doc_idx: int) -> torch.Tensor:
        """Slice the flat embedding tensor for a single document."""
        start = int(self.doc_offsets[doc_idx])
        length = int(self.doc_lengths[doc_idx])
        return self.flat_embs[start : start + length]  # (Ld, D)

    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Retrieve top-k documents for a single query.

        Returns list of (doc_id, score) tuples sorted by descending score.
        """
        enc = self.tokenizer(
            query,
            max_length=self.model_cfg.query_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        q_ids = enc["input_ids"].to(self.device)
        q_mask = enc["attention_mask"].to(self.device)

        q_embs, q_mask_out = self.model.encode_query(q_ids, q_mask)
        q_embs = q_embs.squeeze(0)              # (Lq, D)
        q_active = q_mask_out.squeeze(0).bool()  # (Lq,)
        q_embs = q_embs[q_active]                # (Lq', D)

        scores: List[Tuple[int, float]] = []
        for doc_idx in range(len(self.doc_ids)):
            d_embs = self._get_doc_embs(doc_idx)   # (Ld, D)
            sim = q_embs @ d_embs.T                 # (Lq', Ld)
            max_sim_per_q, _ = sim.max(dim=1)       # (Lq',)
            score = max_sim_per_q.sum().item()
            scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.doc_ids[idx], s) for idx, s in scores[:top_k]]

    @torch.no_grad()
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """Retrieve top-k for multiple queries."""
        return [self.retrieve(q, top_k) for q in tqdm(queries, desc="Retrieving")]


# ---------------------------------------------------------------------------
# BM25 Baseline
# ---------------------------------------------------------------------------

class BM25Baseline:
    """Lexical BM25 baseline using ``rank_bm25``."""

    def __init__(self, corpus_texts: List[str], doc_ids: Optional[List[str]] = None) -> None:
        from rank_bm25 import BM25Okapi

        self.corpus_texts = corpus_texts
        self.doc_ids = doc_ids or [str(i) for i in range(len(corpus_texts))]
        tokenized = [text.lower().split() for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked]

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        return [self.retrieve(q, top_k) for q in tqdm(queries, desc="BM25 retrieving")]
