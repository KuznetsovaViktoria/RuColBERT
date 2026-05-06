from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

DENSE_MODELS: Dict[str, str] = {
    "E5-base": "intfloat/multilingual-e5-base",
}


class DenseBaseline:
    """Encode, index, and retrieve with a dense bi-encoder model.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model id (e.g. ``intfloat/multilingual-e5-base``).
    device : str or None
        ``"cuda"`` / ``"cpu"``; auto-detected when *None*.
    batch_size : int
        Encoding batch size.
    normalize : bool
        L2-normalize embeddings (required for cosine similarity via inner product).
    query_prefix : str
        Prefix prepended to queries before encoding (E5 needs ``"query: "``).
    passage_prefix : str
        Prefix prepended to passages before encoding (E5 needs ``"passage: "``).
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        logger.info("Loading dense model: %s on %s", model_name_or_path, self.device)
        self.model = SentenceTransformer(model_name_or_path, device=self.device)

        self._index = None
        self._doc_ids: List[str] = []
        self._corpus_embeddings: Optional[np.ndarray] = None
        self._index_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _add_prefix(self, texts: List[str], prefix: str) -> List[str]:
        if not prefix:
            return texts
        return [prefix + t for t in texts]

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        texts = self._add_prefix(queries, self.query_prefix)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def encode_single_query(self, query: str) -> np.ndarray:
        text = self.query_prefix + query if self.query_prefix else query
        return self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_corpus(self, texts: List[str]) -> np.ndarray:
        texts = self._add_prefix(texts, self.passage_prefix)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(
        self,
        corpus_texts: List[str],
        doc_ids: Optional[List[str]] = None,
        index_dir: Optional[str] = None,
    ) -> float:
        """Encode corpus and build a FAISS IndexFlatIP.

        Returns index build time in seconds.
        """
        import faiss

        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(corpus_texts))]
        self._doc_ids = doc_ids

        t0 = time.perf_counter()
        self._corpus_embeddings = self.encode_corpus(corpus_texts)
        dim = self._corpus_embeddings.shape[1]

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._corpus_embeddings.astype(np.float32))
        build_time = time.perf_counter() - t0

        if index_dir:
            self._index_dir = Path(index_dir)
            self._index_dir.mkdir(parents=True, exist_ok=True)
            np.save(self._index_dir / "corpus_embeddings.npy", self._corpus_embeddings)
            faiss.write_index(self._index, str(self._index_dir / "index.faiss"))
            import json
            with open(self._index_dir / "doc_ids.json", "w") as f:
                json.dump(doc_ids, f)
            with open(self._index_dir / "metadata.json", "w") as f:
                json.dump({
                    "model": self.model_name,
                    "num_docs": len(doc_ids),
                    "embedding_dim": int(dim),
                    "index_size_mb": self._corpus_embeddings.nbytes / (1024 ** 2),
                    "build_time_sec": build_time,
                }, f, indent=2)

        logger.info(
            "Dense index built: %d docs, dim=%d, %.1f MB, %.1f sec",
            len(doc_ids), dim,
            self._corpus_embeddings.nbytes / (1024 ** 2),
            build_time,
        )
        return build_time

    def load_index(self, index_dir: str) -> None:
        """Load a previously saved FAISS index."""
        import faiss
        import json

        idx = Path(index_dir)
        self._corpus_embeddings = np.load(idx / "corpus_embeddings.npy")
        self._index = faiss.read_index(str(idx / "index.faiss"))
        with open(idx / "doc_ids.json") as f:
            self._doc_ids = json.load(f)
        self._index_dir = idx

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve top-k documents for a single query.

        Compatible with ``SystemMetrics.measure_latency`` and
        ``evaluate_robustness`` from ``evaluate.py``.
        """
        q_emb = self.encode_single_query(query)  # (1, D)
        scores, indices = self._index.search(q_emb.astype(np.float32), top_k)
        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self._doc_ids[idx], float(score)))
        return results

    def retrieve_fn(self, top_k: int = 100) -> Callable[[str], List[Tuple[str, float]]]:
        """Return a closure suitable for ``SystemMetrics.measure_latency``."""
        def _fn(query: str) -> List[Tuple[str, float]]:
            return self.retrieve(query, top_k=top_k)
        return _fn

    def encode_only_fn(self) -> Callable[[str], np.ndarray]:
        """Return a closure that only encodes a query (no retrieval)."""
        def _fn(query: str) -> np.ndarray:
            return self.encode_single_query(query)
        return _fn

    def index_size_mb(self) -> float:
        if self._corpus_embeddings is not None:
            return self._corpus_embeddings.nbytes / (1024 ** 2)
        return 0.0


def build_e5_baseline(**kwargs) -> DenseBaseline:
    """Build multilingual-e5-base baseline with correct prefixes."""
    return DenseBaseline(
        DENSE_MODELS["E5-base"],
        query_prefix="query: ",
        passage_prefix="passage: ",
        **kwargs,
    )
