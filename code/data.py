from __future__ import annotations

import gzip
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import PreTrainedTokenizerFast

from config import DataConfig, ModelConfig

logger = logging.getLogger(__name__)

@dataclass
class RetrievalExample:
    query: str
    positive: str
    negatives: List[str]
    lang: str = "ru"


def _resolve_mmarco_triple_cap(cfg: DataConfig) -> Optional[int]:
    if cfg.mmarco_max_triples == 0:
        return None
    caps: List[int] = []
    if cfg.mmarco_max_triples and cfg.mmarco_max_triples > 0:
        caps.append(cfg.mmarco_max_triples)
    if cfg.max_train_samples is not None:
        caps.append(cfg.max_train_samples)
    if not caps:
        return 100_000
    return min(caps)


def _load_mmarco_examples_from_hub_files(
    lang: str,
    cache_dir: str,
    max_triples: Optional[int],
) -> List[RetrievalExample]:
    """Load mMARCO train triples by downloading TSVs from the Hub.

    ``unicamp-dl/mmarco`` ships ``mmarco.py``; ``load_dataset`` fails on
    ``datasets>=3`` with *Dataset scripts are no longer supported*. The parquet
    conversion branch also omits several languages (e.g. Russian).  This path
    uses ``hf_hub_download`` and mirrors the logic of the official script.

    **RAM:** Loading the full ``*_collection.tsv`` into a dict can exceed Colab RAM
    (~8M passages).  When ``max_triples`` is set, only that many triple lines are
    read first; then only the **referenced** query/doc IDs are loaded from the
    collection and query files (two-pass, bounded memory).
    """
    from huggingface_hub import hf_hub_download

    repo_id = "unicamp-dl/mmarco"
    kw = {"repo_id": repo_id, "repo_type": "dataset", "cache_dir": cache_dir}

    collection_path = hf_hub_download(
        filename=f"data/google/collections/{lang}_collection.tsv",
        **kw,
    )
    queries_path = hf_hub_download(
        filename=f"data/google/queries/train/{lang}_queries.train.tsv",
        **kw,
    )
    triples_path = hf_hub_download(
        filename="data/triples.train.ids.small.tsv",
        **kw,
    )

    lang_tag = "ru" if lang == "russian" else "en"

    if max_triples is None:
        logger.warning(
            "mMARCO: loading FULL collection + queries into RAM (--mmarco_max_triples 0). "
            "Requires tens of GB; prefer a finite cap on Colab.",
        )
        return _load_mmarco_examples_from_hub_files_full_ram(
            lang_tag, collection_path, queries_path, triples_path
        )

    # --- Pass 1: read up to max_triples lines of triples; collect IDs ---
    triple_lines: List[Tuple[str, str, str]] = []
    needed_q: Set[str] = set()
    needed_d: Set[str] = set()

    with open(triples_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_triples is not None and i >= max_triples:
                break
            parts = line.rstrip().split("\t")
            if len(parts) != 3:
                continue
            qid, pos_id, neg_id = parts
            triple_lines.append((qid, pos_id, neg_id))
            needed_q.add(qid)
            needed_d.add(pos_id)
            needed_d.add(neg_id)

    logger.info(
        "mMARCO %s: %d triple lines → loading %d unique queries, %d unique docs (subset)",
        lang,
        len(triple_lines),
        len(needed_q),
        len(needed_d),
    )

    # --- Pass 2: load only required queries ---
    queries_map: Dict[str, str] = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            qid, qtext = line.rstrip().split("\t", 1)
            if qid in needed_q:
                queries_map[qid] = qtext

    # --- Pass 3: stream collection, keep only needed doc IDs (stop early when all found) ---
    collection: Dict[str, str] = {}
    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            doc_id, doc = line.rstrip().split("\t", 1)
            if doc_id in needed_d:
                collection[doc_id] = doc
                if len(collection) >= len(needed_d):
                    break

    examples: List[RetrievalExample] = []
    for query_id, pos_id, neg_id in triple_lines:
        q = queries_map.get(query_id)
        p = collection.get(pos_id)
        n = collection.get(neg_id)
        if q is None or p is None or n is None:
            continue
        examples.append(
            RetrievalExample(
                query=q,
                positive=p,
                negatives=[n],
                lang=lang_tag,
            )
        )
    return examples


def _load_mmarco_examples_from_hub_files_full_ram(
    lang_tag: str,
    collection_path: str,
    queries_path: str,
    triples_path: str,
) -> List[RetrievalExample]:
    """Original all-in-RAM path for unlimited triples (large machines only)."""
    collection: Dict[str, str] = {}
    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            doc_id, doc = line.rstrip().split("\t", 1)
            collection[doc_id] = doc

    queries_map: Dict[str, str] = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            qid, qtext = line.rstrip().split("\t", 1)
            queries_map[qid] = qtext

    examples: List[RetrievalExample] = []
    with open(triples_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) != 3:
                continue
            query_id, pos_id, neg_id = parts
            q = queries_map.get(query_id)
            p = collection.get(pos_id)
            n = collection.get(neg_id)
            if q is None or p is None or n is None:
                continue
            examples.append(
                RetrievalExample(
                    query=q,
                    positive=p,
                    negatives=[n],
                    lang=lang_tag,
                )
            )
    return examples


# ---------------------------------------------------------------------------
# MIRACL: Hub TSV + streaming corpus (ru + datasets>=3; avoids full corpus in RAM)
# ---------------------------------------------------------------------------

_MIRACL_CORPUS_NUM_FILES: Dict[str, int] = {
    "ar": 5,
    "bn": 1,
    "en": 66,
    "es": 21,
    "fa": 5,
    "fi": 4,
    "fr": 30,
    "hi": 2,
    "id": 3,
    "ja": 14,
    "ko": 3,
    "ru": 20,
    "sw": 1,
    "te": 2,
    "th": 2,
    "zh": 10,
    "de": 32,
    "yo": 1,
}


def _miracl_topics_and_qrels_paths(lang: str, split: str) -> Tuple[str, str]:
    """Relative paths under ``miracl/miracl`` repo."""
    if split == "train":
        return (
            f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-train.tsv",
            f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-train.tsv",
        )
    if split == "dev":
        return (
            f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv",
            f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv",
        )
    raise ValueError(f"Unsupported MIRACL split for Hub load: {split}")


def _parse_miracl_topics(path: str) -> Dict[str, str]:
    qid2topic: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, topic = line.split("\t", 1)
            qid2topic[qid] = topic
    return qid2topic


def _parse_miracl_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels[qid][docid] = rel
    return dict(qrels)


def _load_miracl_corpus_subset(
    lang: str,
    needed_docids: Set[str],
    cache_dir: str,
) -> Dict[str, Tuple[str, str]]:
    """Stream ``docs-*.jsonl.gz`` and keep only ``needed_docids`` (title, text)."""
    from huggingface_hub import hf_hub_download

    n_files = _MIRACL_CORPUS_NUM_FILES.get(lang)
    if n_files is None:
        raise ValueError(f"Unknown MIRACL corpus language: {lang}")

    repo_corpus = "miracl/miracl-corpus"
    out: Dict[str, Tuple[str, str]] = {}
    if not needed_docids:
        return out

    remaining = set(needed_docids)
    prefix = f"miracl-corpus-v1.0-{lang}/docs-"

    for i in range(n_files):
        if not remaining:
            break
        fn = f"{prefix}{i}.jsonl.gz"
        local = hf_hub_download(
            repo_id=repo_corpus,
            filename=fn,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        with gzip.open(local, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                did = row["docid"]
                if did in remaining:
                    out[did] = (row.get("title") or "", row.get("text") or "")
                    remaining.discard(did)
        logger.debug("MIRACL corpus shard %s: %d docs collected, %d remaining", fn, len(out), len(remaining))

    if remaining:
        logger.warning(
            "MIRACL corpus: %d docids not found in shards (showing up to 5): %s",
            len(remaining),
            list(remaining)[:5],
        )
    return out


def _load_miracl_examples_from_hub_files(
    cfg: DataConfig,
    split: str,
) -> List[RetrievalExample]:
    """Load MIRACL without ``load_dataset`` (works for ``ru`` and ``datasets>=3``).

    Loads topics + qrels from Hub, then streams corpus JSONL shards only for
    docids referenced by the selected queries (bounded RAM).
    """
    lang = cfg.miracl_lang
    topics_path, qrels_path = _miracl_topics_and_qrels_paths(lang, split)

    from huggingface_hub import hf_hub_download

    repo = cfg.miracl_dataset
    base_kw = {"repo_id": repo, "repo_type": "dataset", "cache_dir": cfg.cache_dir}

    tp = hf_hub_download(filename=topics_path, **base_kw)
    qp = hf_hub_download(filename=qrels_path, **base_kw)

    qid2topic = _parse_miracl_topics(tp)
    qrels = _parse_miracl_qrels(qp)

    max_q: Optional[int] = None
    if split == "train" and cfg.max_train_samples is not None:
        max_q = cfg.max_train_samples
    if split == "dev" and cfg.max_val_samples is not None:
        max_q = cfg.max_val_samples

    ordered_qids = list(qid2topic.keys())
    if max_q is not None:
        ordered_qids = ordered_qids[:max_q]

    needed_docids: Set[str] = set()
    for qid in ordered_qids:
        for docid, rel in qrels.get(qid, {}).items():
            if rel == 1 or rel == 0:
                needed_docids.add(docid)

    logger.info(
        "MIRACL %s %s: %d queries, %d unique docids to resolve",
        lang,
        split,
        len(ordered_qids),
        len(needed_docids),
    )

    docid2doc = _load_miracl_corpus_subset(lang, needed_docids, cfg.cache_dir)

    examples: List[RetrievalExample] = []
    for qid in ordered_qids:
        query = qid2topic[qid]
        pos_docids = [d for d, r in qrels.get(qid, {}).items() if r == 1]
        neg_docids = [d for d, r in qrels.get(qid, {}).items() if r == 0]

        positives: List[str] = []
        for did in pos_docids:
            if did in docid2doc:
                _, text = docid2doc[did]
                positives.append(text)

        neg_texts: List[str] = []
        for did in neg_docids:
            if did in docid2doc:
                _, text = docid2doc[did]
                neg_texts.append(text)

        if not positives:
            continue

        examples.append(
            RetrievalExample(
                query=query,
                positive=positives[0],
                negatives=neg_texts,
                lang=lang,
            )
        )
    return examples


# ---------------------------------------------------------------------------
# BM25-based hard-negative miner
# ---------------------------------------------------------------------------

class BM25HardNegativeMiner:
    """Mine hard negatives from a document corpus using BM25 scoring.

    Uses ``rank_bm25`` for lightweight lexical scoring.  Documents that rank
    highly for a query but are *not* the gold positive are treated as hard
    negatives.
    """

    def __init__(self, corpus_texts: Sequence[str], tokenize_fn=None) -> None:
        from rank_bm25 import BM25Okapi

        self._tokenize = tokenize_fn or (lambda t: t.lower().split())
        tokenized = [self._tokenize(doc) for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus_texts = list(corpus_texts)

    def mine(
        self,
        query: str,
        positive_text: str,
        top_k: int = 30,
        num_negatives: int = 7,
    ) -> List[str]:
        """Return up to ``num_negatives`` hard-negative passages for *query*.

        Excludes the gold *positive_text* from results.
        """
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked_idxs = np.argsort(scores)[::-1]

        negatives: List[str] = []
        for idx in ranked_idxs[:top_k]:
            candidate = self.corpus_texts[idx]
            if candidate != positive_text:
                negatives.append(candidate)
            if len(negatives) >= num_negatives:
                break
        return negatives

# ---------------------------------------------------------------------------
# Core PyTorch Dataset
# ---------------------------------------------------------------------------

class RetrievalTripleDataset(Dataset):
    """PyTorch dataset that yields tokenized (query, positive, negative) triples.

    Supports three negative-sampling strategies:
      * ``random``  — uniformly sampled from the corpus
      * ``hard``    — pre-mined via BM25 or bi-encoder
      * ``mixed``   — combination of random and hard negatives
    In-batch negatives are handled at collation/loss level, not here.

    Parameters
    ----------
    examples : list[RetrievalExample]
        Pre-built examples with at least one negative per example.
    tokenizer : PreTrainedTokenizerFast
        HuggingFace tokenizer (XLM-RoBERTa).
    query_max_len : int
        Maximum token length for queries.
    doc_max_len : int
        Maximum token length for documents.
    """

    def __init__(
        self,
        examples: List[RetrievalExample],
        tokenizer: PreTrainedTokenizerFast,
        query_max_len: int = 32,
        doc_max_len: int = 180,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len

    def __len__(self) -> int:
        return len(self.examples)

    def _tokenize(self, text: str, max_len: int) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        neg = random.choice(ex.negatives) if ex.negatives else ""

        q_enc = self._tokenize(ex.query, self.query_max_len)
        p_enc = self._tokenize(ex.positive, self.doc_max_len)
        n_enc = self._tokenize(neg, self.doc_max_len)

        return {
            "query_input_ids": q_enc["input_ids"].squeeze(0),
            "query_attention_mask": q_enc["attention_mask"].squeeze(0),
            "pos_input_ids": p_enc["input_ids"].squeeze(0),
            "pos_attention_mask": p_enc["attention_mask"].squeeze(0),
            "neg_input_ids": n_enc["input_ids"].squeeze(0),
            "neg_attention_mask": n_enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Mixed-language batch sampler
# ---------------------------------------------------------------------------

class MixedLanguageBatchSampler(Sampler[List[int]]):
    """Yields batches with a controlled mix of Russian and English examples.

    Parameters
    ----------
    lang_labels : list[str]
        Per-example language tag (``"ru"`` or ``"en"``).
    batch_size : int
        Total batch size.
    en_ratio : float
        Target fraction of English examples in each batch (0–1).
    drop_last : bool
        Drop the final incomplete batch.
    """

    def __init__(
        self,
        lang_labels: List[str],
        batch_size: int,
        en_ratio: float = 0.3,
        drop_last: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.en_ratio = en_ratio
        self.drop_last = drop_last

        self.ru_indices = [i for i, l in enumerate(lang_labels) if l == "ru"]
        self.en_indices = [i for i, l in enumerate(lang_labels) if l == "en"]

        if not self.en_indices:
            logger.warning("No English examples found — batches will be monolingual Russian.")
        if not self.ru_indices:
            logger.warning("No Russian examples found — batches will be monolingual English.")

    def __iter__(self) -> Iterator[List[int]]:
        ru_pool = self.ru_indices.copy()
        en_pool = self.en_indices.copy()
        random.shuffle(ru_pool)
        random.shuffle(en_pool)

        n_en = max(1, int(self.batch_size * self.en_ratio)) if en_pool else 0
        n_ru = self.batch_size - n_en

        batches: List[List[int]] = []
        while len(ru_pool) >= n_ru:
            batch = ru_pool[:n_ru]
            ru_pool = ru_pool[n_ru:]

            if n_en > 0 and len(en_pool) >= n_en:
                batch.extend(en_pool[:n_en])
                en_pool = en_pool[n_en:]
            elif n_en > 0 and en_pool:
                batch.extend(en_pool)
                en_pool = []

            random.shuffle(batch)
            batches.append(batch)

        if not self.drop_last and ru_pool:
            batch = ru_pool
            if en_pool:
                batch.extend(en_pool)
            batches.append(batch)

        random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        n_en = max(1, int(self.batch_size * self.en_ratio)) if self.en_indices else 0
        n_ru = self.batch_size - n_en
        if n_ru == 0:
            return 0
        total = len(self.ru_indices) // n_ru
        if not self.drop_last and len(self.ru_indices) % n_ru:
            total += 1
        return total


# ---------------------------------------------------------------------------
# High-level data-loading helpers
# ---------------------------------------------------------------------------

def _load_mmarco_examples(
    cfg: DataConfig,
    split: str = "train",
    lang: str = "russian",
) -> List[RetrievalExample]:
    """Load mMARCO triples for a given language split.

    Uses Hub TSV downloads by default so ``datasets>=3`` works (no legacy
    ``mmarco.py`` script).  Only the **train** triple split exists in the
    published files; other splits return an empty list.
    """
    if split != "train":
        logger.warning(
            "mMARCO triples are only available for split=train (got %r); returning [].",
            split,
        )
        return []

    if cfg.mmarco_use_hub_files:
        try:
            cap = _resolve_mmarco_triple_cap(cfg)
            examples = _load_mmarco_examples_from_hub_files(lang, cfg.cache_dir, cap)
        except Exception as e:
            logger.warning(
                "mMARCO Hub-file load failed (%s); trying load_dataset (needs datasets<3 or parquet).",
                e,
            )
            examples = _load_mmarco_examples_hf_dataset(cfg, lang, split)
    else:
        examples = _load_mmarco_examples_hf_dataset(cfg, lang, split)

    if cfg.max_train_samples:
        examples = examples[: min(cfg.max_train_samples, len(examples))]
    return examples


def _load_mmarco_examples_hf_dataset(
    cfg: DataConfig,
    lang: str,
    split: str,
) -> List[RetrievalExample]:
    """Legacy path: ``datasets.load_dataset`` (works with ``datasets<3`` or if Hub adds parquet)."""
    from datasets import load_dataset

    load_kw: Dict[str, object] = {
        "path": cfg.mmarco_dataset,
        "name": lang,
        "split": split,
        "cache_dir": cfg.cache_dir,
    }
    if cfg.mmarco_revision:
        load_kw["revision"] = cfg.mmarco_revision
    ds = load_dataset(**load_kw)

    examples: List[RetrievalExample] = []
    for row in ds:
        examples.append(
            RetrievalExample(
                query=row["query"],
                positive=row["positive"],
                negatives=[row["negative"]] if row.get("negative") is not None else [],
                lang="ru" if lang == "russian" else "en",
            )
        )
    return examples


def _load_miracl_examples(
    cfg: DataConfig,
    split: str = "train",
) -> List[RetrievalExample]:
    """Load MIRACL as RetrievalExamples.

    Default: Hub TSV + streaming corpus (``miracl_use_hub_files=True``) — works for
    **Russian** and ``datasets>=3`` (no legacy ``miracl.py``).  Falls back to
    ``load_dataset(..., revision=refs/convert/parquet)`` only when Hub load is off
    and the language exists on the parquet branch (``ru`` is **not** there).
    """
    if cfg.miracl_use_hub_files:
        try:
            return _load_miracl_examples_from_hub_files(cfg, split)
        except Exception as e:
            logger.warning("MIRACL Hub-file load failed (%s); trying load_dataset.", e)

    from datasets import load_dataset

    load_kw: Dict[str, object] = {
        "path": cfg.miracl_dataset,
        "name": cfg.miracl_lang,
        "split": split,
        "cache_dir": cfg.cache_dir,
    }
    if cfg.miracl_revision:
        load_kw["revision"] = cfg.miracl_revision

    ds = load_dataset(**load_kw)

    if cfg.max_train_samples and split == "train":
        ds = ds.select(range(min(cfg.max_train_samples, len(ds))))
    if cfg.max_val_samples and split == "dev":
        ds = ds.select(range(min(cfg.max_val_samples, len(ds))))

    lang_tag = cfg.miracl_lang if cfg.miracl_lang in ("ru", "en") else "ru"

    examples: List[RetrievalExample] = []
    for row in ds:
        positives = [p["text"] for p in row.get("positive_passages", []) if p.get("text")]
        neg_texts = [n["text"] for n in row.get("negative_passages", []) if n.get("text")]

        if not positives:
            continue

        examples.append(
            RetrievalExample(
                query=row["query"],
                positive=positives[0],
                negatives=neg_texts if neg_texts else [],
                lang=lang_tag,
            )
        )
    return examples


def enrich_with_hard_negatives(
    examples: List[RetrievalExample],
    corpus_texts: Sequence[str],
    cfg: DataConfig,
    device: str = "cpu",
) -> List[RetrievalExample]:
    """Add BM25 or bi-encoder hard negatives to each example."""
    if cfg.hard_negative_source == "bm25":
        miner = BM25HardNegativeMiner(corpus_texts)
    else:
        miner = BiEncoderHardNegativeMiner(corpus_texts, device=device)

    for ex in examples:
        hard_negs = miner.mine(ex.query, ex.positive, num_negatives=cfg.num_hard_negatives)
        ex.negatives = hard_negs + ex.negatives
    return examples


def build_dataloader(
    examples: List[RetrievalExample],
    tokenizer: PreTrainedTokenizerFast,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg_batch_size: int = 32,
    num_workers: int = 4,
    is_train: bool = True,
) -> DataLoader:
    """Construct a DataLoader with optional mixed-language batch sampling."""
    dataset = RetrievalTripleDataset(
        examples=examples,
        tokenizer=tokenizer,
        query_max_len=model_cfg.query_max_len,
        doc_max_len=model_cfg.doc_max_len,
    )

    if is_train and data_cfg.mixed_lang_ratio > 0:
        lang_labels = [ex.lang for ex in examples]
        sampler = MixedLanguageBatchSampler(
            lang_labels=lang_labels,
            batch_size=train_cfg_batch_size,
            en_ratio=data_cfg.mixed_lang_ratio,
            drop_last=True,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    return DataLoader(
        dataset,
        batch_size=train_cfg_batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
        pin_memory=True,
    )


def load_training_data(
    tokenizer: PreTrainedTokenizerFast,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Top-level helper: load train + validation DataLoaders.

    Returns
    -------
    train_loader, val_loader
    """
    logger.info("Loading mMARCO Russian training data…")
    ru_train = _load_mmarco_examples(data_cfg, split="train", lang="russian")

    en_train: List[RetrievalExample] = []
    if data_cfg.mixed_lang_ratio > 0:
        logger.info("Loading mMARCO English data for mixed batching…")
        n_en = int(len(ru_train) * data_cfg.mixed_lang_ratio / (1 - data_cfg.mixed_lang_ratio))
        en_cfg = DataConfig(**{**data_cfg.__dict__, "max_train_samples": n_en})
        en_train = _load_mmarco_examples(en_cfg, split="train", lang="english")

    logger.info("Loading MIRACL Russian training data…")
    miracl_train = _load_miracl_examples(data_cfg, split="train")

    train_examples = ru_train + en_train + miracl_train
    random.shuffle(train_examples)

    logger.info("Total training examples: %d", len(train_examples))

    val_examples = _load_miracl_examples(data_cfg, split="dev")

    train_loader = build_dataloader(
        train_examples, tokenizer, model_cfg, data_cfg,
        train_cfg_batch_size=train_batch_size,
        num_workers=num_workers,
        is_train=True,
    )
    val_loader = build_dataloader(
        val_examples, tokenizer, model_cfg, data_cfg,
        train_cfg_batch_size=train_batch_size,
        num_workers=num_workers,
        is_train=False,
    )
    return train_loader, val_loader
