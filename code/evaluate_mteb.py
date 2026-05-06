from __future__ import annotations

import contextlib
import logging
import sys
import torch
import numpy as np
from typing import List, Dict, Union, Any, Optional, Sequence, Set, Tuple
from tqdm import tqdm

try:
    from mteb.types import PromptType, Array
    from mteb import TaskMetadata
    from torch.utils.data import DataLoader
except ImportError:
    pass

from rank_bm25 import BM25Okapi

from config import ModelConfig
from modeling import ColBERTModel, build_model_and_tokenizer
from mteb.models import EncoderProtocol

logger = logging.getLogger(__name__)


def _tokenize_for_bm25(texts: Sequence[str]) -> List[List[str]]:
    """Lightweight tokenization for BM25 (first stage)."""
    return [t.lower().split() for t in texts]


def _bm25_top_k_candidates(
    query_texts: Sequence[str],
    corpus_texts: Sequence[str],
    top_k: int,
) -> List[Set[int]]:
    """Return for each query a set of corpus indices with highest BM25 scores."""
    n_c = len(corpus_texts)
    if n_c == 0:
        return [set() for _ in query_texts]

    k = min(top_k, n_c)
    tokenized_corpus = _tokenize_for_bm25(corpus_texts)
    tokenized_queries = _tokenize_for_bm25(query_texts)
    bm25 = BM25Okapi(tokenized_corpus)

    out: List[Set[int]] = []
    for q_toks in tokenized_queries:
        scores = bm25.get_scores(q_toks)
        scores = np.asarray(scores, dtype=np.float64)
        if k >= n_c:
            idx = np.arange(n_c)
        else:
            # Partial selection: O(n) instead of full sort
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        out.append(set(int(i) for i in idx[:k]))
    return out

class MTEBDenseWrapper(EncoderProtocol):
    """Wrapper to integrate dense models into the MTEB framework with optional two-stage BM25 retrieval."""

    mteb_model_meta = None

    def __init__(
        self,
        model: Any,
        two_stage: bool = False,
        bm25_top_k: int = 2000,
    ):
        self.model = model
        self.two_stage = two_stage
        self.bm25_top_k = bm25_top_k

        # Filled during encoding (same order as embedding lists) for BM25 stage
        self._last_query_texts: Optional[List[str]] = None
        self._last_corpus_texts: Optional[List[str]] = None

        self.model_name = "dense-baseline"
        self.revision = "main"

    def encode(
        self,
        inputs: Any,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        is_query = (prompt_type == "query") or kwargs.get("is_query", False)
        
        texts = []
        for batch_or_doc in inputs:
            if isinstance(batch_or_doc, dict):
                text_val = batch_or_doc.get("text", "")
                title_val = batch_or_doc.get("title", "")
                
                if isinstance(text_val, list):
                    titles = title_val if isinstance(title_val, list) else [""] * len(text_val)
                    for t, txt in zip(titles, text_val):
                        t_str = " ".join(str(x) for x in t) if isinstance(t, list) else str(t)
                        txt_str = " ".join(str(x) for x in txt) if isinstance(txt, list) else str(txt)
                        texts.append((t_str + " " + txt_str).strip())
                else:
                    t_str = " ".join(str(x) for x in title_val) if isinstance(title_val, list) else str(title_val)
                    txt_str = " ".join(str(x) for x in text_val) if isinstance(text_val, list) else str(text_val)
                    texts.append((t_str + " " + txt_str).strip())
            elif isinstance(batch_or_doc, str):
                texts.append(batch_or_doc)
            elif isinstance(batch_or_doc, (list, tuple)):
                for doc in batch_or_doc:
                    if isinstance(doc, dict):
                        title = doc.get("title", "")
                        if isinstance(title, list): title = " ".join(str(t) for t in title)
                        text = doc.get("text", "")
                        if isinstance(text, list): text = " ".join(str(t) for t in text)
                        texts.append((str(title) + " " + str(text)).strip())
                    elif isinstance(doc, str):
                        texts.append(doc)
                
        # Handle prefix for e5-base
        if "e5" in str(self.model).lower() or hasattr(self.model, "model_card_data") and "e5" in str(self.model.model_card_data).lower():
             prefixed_texts = ["query: " + t if is_query else "passage: " + t for t in texts]
        else:
             prefixed_texts = texts

        if is_query:
            if self._last_query_texts is None:
                self._last_query_texts = []
            self._last_query_texts.extend(texts)
        else:
            if self._last_corpus_texts is None:
                self._last_corpus_texts = []
            self._last_corpus_texts.extend(texts)

        # Handle sentence-transformers API directly
        return self.model.encode(
             prefixed_texts,
             batch_size=kwargs.get("batch_size", 32),
             show_progress_bar=False,
             convert_to_numpy=True,
             normalize_embeddings=True
        )

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        self._last_query_texts = list(queries)
        return self.encode(queries, is_query=True, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: Union[List[Dict[str, str]], List[str]], batch_size: int = 16, **kwargs) -> np.ndarray:
        texts = []
        for doc in corpus:
            if isinstance(doc, dict):
                title = doc.get("title", "")
                if isinstance(title, list): title = " ".join(str(t) for t in title)
                text = doc.get("text", "")
                if isinstance(text, list): text = " ".join(str(t) for t in text)
                texts.append((str(title) + " " + str(text)).strip())
            else:
                texts.append(doc)
        self._last_corpus_texts = texts
        return self.encode(texts, is_query=False, batch_size=batch_size, **kwargs)

    def similarity(self, queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Computes dot product similarity between queries and corpus documents.

        With ``two_stage=True``, BM25 selects up to ``bm25_top_k`` candidate docs per
        query; dot product is computed only for those (others stay at ``-1e9``).
        """
        num_q = queries.shape[0]
        num_c = corpus.shape[0]
        neg_fill = -1e9

        use_two_stage = (
            self.two_stage
            and self._last_query_texts is not None
            and self._last_corpus_texts is not None
            and len(self._last_query_texts) == num_q
            and len(self._last_corpus_texts) == num_c
        )

        if self.two_stage and not use_two_stage:
            logger.warning(
                "Two-stage BM25 enabled but cached texts are missing or length-mismatched "
                "with embeddings; using dense dot product."
            )

        if not use_two_stage:
            scores = np.dot(queries, corpus.T)
        else:
            scores = np.full((num_q, num_c), neg_fill, dtype=np.float32)
            cand_sets = _bm25_top_k_candidates(
                self._last_query_texts,
                self._last_corpus_texts,
                self.bm25_top_k,
            )
            total_pairs = sum(len(s) for s in cand_sets)
            logger.info(
                "Dense Two-stage: BM25 selected %d doc-query pairs (top-%d per query, |corpus|=%d).",
                total_pairs,
                min(self.bm25_top_k, num_c),
                num_c,
            )
            for i in tqdm(range(num_q), desc="Dense DotProduct (two-stage)"):
                cand = sorted(list(cand_sets[i]))
                if not cand:
                    continue
                c_docs = corpus[cand]
                row_scores = np.dot(queries[i:i+1], c_docs.T)[0]
                scores[i, cand] = row_scores

        self._last_query_texts = None
        self._last_corpus_texts = None

        return scores

    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        return np.diag(self.similarity(embeddings1, embeddings2))

class MTEBColBERTWrapper(EncoderProtocol):
    """Wrapper to integrate ColBERT-RU into the MTEB (ruMTEB / RusBEIR) framework.
    
    Since ColBERT is a late-interaction multi-vector model, MTEB requires 
    the wrapper to return Lists of Tensors (or 3D arrays) for encode_queries 
    and encode_corpus, and must implement a custom `similarity` method.
    
    Optional **two-stage retrieval**: BM25 pre-selects ``bm25_top_k`` docs per query,
    then MaxSim is computed only on those candidates (others get a large negative score).
    Raw texts are cached from the last ``_encode`` / ``encode_queries`` / ``encode_corpus`` calls.
    """

    mteb_model_meta = None

    def __init__(
        self,
        model: ColBERTModel,
        tokenizer,
        device: torch.device,
        *,
        two_stage: bool = False,
        bm25_top_k: int = 2000,
        maxsim_doc_batch: int = 512,
        query_batch_size: int = 8,
        maxsim_fp16: bool = True,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        self.two_stage = two_stage
        self.bm25_top_k = bm25_top_k
        self.maxsim_doc_batch = maxsim_doc_batch
        self.query_batch_size = query_batch_size
        self.maxsim_fp16 = maxsim_fp16 and device.type == "cuda"

        # Filled during encoding (same order as embedding lists) for BM25 stage
        self._last_query_texts: Optional[List[str]] = None
        self._last_corpus_texts: Optional[List[str]] = None
        
        # MTEB Encoder protocol requirements
        self.model_name = "colbert-ru"
        self.revision = "main"

    def encode(
        self,
        inputs: Any,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: str | None = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """Encodes the given sentences using the encoder."""
        is_query = (prompt_type == "query") or kwargs.get("is_query", False)
        
        texts = []
        for batch_or_doc in inputs:
            if isinstance(batch_or_doc, dict):
                # Handle possible batches from DataLoader
                text_val = batch_or_doc.get("text", "")
                title_val = batch_or_doc.get("title", "")
                
                if isinstance(text_val, list):
                    titles = title_val if isinstance(title_val, list) else [""] * len(text_val)
                    for t, txt in zip(titles, text_val):
                        t_str = " ".join(str(x) for x in t) if isinstance(t, list) else str(t)
                        txt_str = " ".join(str(x) for x in txt) if isinstance(txt, list) else str(txt)
                        texts.append((t_str + " " + txt_str).strip())
                else:
                    t_str = " ".join(str(x) for x in title_val) if isinstance(title_val, list) else str(title_val)
                    txt_str = " ".join(str(x) for x in text_val) if isinstance(text_val, list) else str(text_val)
                    texts.append((t_str + " " + txt_str).strip())
            elif isinstance(batch_or_doc, str):
                texts.append(batch_or_doc)
            elif isinstance(batch_or_doc, (list, tuple)):
                for doc in batch_or_doc:
                    if isinstance(doc, dict):
                        title = doc.get("title", "")
                        if isinstance(title, list): title = " ".join(str(t) for t in title)
                        text = doc.get("text", "")
                        if isinstance(text, list): text = " ".join(str(t) for t in text)
                        texts.append((str(title) + " " + str(text)).strip())
                    elif isinstance(doc, str):
                        texts.append(doc)
                
        batch_size = kwargs.get("batch_size", 32)
        # Accumulate texts across chunked ``encode`` calls (order must match embeddings).
        if is_query:
            if self._last_query_texts is None:
                self._last_query_texts = []
            self._last_query_texts.extend(texts)
        else:
            if self._last_corpus_texts is None:
                self._last_corpus_texts = []
            self._last_corpus_texts.extend(texts)
        return self._encode(texts, batch_size, is_query)
        
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """Encode queries into multi-vector representations. (Legacy interface)"""
        self._last_query_texts = list(queries)
        return self._encode(queries, batch_size, is_query=True)

    def encode_corpus(self, corpus: Union[List[Dict[str, str]], List[str]], batch_size: int = 16, **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """Encode corpus documents into multi-vector representations. (Legacy interface)"""
        texts = []
        for doc in corpus:
            if isinstance(doc, dict):
                title = doc.get("title", "")
                if isinstance(title, list): title = " ".join(str(t) for t in title)
                text = doc.get("text", "")
                if isinstance(text, list): text = " ".join(str(t) for t in text)
                texts.append((str(title) + " " + str(text)).strip())
            else:
                texts.append(doc)
        self._last_corpus_texts = texts
        return self._encode(texts, batch_size, is_query=False)

    @torch.no_grad()
    def _encode(self, texts: List[str], batch_size: int, is_query: bool) -> List[np.ndarray]:
        """Encode texts to lists of multi-vector embeddings.
        Returns a list of arrays, where each array is of shape (Length, Dim).
        """
        max_length = self.model.cfg.query_max_len if is_query else self.model.cfg.doc_max_len
        all_embs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding" + (" queries" if is_query else " corpus")):
            batch_texts = texts[i:i + batch_size]
            
            enc = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length", # or "longest"
                truncation=True,
                return_tensors="pt",
            )
            
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            
            if is_query:
                embs, active_mask = self.model.encode_query(input_ids, attention_mask)
            else:
                embs, active_mask = self.model.encode_doc(input_ids, attention_mask)
            
            # embs: (B, L, D), active_mask: (B, L)
            # Remove padding tokens for each item individually
            for b in range(embs.size(0)):
                valid_len = active_mask[b].sum().item()
                # For ColBERT, we only keep the active tokens
                item_embs = embs[b, :valid_len, :].cpu().numpy().astype(np.float32)
                all_embs.append(item_embs)
                
        return all_embs

    def _maxsim_block(
        self,
        q_list: List[np.ndarray],
        c_list: List[np.ndarray],
    ) -> np.ndarray:
        """Batched MaxSim: ``q_list`` (each Lq×D) vs ``c_list`` (each Ld×D).

        Returns an array of shape ``(len(q_list), len(c_list))``.
        """
        if not q_list or not c_list:
            return np.zeros((len(q_list), len(c_list)), dtype=np.float32)

        Bq = len(q_list)
        Bc = len(c_list)
        max_lq = max(q.shape[0] for q in q_list)
        max_ld = max(doc.shape[0] for doc in c_list)
        D = q_list[0].shape[1]

        dev = self.device
        Q = torch.zeros(Bq, max_lq, D, device=dev, dtype=torch.float32)
        MQ = torch.zeros(Bq, max_lq, dtype=torch.bool, device=dev)
        for b, q in enumerate(q_list):
            lq = int(q.shape[0])
            Q[b, :lq].copy_(torch.from_numpy(q).to(dev, dtype=torch.float32))
            MQ[b, :lq] = True

        C = torch.zeros(Bc, max_ld, D, device=dev, dtype=torch.float32)
        MC = torch.zeros(Bc, max_ld, dtype=torch.bool, device=dev)
        for c, doc in enumerate(c_list):
            ld = int(doc.shape[0])
            C[c, :ld].copy_(torch.from_numpy(doc).to(dev, dtype=torch.float32))
            MC[c, :ld] = True

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.maxsim_fp16
            else contextlib.nullcontext()
        )

        with torch.no_grad(), amp_ctx:
            # (Bq, Bc, Lq, Ld)
            sim = torch.einsum("bqd,cjd->bcqj", Q, C)
            neg = torch.tensor(-1e4, device=dev, dtype=sim.dtype)
            sim = sim.masked_fill(~MC.unsqueeze(0).unsqueeze(2), neg)
            max_over_doc, _ = sim.max(dim=3)
            max_over_doc = max_over_doc.masked_fill(~MQ.unsqueeze(1), 0.0)
            scores = max_over_doc.sum(dim=2)

        return scores.float().cpu().numpy().astype(np.float32)

    def similarity(self, queries: List[np.ndarray], corpus: List[np.ndarray]) -> np.ndarray:
        """Computes MaxSim similarity between queries and corpus documents.

        With ``two_stage=True``, BM25 selects up to ``bm25_top_k`` candidate docs per
        query; MaxSim is computed only for those (others stay at ``-1e9``).

        Otherwise, a dense score matrix is computed with batched queries and batched
        documents (see ``query_batch_size`` and ``maxsim_doc_batch``).
        """
        num_q = len(queries)
        num_c = len(corpus)
        neg_fill = -1e9

        use_two_stage = (
            self.two_stage
            and self._last_query_texts is not None
            and self._last_corpus_texts is not None
            and len(self._last_query_texts) == num_q
            and len(self._last_corpus_texts) == num_c
        )
        if self.two_stage and not use_two_stage:
            logger.warning(
                "Two-stage BM25 enabled but cached texts are missing or length-mismatched "
                "with embeddings; using dense MaxSim."
            )

        scores = np.full((num_q, num_c), neg_fill, dtype=np.float32)

        try:
            if use_two_stage:
                cand_sets = _bm25_top_k_candidates(
                    self._last_query_texts,
                    self._last_corpus_texts,
                    self.bm25_top_k,
                )
                total_pairs = sum(len(s) for s in cand_sets)
                logger.info(
                    "Two-stage: BM25 selected %d doc-query pairs (top-%d per query, |corpus|=%d).",
                    total_pairs,
                    min(self.bm25_top_k, num_c),
                    num_c,
                )
                doc_bs = self.maxsim_doc_batch
                for i in tqdm(range(num_q), desc="MaxSim (two-stage)"):
                    cand = sorted(cand_sets[i])
                    if not cand:
                        continue
                    qi = queries[i]
                    for c_start in range(0, len(cand), doc_bs):
                        chunk_idx = cand[c_start : c_start + doc_bs]
                        c_docs = [corpus[j] for j in chunk_idx]
                        row = self._maxsim_block([qi], c_docs)[0]
                        scores[i, chunk_idx] = row
            else:
                doc_bs = self.maxsim_doc_batch
                q_bs = max(1, self.query_batch_size)
                for q_start in tqdm(
                    range(0, num_q, q_bs),
                    desc="MaxSim (dense, query batches)",
                ):
                    q_batch = queries[q_start : q_start + q_bs]
                    for c_start in range(0, num_c, doc_bs):
                        c_end = min(c_start + doc_bs, num_c)
                        c_chunk = corpus[c_start:c_end]
                        block = self._maxsim_block(q_batch, c_chunk)
                        scores[q_start : q_start + len(q_batch), c_start:c_end] = block
        finally:
            self._last_query_texts = None
            self._last_corpus_texts = None

        return scores

    def similarity_pairwise(self, embeddings1: List[np.ndarray], embeddings2: List[np.ndarray]) -> np.ndarray:
        """Pairwise similarity for MTEB Encoder protocol."""
        return np.diag(self.similarity(embeddings1, embeddings2))


DEFAULT_MTEB_TASKS: Sequence[str] = (
    "RuBQRetrieval",
    "RuSciBenchCiteRetrieval",
    "XQuADRetrieval",
)


COLBERT_XM_CHECKPOINT_ALIASES: Sequence[str] = ("antoinelouis/colbert-xm", "colbert-xm")
DEFAULT_COLBERT_XM_XMOD_LANGUAGE: str = "ru_RU"


def _is_colbert_xm_hub_checkpoint(checkpoint: Optional[str]) -> bool:
    if not checkpoint or checkpoint.lower() == "none":
        return False
    return checkpoint in COLBERT_XM_CHECKPOINT_ALIASES


def run_mteb_evaluation(
    *,
    checkpoint: Optional[str] = None,
    encoder_name: str = "xlm-roberta-base",
    output_folder: str = "./mteb_results",
    tasks: Optional[Sequence[str]] = None,
    batch_size: int = 128,
    query_max_len: int = 64,
    doc_max_len: int = 256,
    two_stage: bool = True,
    bm25_top_k: int = 2000,
    maxsim_doc_batch: int = 512,
    query_batch_size: int = 8,
    maxsim_fp16: bool = True,
    colbert_xm_faithful: bool = False,
    xmod_default_language: str = DEFAULT_COLBERT_XM_XMOD_LANGUAGE,
) -> None:
    import mteb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on %s", device)

    task_list = list(tasks) if tasks is not None else list(DEFAULT_MTEB_TASKS)

    use_faithful_colbert_xm = colbert_xm_faithful and _is_colbert_xm_hub_checkpoint(checkpoint)

    encoder_effective = encoder_name
    if use_faithful_colbert_xm:
        encoder_effective = "facebook/xmod-base"

    model_cfg = ModelConfig(
        encoder_name=encoder_effective,
        embedding_dim=128,
        query_max_len=query_max_len,
        doc_max_len=doc_max_len,
    )

    model, tokenizer = build_model_and_tokenizer(model_cfg)
    if checkpoint and checkpoint.lower() != "none":
        logger.info("Loading ColBERT-RU from %s", checkpoint)

        if _is_colbert_xm_hub_checkpoint(checkpoint):
            from huggingface_hub import hf_hub_download
            import safetensors.torch

            logger.info("Downloading antoinelouis/colbert-xm from HuggingFace...")
            model_path = hf_hub_download("antoinelouis/colbert-xm", "model.safetensors")
            hf_state_dict = safetensors.torch.load_file(model_path)

            new_state_dict = {}
            for k, v in hf_state_dict.items():
                if k.startswith("roberta."):
                    new_k = k.replace("roberta.", "encoder.backbone.", 1)
                    new_state_dict[new_k] = v
                elif k == "linear.weight":
                    new_state_dict["encoder.linear.weight"] = v
                else:
                    new_state_dict[k] = v

            incompat = model.load_state_dict(new_state_dict, strict=False)
            logger.info(
                "colbert-xm: %d keys in remapped checkpoint; missing_keys=%d unexpected_keys=%d",
                len(new_state_dict),
                len(incompat.missing_keys),
                len(incompat.unexpected_keys),
            )
            if incompat.missing_keys:
                logger.warning(
                    "colbert-xm missing_keys (in model, not in checkpoint), first 40: %s",
                    incompat.missing_keys[:40],
                )
            if incompat.unexpected_keys:
                logger.warning(
                    "colbert-xm unexpected_keys (in checkpoint, not used by model), first 40: %s",
                    incompat.unexpected_keys[:40],
                )
            logger.info("Successfully loaded and mapped antoinelouis/colbert-xm weights!")
            if use_faithful_colbert_xm:
                bb = model.encoder.backbone
                if hasattr(bb, "set_default_language"):
                    bb.set_default_language(xmod_default_language)
                    logger.info(
                        "ColBERT-XM faithful: set_default_language(%r) — языковые адаптеры X-MOD задействованы при forward.",
                        xmod_default_language,
                    )
                else:
                    logger.warning(
                        "colbert-xm-faithful: ожидался XmodModel с set_default_language, получен %s.",
                        type(bb).__name__,
                    )

        else:
            ckpt = torch.load(checkpoint, map_location="cpu")
            if any(k.startswith("module.") for k in ckpt["model_state_dict"].keys()):
                state_dict = {}
                for k, v in ckpt["model_state_dict"].items():
                    if k.startswith("module."):
                        new_k = k[len("module.") :]
                    else:
                        new_k = k
                    state_dict[new_k] = v
                ckpt["model_state_dict"] = state_dict
     
            model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)

    mteb_model = MTEBColBERTWrapper(
        model,
        tokenizer,
        device=device,
        two_stage=two_stage,
        bm25_top_k=bm25_top_k,
        maxsim_doc_batch=maxsim_doc_batch,
        query_batch_size=query_batch_size,
        maxsim_fp16=maxsim_fp16,
    )
    logger.info(
        "MTEB wrapper: two_stage=%s bm25_top_k=%d maxsim_doc_batch=%d query_batch_size=%d maxsim_fp16=%s",
        mteb_model.two_stage,
        mteb_model.bm25_top_k,
        mteb_model.maxsim_doc_batch,
        mteb_model.query_batch_size,
        mteb_model.maxsim_fp16,
    )

    logger.info("Fetching MTEB tasks: %s", task_list)
    evaluation_tasks = mteb.get_tasks(tasks=task_list, languages=["rus"])

    logger.info("Starting MTEB evaluation...")
    evaluation = mteb.MTEB(tasks=evaluation_tasks)
    evaluation.run(
        mteb_model,
        output_folder=output_folder,
        encode_kwargs={"batch_size": batch_size},
    )

    logger.info("Evaluation complete! Results saved in %s", output_folder)


RUMTEB_RETRIEVAL_FOUR_TASKS: Sequence[str] = (
    "RuSciBenchCiteRetrieval",
    "RuBQRetrieval",
    "XQuADRetrieval",
    "PublicHealthQA",
)

DENSE_BASELINE_MODEL_IDS: Dict[str, str] = {
    "E5-base": "intfloat/multilingual-e5-base"
}


def run_dense_mteb_retrieval(
    model_id: str,
    *,
    output_folder: str,
    tasks: Optional[Sequence[str]] = None,
    batch_size: int = 128,
    languages: Optional[Sequence[str]] = None,
    two_stage: bool = True,
    bm25_top_k: int = 2000,
) -> None:
    import mteb

    task_list = list(tasks) if tasks is not None else list(RUMTEB_RETRIEVAL_FOUR_TASKS)
    langs = list(languages) if languages is not None else ["rus"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dense MTEB retrieval: model_id=%s device=%s tasks=%s languages=%s, two_stage=%s, bm25_top_k=%d", model_id, device, task_list, langs, two_stage, bm25_top_k)

    from sentence_transformers import SentenceTransformer
    base_model = SentenceTransformer(model_id, device=str(device))
    
    mteb_model = MTEBDenseWrapper(
        base_model,
        two_stage=two_stage,
        bm25_top_k=bm25_top_k
    )

    evaluation_tasks = mteb.get_tasks(tasks=task_list, languages=langs)
    evaluation = mteb.MTEB(tasks=evaluation_tasks)
    logger.info("Starting MTEB dense evaluation → %s", output_folder)
    evaluation.run(
        mteb_model,
        output_folder=output_folder,
        encode_kwargs={"batch_size": batch_size},
    )
    logger.info("Dense evaluation complete! Results in %s", output_folder)
