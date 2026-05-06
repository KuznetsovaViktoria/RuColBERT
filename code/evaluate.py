from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from config import EvalConfig, ModelConfig, PipelineConfig

logger = logging.getLogger(__name__)


# ======================================================================
# 1. IR Quality Metrics
# ======================================================================

class IRMetrics:
    """Compute standard IR metrics from ranked result lists and qrels.

    Uses ``pytrec_eval`` when available, otherwise falls back to a custom
    pure-Python implementation.

    Parameters
    ----------
    qrels : dict[str, dict[str, int]]
        ``{query_id: {doc_id: relevance_grade, …}, …}``
    k_values : list[int]
        Cut-off values for *@k* metrics.
    """

    def __init__(
        self,
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int] | None = None,
    ) -> None:
        self.qrels = qrels
        self.k_values = k_values or [1, 5, 10, 20, 100]

    # ------------------------------------------------------------------
    # pytrec_eval path
    # ------------------------------------------------------------------

    def _try_pytrec(
        self,
        run: Dict[str, Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        try:
            import pytrec_eval

            metrics_set: set[str] = set()
            for k in self.k_values:
                metrics_set |= {
                    f"map_cut_{k}",
                    f"ndcg_cut_{k}",
                    f"recall_{k}",
                }
            metrics_set.add("recip_rank")

            evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, list(metrics_set))
            per_query = evaluator.evaluate(run)

            aggregated: Dict[str, float] = {}
            for metric in metrics_set:
                vals = [per_query[qid].get(metric, 0.0) for qid in per_query]
                aggregated[metric] = float(np.mean(vals)) if vals else 0.0
            return aggregated
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Pure-Python fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _dcg(relevances: List[int], k: int) -> float:
        return sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(relevances[:k])
        )

    def _ndcg_at_k(
        self,
        ranked_docs: List[str],
        gold: Dict[str, int],
        k: int,
    ) -> float:
        rels = [gold.get(d, 0) for d in ranked_docs[:k]]
        dcg = self._dcg(rels, k)
        ideal = sorted(gold.values(), reverse=True)[:k]
        idcg = self._dcg(ideal, k)
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _ap_at_k(ranked_docs: List[str], gold: Dict[str, int], k: int) -> float:
        hits = 0
        precision_sum = 0.0
        for i, doc in enumerate(ranked_docs[:k]):
            if gold.get(doc, 0) > 0:
                hits += 1
                precision_sum += hits / (i + 1)
        n_relevant = sum(1 for v in gold.values() if v > 0)
        return precision_sum / min(n_relevant, k) if n_relevant > 0 else 0.0

    @staticmethod
    def _recall_at_k(ranked_docs: List[str], gold: Dict[str, int], k: int) -> float:
        relevant = {d for d, v in gold.items() if v > 0}
        if not relevant:
            return 0.0
        retrieved_relevant = sum(1 for d in ranked_docs[:k] if d in relevant)
        return retrieved_relevant / len(relevant)

    @staticmethod
    def _rr(ranked_docs: List[str], gold: Dict[str, int]) -> float:
        for i, doc in enumerate(ranked_docs):
            if gold.get(doc, 0) > 0:
                return 1.0 / (i + 1)
        return 0.0

    def _fallback_evaluate(
        self,
        run: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        results: Dict[str, List[float]] = {}

        for qid, doc_scores in run.items():
            gold = self.qrels.get(qid, {})
            ranked = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)

            for k in self.k_values:
                results.setdefault(f"map_cut_{k}", []).append(self._ap_at_k(ranked, gold, k))
                results.setdefault(f"ndcg_cut_{k}", []).append(self._ndcg_at_k(ranked, gold, k))
                results.setdefault(f"recall_{k}", []).append(self._recall_at_k(ranked, gold, k))

            results.setdefault("recip_rank", []).append(self._rr(ranked, gold))

        return {metric: float(np.mean(vals)) for metric, vals in results.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        run: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Evaluate *run* against stored qrels.

        Parameters
        ----------
        run : {query_id: {doc_id: score, …}, …}

        Returns
        -------
        Aggregated metric dict.
        """
        result = self._try_pytrec(run)
        if result is not None:
            return result
        logger.info("pytrec_eval not available — using fallback implementation")
        return self._fallback_evaluate(run)

    @staticmethod
    def results_to_run(
        results: Dict[str, List[Tuple[str, float]]],
    ) -> Dict[str, Dict[str, float]]:
        """Convert retriever output ``{qid: [(doc_id, score)]}`` to pytrec run
        format ``{qid: {doc_id: score}}``."""
        return {
            qid: {doc_id: score for doc_id, score in doc_scores}
            for qid, doc_scores in results.items()
        }


# ======================================================================
# 2. System-Level Metrics
# ======================================================================

class SystemMetrics:
    """Measure index size, latency, throughput, and vectors-per-document."""

    def __init__(self, index_dir: str) -> None:
        self.index_dir = Path(index_dir)

    def index_size_mb(self) -> float:
        emb_file = self.index_dir / "embeddings.npy"
        if emb_file.exists():
            return emb_file.stat().st_size / (1024 ** 2)
        return 0.0

    def avg_vectors_per_doc(self) -> float:
        lengths_file = self.index_dir / "doc_lengths.npy"
        if lengths_file.exists():
            lengths = np.load(lengths_file)
            return float(np.mean(lengths))
        return 0.0

    @staticmethod
    def _cuda_sync() -> None:
        """Call ``torch.cuda.synchronize()`` if a CUDA device is available."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def measure_latency(
        self,
        retrieve_fn: Callable[[str], Any],
        queries: List[str],
        warmup: int = 50,
        test: int = 200,
        encode_fn: Optional[Callable[[str], Any]] = None,
    ) -> Dict[str, float]:
        """Measure query latency, throughput, and optional encode-only latency.

        Parameters
        ----------
        retrieve_fn : callable
            ``retrieve_fn(query: str) -> results``
        queries : list[str]
            Pool of test queries (sampled with replacement if needed).
        warmup : int
            Queries to run before measuring.
        test : int
            Queries to time.
        encode_fn : callable, optional
            ``encode_fn(query: str) -> embedding`` — if provided, encoding
            latency is measured separately (without retrieval).

        Returns
        -------
        dict with ``avg_latency_ms``, ``p50_latency_ms``, ``p95_latency_ms``,
        ``p99_latency_ms``, ``throughput_qps``, and optionally
        ``query_encode_latency_ms``.
        """
        pool = queries if len(queries) >= warmup + test else queries * ((warmup + test) // len(queries) + 1)

        for q in pool[:warmup]:
            retrieve_fn(q)

        latencies: List[float] = []
        for q in pool[warmup : warmup + test]:
            self._cuda_sync()
            t0 = time.perf_counter()
            retrieve_fn(q)
            self._cuda_sync()
            latencies.append((time.perf_counter() - t0) * 1000)

        arr = np.array(latencies)
        total_sec = arr.sum() / 1000
        result = {
            "avg_latency_ms": float(np.mean(arr)),
            "p50_latency_ms": float(np.percentile(arr, 50)),
            "p95_latency_ms": float(np.percentile(arr, 95)),
            "p99_latency_ms": float(np.percentile(arr, 99)),
            "throughput_qps": test / total_sec if total_sec > 0 else 0.0,
        }

        if encode_fn is not None:
            for q in pool[:warmup]:
                encode_fn(q)
            enc_latencies: List[float] = []
            for q in pool[warmup : warmup + test]:
                self._cuda_sync()
                t0 = time.perf_counter()
                encode_fn(q)
                self._cuda_sync()
                enc_latencies.append((time.perf_counter() - t0) * 1000)
            enc_arr = np.array(enc_latencies)
            result["query_encode_latency_ms"] = float(np.mean(enc_arr))

        return result

    def full_report(
        self,
        retrieve_fn: Callable[[str], Any],
        queries: List[str],
        warmup: int = 50,
        test: int = 200,
    ) -> Dict[str, float]:
        report: Dict[str, float] = {
            "index_size_mb": self.index_size_mb(),
            "avg_vectors_per_doc": self.avg_vectors_per_doc(),
        }
        report.update(self.measure_latency(retrieve_fn, queries, warmup, test))
        return report


# ======================================================================
# 3. Robustness / Augmentation Testing
# ======================================================================

_RU_KEYBOARD_NEIGHBORS: Dict[str, str] = {
    "й": "цу", "ц": "йук", "у": "цке", "к": "уен", "е": "кнг",
    "н": "егш", "г": "ншщ", "ш": "гщз", "щ": "шзх", "з": "щхъ",
    "ф": "ыв", "ы": "фва", "в": "ыап", "а": "впр", "п": "аро",
    "р": "пол", "о": "рлд", "л": "одж", "д": "лжэ", "ж": "дэ",
    "я": "чс", "ч": "ясм", "с": "чми", "м": "сит", "и": "мть",
    "т": "иьб", "ь": "тбю", "б": "ью", "ю": "бь",
    "х": "зъ", "ъ": "хз", "э": "жд",
}

_RU_SYNONYMS: Dict[str, List[str]] = {
    "быстрый": ["скорый", "стремительный", "проворный"],
    "большой": ["крупный", "огромный", "значительный"],
    "маленький": ["небольшой", "мелкий", "крошечный"],
    "хороший": ["отличный", "прекрасный", "замечательный"],
    "плохой": ["скверный", "дурной", "негодный"],
    "делать": ["выполнять", "совершать", "осуществлять"],
    "говорить": ["сказать", "произнести", "утверждать"],
    "найти": ["обнаружить", "отыскать", "выявить"],
    "использовать": ["применять", "употреблять", "задействовать"],
    "получить": ["приобрести", "обрести", "достать"],
    "информация": ["сведения", "данные"],
    "документ": ["файл", "бумага", "акт"],
    "система": ["платформа", "комплекс"],
    "результат": ["итог", "исход", "вывод"],
    "вопрос": ["запрос", "обращение"],
    "ответ": ["отклик", "реакция"],
    "работа": ["труд", "деятельность"],
    "метод": ["способ", "подход", "приём"],
    "язык": ["наречие", "речь"],
    "модель": ["схема", "образец"],
}


class RobustnessAugmenter:
    """Generate augmented versions of Russian queries to test model robustness.

    Supports two augmentation types:
      * **Typo injection**: randomly swap characters with keyboard neighbours.
      * **Synonym substitution**: replace known words with synonyms.
    """

    def __init__(
        self,
        typo_rate: float = 0.15,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.typo_rate = typo_rate
        self.synonyms = synonym_dict or _RU_SYNONYMS

    def inject_typos(self, text: str) -> str:
        """Insert keyboard-neighbour typos into *text*."""
        chars = list(text)
        for i, ch in enumerate(chars):
            if random.random() < self.typo_rate and ch.lower() in _RU_KEYBOARD_NEIGHBORS:
                neighbors = _RU_KEYBOARD_NEIGHBORS[ch.lower()]
                replacement = random.choice(list(neighbors))
                chars[i] = replacement.upper() if ch.isupper() else replacement
        return "".join(chars)

    def substitute_synonyms(self, text: str) -> str:
        """Replace one word in *text* with a synonym (if available)."""
        words = text.split()
        replaceable = [(i, w) for i, w in enumerate(words) if w.lower() in self.synonyms]
        if not replaceable:
            return text
        idx, word = random.choice(replaceable)
        syn = random.choice(self.synonyms[word.lower()])
        words[idx] = syn
        return " ".join(words)

    def augment(self, text: str, n: int = 5) -> List[str]:
        """Generate *n* augmented variants of *text* using both methods."""
        variants: List[str] = []
        for _ in range(n):
            if random.random() < 0.5:
                variants.append(self.inject_typos(text))
            else:
                variants.append(self.substitute_synonyms(text))
        return variants


def evaluate_robustness(
    retrieve_fn: Callable[[str], List[Tuple[str, float]]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    eval_cfg: EvalConfig,
) -> Dict[str, Any]:
    """Compare IR metrics on original vs. augmented queries.

    Parameters
    ----------
    retrieve_fn : callable
        ``retrieve_fn(query: str) -> [(doc_id, score), …]``
    queries : {query_id: query_text}
    qrels : {query_id: {doc_id: relevance}}
    eval_cfg : EvalConfig

    Returns
    -------
    dict with ``original_metrics``, ``augmented_metrics``, ``degradation``.
    """
    augmenter = RobustnessAugmenter(typo_rate=eval_cfg.robustness_typo_rate)
    ir = IRMetrics(qrels, eval_cfg.k_values)

    # Original queries
    original_run: Dict[str, Dict[str, float]] = {}
    for qid, text in queries.items():
        results = retrieve_fn(text)
        original_run[qid] = {doc_id: score for doc_id, score in results}

    original_metrics = ir.evaluate(original_run)

    all_aug_runs: List[Dict[str, Dict[str, float]]] = []
    for _ in range(eval_cfg.robustness_num_augments):
        aug_run: Dict[str, Dict[str, float]] = {}
        for qid, text in queries.items():
            aug_text = augmenter.augment(text, n=1)[0]
            results = retrieve_fn(aug_text)
            aug_run[qid] = {doc_id: score for doc_id, score in results}
        all_aug_runs.append(aug_run)

    aug_metrics_list = [ir.evaluate(run) for run in all_aug_runs]

    avg_aug_metrics: Dict[str, float] = {}
    for key in aug_metrics_list[0]:
        avg_aug_metrics[key] = float(np.mean([m[key] for m in aug_metrics_list]))

    degradation: Dict[str, float] = {}
    for key in original_metrics:
        orig = original_metrics[key]
        aug = avg_aug_metrics.get(key, 0.0)
        degradation[key] = (orig - aug) / orig if orig > 0 else 0.0

    return {
        "original_metrics": original_metrics,
        "augmented_metrics": avg_aug_metrics,
        "degradation_pct": {k: round(v * 100, 2) for k, v in degradation.items()},
    }
