from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    encoder_name: str = "xlm-roberta-base"
    embedding_dim: int = 128
    query_max_len: int = 32
    doc_max_len: int = 180
    query_token: str = "[unused0]"
    doc_token: str = "[unused1]"
    normalize_embeddings: bool = True
    compression_strategy: str = "linear"
    pruning_threshold: float = 0.0
    mask_punctuation: bool = True


@dataclass
class DataConfig:
    mmarco_dataset: str = "unicamp-dl/mmarco"
    mmarco_lang: str = "russian"
    miracl_dataset: str = "miracl/miracl"
    miracl_lang: str = "ru"
    train_split: str = "train"
    val_split: str = "dev"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    negative_strategy: str = "mixed"  # "random" | "in_batch" | "hard" | "mixed"
    num_hard_negatives: int = 7
    hard_negative_source: str = "bm25"
    mixed_lang_ratio: float = 0.3  # fraction of English examples in a batch
    cache_dir: str = "./cache"
    mmarco_use_hub_files: bool = True
    mmarco_revision: Optional[str] = None
    mmarco_max_triples: int = 100_000
    miracl_use_hub_files: bool = True
    miracl_revision: Optional[str] = "refs/convert/parquet"


@dataclass
class TrainConfig:
    output_dir: str = "./checkpoints"
    seed: int = 42
    per_device_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs_phase1: int = 5
    num_epochs_phase2: int = 3
    phase1_max_len: int = 128
    phase2_max_len: int = 256
    temperature: float = 0.05
    fp16: bool = True
    log_every_n_steps: int = 50
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    max_grad_norm: float = 1.0
    num_workers: int = 4


@dataclass
class IndexConfig:
    index_dir: str = "./index"
    corpus_path: Optional[str] = None
    batch_size: int = 256
    num_partitions: int = 1
    use_gpu: bool = True
    save_fp16: bool = True


@dataclass
class EvalConfig:
    metrics: List[str] = field(default_factory=lambda: ["map", "mrr", "ndcg", "recall"])
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 100])
    robustness_typo_rate: float = 0.15
    robustness_num_augments: int = 5
    latency_warmup_queries: int = 50
    latency_test_queries: int = 200


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ColBERT-RU: Multilingual Late-Interaction Retrieval for Russian",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Model")
    g.add_argument("--encoder_name", type=str, default="xlm-roberta-base")
    g.add_argument("--embedding_dim", type=int, default=128)
    g.add_argument("--query_max_len", type=int, default=32)
    g.add_argument("--doc_max_len", type=int, default=180)
    g.add_argument("--compression_strategy", choices=["linear", "token_pruning", "none"], default="linear")
    g.add_argument("--pruning_threshold", type=float, default=0.0)

    g = p.add_argument_group("Data")
    g.add_argument("--negative_strategy", choices=["random", "in_batch", "hard", "mixed"], default="mixed")
    g.add_argument("--num_hard_negatives", type=int, default=7)
    g.add_argument("--mixed_lang_ratio", type=float, default=0.3)
    g.add_argument("--max_train_samples", type=int, default=None)
    g.add_argument("--max_val_samples", type=int, default=None)
    g.add_argument("--cache_dir", type=str, default="./cache")
    g.add_argument(
        "--mmarco_use_hub_files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load mMARCO via Hub TSV files (recommended for datasets>=3).",
    )
    g.add_argument("--mmarco_revision", type=str, default=None)
    g.add_argument(
        "--mmarco_max_triples",
        type=int,
        default=100_000,
        help="Cap mMARCO train triples for RAM (0 = load all; needs ~30+ GB RAM).",
    )
    g.add_argument(
        "--miracl_use_hub_files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load MIRACL via Hub TSV + streaming corpus (required for ru + datasets>=3).",
    )
    g.add_argument(
        "--miracl_revision",
        type=str,
        default="refs/convert/parquet",
        help="Revision for load_dataset fallback when parquet exists for this language.",
    )

    g = p.add_argument_group("Training")
    g.add_argument("--output_dir", type=str, default="./checkpoints")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--per_device_batch_size", type=int, default=32)
    g.add_argument("--gradient_accumulation_steps", type=int, default=1)
    g.add_argument("--learning_rate", type=float, default=3e-6)
    g.add_argument("--num_epochs_phase1", type=int, default=5)
    g.add_argument("--num_epochs_phase2", type=int, default=3)
    g.add_argument("--temperature", type=float, default=0.05)
    g.add_argument("--fp16", action="store_true", default=True)
    g.add_argument("--log_every_n_steps", type=int, default=50)
    g.add_argument("--eval_every_n_steps", type=int, default=500)

    g = p.add_argument_group("Index")
    g.add_argument("--index_dir", type=str, default="./index")
    g.add_argument("--corpus_path", type=str, default=None)
    g.add_argument("--index_batch_size", type=int, default=256)

    g = p.add_argument_group("Evaluation")
    g.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    g.add_argument("--robustness_typo_rate", type=float, default=0.15)

    return p


def load_config_from_args(args: Optional[argparse.Namespace] = None) -> PipelineConfig:
    """Parse CLI args and populate a ``PipelineConfig``."""
    if args is None:
        parser = build_parser()
        args = parser.parse_args()

    model_cfg = ModelConfig(
        encoder_name=args.encoder_name,
        embedding_dim=args.embedding_dim,
        query_max_len=args.query_max_len,
        doc_max_len=args.doc_max_len,
        compression_strategy=args.compression_strategy,
        pruning_threshold=args.pruning_threshold,
    )
    data_cfg = DataConfig(
        negative_strategy=args.negative_strategy,
        num_hard_negatives=args.num_hard_negatives,
        mixed_lang_ratio=args.mixed_lang_ratio,
        max_train_samples=args.max_train_samples,
        max_val_samples=getattr(args, "max_val_samples", None),
        cache_dir=args.cache_dir,
        mmarco_use_hub_files=args.mmarco_use_hub_files,
        mmarco_revision=args.mmarco_revision,
        mmarco_max_triples=args.mmarco_max_triples,
        miracl_use_hub_files=args.miracl_use_hub_files,
        miracl_revision=args.miracl_revision,
    )
    train_cfg = TrainConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs_phase1=args.num_epochs_phase1,
        num_epochs_phase2=args.num_epochs_phase2,
        temperature=args.temperature,
        fp16=args.fp16,
        log_every_n_steps=args.log_every_n_steps,
        eval_every_n_steps=args.eval_every_n_steps,
    )
    index_cfg = IndexConfig(
        index_dir=args.index_dir,
        corpus_path=args.corpus_path,
        batch_size=args.index_batch_size,
    )
    eval_cfg = EvalConfig(
        k_values=args.k_values,
        robustness_typo_rate=args.robustness_typo_rate,
    )

    return PipelineConfig(
        model=model_cfg,
        data=data_cfg,
        train=train_cfg,
        index=index_cfg,
        eval=eval_cfg,
    )
