"""Config loader for retrieval evaluation."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig:  # Evaluation run settings
    output_dir: Path
    seed: int
    batch_size: int
    num_proc: int
    device: str
    resume_from_partial: bool
    write_predictions: bool
    show_progress_bar: bool


# Query language / passage language combinations evaluated by default.
DEFAULT_MSMARCO_LANGUAGE_PAIRS: tuple[tuple[str, str], ...] = (
    ("cz", "cz"),
    ("cz", "en"),
    ("en", "cz"),
    ("en", "en"),
)


@dataclass
class RerankingConfig:  # Optional cross-encoder second stage over a task's own run
    model: str
    top_k: int = 100


@dataclass
class MSMarcoConfig:  # MSMarco dataset settings
    enabled: bool
    input_path: Path
    splits: tuple[str, ...]
    language_pairs: tuple[tuple[str, str], ...]
    limit: int | None
    reranking: RerankingConfig | None = None


@dataclass
class CTDCSyntheticConfig:  # CTDC Synthetic dataset settings
    enabled: bool
    synthetic_path: Path
    corpus_dir: Path
    query_limit: int | None
    corpus_limit: int | None
    reranking: RerankingConfig | None = None


@dataclass
class ModelConfig:  # Model settings for evaluation
    name: str
    normalize_embeddings: bool
    use_safetensors: bool
    trust_remote_code: bool
    query_prompt: str | None = None
    document_prompt: str | None = None
    device_map: str | None = None
    model_dtype: str | None = None
    max_seq_length: int | None = None


@dataclass
class MultilingualMTEBConfig:  # Official MTEB multilingual benchmark settings
    enabled: bool
    include_tasks: tuple[str, ...]


@dataclass
class EvaluationConfig:  # Overall evaluation configuration
    run: RunConfig
    msmarco: MSMarcoConfig | None
    ctdc_synthetic: CTDCSyntheticConfig | None
    multilingual_mteb: MultilingualMTEBConfig | None
    models: tuple[ModelConfig, ...]
    config_path: Path
    repo_root: Path


def load_config(config_path: str | Path) -> EvaluationConfig:
    path = Path(config_path).expanduser().resolve()  # config file path
    root = Path(__file__).resolve().parents[2]  # repository root path

    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    run = raw["run"]
    datasets = raw["datasets"]
    msmarco = datasets["msmarco"]
    ctdc_synthetic = datasets["ctdc_synthetic"]
    multilingual_mteb = raw.get("mteb", {}).get("multilingual", {})
    models = raw["models"]

    config = EvaluationConfig(
        run=RunConfig(
            output_dir=_resolve_path(run["output_dir"], root),
            seed=run.get("seed", 42),
            batch_size=run.get("batch_size", 32),
            num_proc=run.get("num_proc", 1),
            device=run.get("device", "cuda"),
            resume_from_partial=run.get("resume_from_partial", False),
            write_predictions=run.get("write_predictions", True),
            show_progress_bar=run.get("show_progress_bar", True),
        ),
        msmarco=MSMarcoConfig(
            enabled=msmarco["enabled"],
            input_path=_resolve_path(msmarco["input_path"], root),
            splits=tuple(msmarco["splits"]),
            language_pairs=tuple(
                tuple(pair)
                for pair in msmarco.get(
                    "language_pairs", DEFAULT_MSMARCO_LANGUAGE_PAIRS
                )
            ),
            limit=msmarco.get("limit"),
            reranking=_load_reranking(msmarco),
        )
        if _task_active(msmarco)
        else None,
        ctdc_synthetic=CTDCSyntheticConfig(
            enabled=ctdc_synthetic["enabled"],
            synthetic_path=_resolve_path(ctdc_synthetic["synthetic_path"], root),
            corpus_dir=_resolve_path(ctdc_synthetic["corpus_dir"], root),
            query_limit=ctdc_synthetic.get("query_limit"),
            corpus_limit=ctdc_synthetic.get("corpus_limit"),
            reranking=_load_reranking(ctdc_synthetic),
        )
        if _task_active(ctdc_synthetic)
        else None,
        multilingual_mteb=MultilingualMTEBConfig(
            enabled=True,
            include_tasks=tuple(multilingual_mteb["include_tasks"]),
        )
        if multilingual_mteb.get("enabled", False)
        else None,
        models=tuple(ModelConfig(**model) for model in models),
        config_path=path,
        repo_root=root,
    )

    return config


def _resolve_path(path: str, repo_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _task_active(section: dict) -> bool:
    """A task is loaded if it retrieves, reranks, or both."""
    return bool(section["enabled"]) or bool(section.get("reranking", False))


def _load_reranking(section: dict) -> RerankingConfig | None:
    if not section.get("reranking", False):
        return None

    model = section.get("reranking_model")
    if not model:
        raise ValueError(
            "`reranking = true` requires `reranking_model` to be set to a "
            "HuggingFace cross-encoder id."
        )
    return RerankingConfig(model=str(model), top_k=int(section.get("reranking_top_k", 100)))
