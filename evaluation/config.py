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
class MSMarcoConfig:  # MSMarco dataset settings
    input_path: Path
    splits: tuple[str, ...]
    language_pairs: tuple[tuple[str, str], ...]
    limit: int | None


@dataclass
class CTDCSyntheticConfig:  # CTDC Synthetic dataset settings
    input_path: Path
    query_field: str
    passage_text_field: str
    limit: int | None


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
    root = Path(__file__).resolve().parents[1]  # repository root path

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
            input_path=_resolve_path(msmarco["input_path"], root),
            splits=tuple(msmarco["splits"]),
            language_pairs=tuple(
                tuple(pair)
                for pair in msmarco.get("language_pairs", DEFAULT_MSMARCO_LANGUAGE_PAIRS)
            ),
            limit=msmarco.get("limit"),
        )
        if msmarco["enabled"]
        else None,
        ctdc_synthetic=CTDCSyntheticConfig(
            input_path=_resolve_path(ctdc_synthetic["input_path"], root),
            query_field=ctdc_synthetic["query_field"],
            passage_text_field=ctdc_synthetic["passage_text_field"],
            limit=ctdc_synthetic.get("limit"),
        )
        if ctdc_synthetic["enabled"]
        else None,
        multilingual_mteb=MultilingualMTEBConfig(
            enabled=multilingual_mteb["enabled"],
            include_tasks=tuple(multilingual_mteb["include_tasks"]),
        )
        if multilingual_mteb["enabled"]
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
