"""Config loader for retrieval evaluation."""

import sys
import tomllib

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig:  # Evaluation run settings
    output_dir: Path
    seed: int
    batch_size: int
    num_proc: int


@dataclass
class MSMarcoConfig:  # MSMarco dataset settings
    input_path: Path
    split: str
    query_field: str
    passage_text_field: str
    limit: int


@dataclass
class CTDCSyntheticConfig:  # CTDC Synthetic dataset settings
    input_path: Path
    query_field: str
    passage_text_field: str
    limit: int


@dataclass
class ModelConfig:  # Model settings for evaluation
    name: str
    query_instruction: str
    document_instruction: str
    normalize_embeddings: bool
    use_safetensors: bool


@dataclass
class MultilingualMTEBConfig:  # Official MTEB multilingual benchmark settings
    enabled: bool
    exclude_tasks: tuple[str, ...]


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
        ),
        msmarco=MSMarcoConfig(
            input_path=_resolve_path(msmarco["input_path"], root),
            split=msmarco["split"],
            query_field=msmarco["query_field"],
            passage_text_field=msmarco["passage_text_field"],
            limit=msmarco["limit"],
        )
        if msmarco["enabled"]
        else None,
        ctdc_synthetic=CTDCSyntheticConfig(
            input_path=_resolve_path(ctdc_synthetic["input_path"], root),
            query_field=ctdc_synthetic["query_field"],
            passage_text_field=ctdc_synthetic["passage_text_field"],
            limit=ctdc_synthetic["limit"],
        )
        if ctdc_synthetic["enabled"]
        else None,
        multilingual_mteb=MultilingualMTEBConfig(
            enabled=multilingual_mteb["enabled"],
            exclude_tasks=tuple(multilingual_mteb["exclude_tasks"]),
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


if __name__ == "__main__":
    config_file = sys.argv[1]
    load_config(config_file)
