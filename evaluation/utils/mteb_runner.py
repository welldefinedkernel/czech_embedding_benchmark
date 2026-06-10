"""MTEB retrieval runner."""

import json
import os
from typing import Any, Sequence, cast

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import mteb

from evaluation.config import EvaluationConfig
from mteb.results import ModelResult


def run_mteb_retrieval(
    config: EvaluationConfig,
    tasks: Sequence[Any],
    models: Sequence[Any],
    dataset_name: str,
) -> dict[str, ModelResult]:
    """Run configured models on given tasks and return MTEB results by model."""
    results: dict[str, ModelResult] = {}

    for model_config, model in zip(config.models, models):
        output_folder = (
            config.run.output_dir / dataset_name / model_config.name.replace("/", "__")
        )
        output_folder.mkdir(parents=True, exist_ok=True)

        result = mteb.evaluate(
            model,
            tasks=list(tasks),
            encode_kwargs=cast(
                Any,
                {
                    "normalize_embeddings": model_config.normalize_embeddings,
                    "batch_size": config.run.batch_size,
                },
            ),
            overwrite_strategy="only-missing",
            prediction_folder=str(output_folder),
            raise_error=False,
            show_progress_bar=True,
            num_proc=config.run.num_proc,
        )
        (output_folder / "metrics.json").write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        _indent_prediction_files(output_folder)
        results[model_config.name] = result

    return results


def run_mteb_multilingual_retrieval(
    config: EvaluationConfig,
    models: Sequence[Any],
) -> dict[str, ModelResult]:
    """Run official MTEB multilingual retrieval tasks."""
    if config.multilingual_mteb is None:
        return {}

    benchmark = mteb.get_benchmark("MTEB(Multilingual, v2)")
    excluded_tasks = set(config.multilingual_mteb.exclude_tasks)
    tasks = [
        task
        for task in benchmark.tasks
        if task.metadata.type == "Retrieval"
        and task.metadata.name not in excluded_tasks
    ]
    return run_mteb_retrieval(
        config=config,
        tasks=tasks,
        models=models,
        dataset_name="mteb_multilingual_retrieval",
    )


def _indent_prediction_files(output_folder: Any) -> None:
    for prediction_file in output_folder.glob("*_predictions.json"):
        predictions = json.loads(prediction_file.read_text(encoding="utf-8"))
        prediction_file.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
