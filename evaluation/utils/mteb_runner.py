"""MTEB retrieval runner."""

import json
import mteb

from evaluation.config import EvaluationConfig
from mteb.benchmarks.benchmark import Benchmark
from mteb.results import BenchmarkResults, ModelResult
from pathlib import Path
from typing import Any, Sequence, cast


def run_mteb_retrieval(
    config: EvaluationConfig,
    tasks: Sequence[Any],
    models: Sequence[Any],
    dataset_name: str,
    benchmark: Benchmark | None = None,
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
            overwrite_strategy="always",
            prediction_folder=str(output_folder),
            raise_error=True,
            show_progress_bar=True,
            num_proc=config.run.num_proc,
        )
        (output_folder / "metrics.json").write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if benchmark is not None:
            _write_benchmark_scores(benchmark, result, output_folder)
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
    included_tasks = set(config.multilingual_mteb.include_tasks)
    tasks = _select_retrieval_tasks(benchmark, included_tasks)
    included_benchmark = Benchmark(
        name=benchmark.name,
        tasks=tasks,
        description=benchmark.description,
        citation=benchmark.citation,
    )
    return run_mteb_retrieval(
        config=config,
        tasks=tasks,
        models=models,
        dataset_name="mteb_multilingual_retrieval",
        benchmark=included_benchmark,
    )


def _select_retrieval_tasks(
    benchmark: Benchmark,
    included_tasks: set[str],
) -> list[Any]:
    available_tasks = {
        task.metadata.name
        for task in benchmark.tasks
        if task.metadata.type == "Retrieval"
    }
    missing_tasks = included_tasks - available_tasks
    if missing_tasks:
        raise ValueError(
            "Included tasks are not MTEB multilingual retrieval tasks: "
            + ", ".join(sorted(missing_tasks))
        )

    return [
        task
        for task in benchmark.tasks
        if task.metadata.type == "Retrieval"
        and (task.metadata.name in included_tasks if included_tasks else True)
    ]


def _write_benchmark_scores(
    benchmark: Benchmark,
    result: ModelResult,
    output_folder: Path,
) -> None:
    benchmark_results = BenchmarkResults(model_results=[result], benchmark=benchmark)
    benchmark_scores = benchmark.get_score(benchmark_results)
    (output_folder / "benchmark_scores.json").write_text(
        json.dumps(benchmark_scores, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _indent_prediction_files(output_folder: Path) -> None:
    for prediction_file in output_folder.glob("*_predictions.json"):
        predictions = json.loads(prediction_file.read_text(encoding="utf-8"))
        prediction_file.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
