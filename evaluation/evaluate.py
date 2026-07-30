"""Evaluation script for the Czech Embedding Benchmark."""

import argparse
import os

from config import load_config
from dotenv import load_dotenv
from dataloaders.msmarco import MSMarcoDatasetLoader
from evaluation.tasks.msmarco_retrieval import MSMarcoRetrievalTask
from evaluation.utils.model_factory import build_model
from evaluation.utils.mteb_runner import (
    run_mteb_multilingual_retrieval,
    run_mteb_retrieval,
)
from mteb.benchmarks.benchmark import Benchmark

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_TOKEN", os.getenv("HF_HUB_TOKEN", ""))


def main(args):
    config = load_config(args.config)
    run = config.run

    # Build models
    models = [
        build_model(model_config, device=run.device) for model_config in config.models
    ]

    # MS Marco evaluation
    if config.msmarco:
        # Task definition: one task per (split, query/passage language pair)
        tasks = []
        for dataset_split in config.msmarco.splits:
            loader = MSMarcoDatasetLoader(
                dataset_dir=config.msmarco.input_path,
                split=dataset_split,
            )
            dataset = loader.load(limit=config.msmarco.limit)
            print(
                f"Loaded MS MARCO '{dataset_split}' split with {len(dataset)} records."
            )

            for query_lang, passage_lang in config.msmarco.language_pairs:
                tasks.append(
                    MSMarcoRetrievalTask(
                        dataset_loader=dataset,
                        dataset_config=config.msmarco,
                        dataset_split=dataset_split,
                        query_lang=query_lang,
                        passage_lang=passage_lang,
                    )
                )
        benchmark = Benchmark(name="MSMarco (cz/en)", tasks=tasks)

        run_mteb_retrieval(
            config=config,
            tasks=tasks,
            models=models,
            dataset_name="msmarco",
            benchmark=benchmark,
        )

    # CTDC Synthetic evaluation
    if config.ctdc_synthetic:
        ...

    # Official MTEB multilingual retrieval evaluation
    if config.multilingual_mteb:
        run_mteb_multilingual_retrieval(
            config=config,
            models=models,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file."
    )
    args = parser.parse_args()

    main(args)
