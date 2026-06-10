"""Evaluation script for the Czech Embedding Benchmark."""

import argparse

from config import load_config
from dataloaders.msmarco import MSMarcoDatasetLoader
from evaluation.tasks.msmarco_retrieval import MSMarcoRetrievalTask
from evaluation.utils.model_factory import build_model
from evaluation.utils.mteb_runner import (
    run_mteb_multilingual_retrieval,
    run_mteb_retrieval,
)


def main(args):
    config = load_config(args.config)

    # Build models
    models = [build_model(model_config) for model_config in config.models]

    # MS Marco evaluation
    if config.msmarco:
        # Dataset loading
        loader = MSMarcoDatasetLoader(
            dataset_dir=config.msmarco.input_path,
            split=config.msmarco.split,
        )
        dataset = loader.load(limit=config.msmarco.limit)
        print(f"Loaded MS MARCO dataset with {len(dataset)} records.")

        # Task definition
        task = MSMarcoRetrievalTask(
            dataset_loader=dataset,
            dataset_config=config.msmarco,
        )

        run_mteb_retrieval(
            config=config,
            tasks=[task],
            models=models,
            dataset_name="msmarco",
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
