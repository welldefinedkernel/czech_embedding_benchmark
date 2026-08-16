"""Evaluation script for the Czech Embedding Benchmark."""

import argparse
import os

from dotenv import load_dotenv
from mteb.benchmarks.benchmark import Benchmark

from dataloaders.czech_text_document import CzechTextDocumentDatasetLoader
from dataloaders.msmarco import MSMarcoDatasetLoader
from evaluation.tasks.ctdc_retrieval import CTDCSyntheticRetrievalTask
from evaluation.tasks.msmarco_retrieval import MSMarcoRetrievalTask
from evaluation.utils.config import load_config
from evaluation.utils.model_factory import build_model
from evaluation.utils.mteb_runner import (
    run_mteb_multilingual_retrieval,
    run_mteb_reranking,
    run_mteb_retrieval,
)

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_TOKEN", os.getenv("HF_HUB_TOKEN", ""))


def main(args):
    config = load_config(args.config)
    run = config.run

    # A rerank-only run never encodes with these, so keep them off the GPU.
    retrieves = (
        (config.msmarco is not None and config.msmarco.enabled)
        or (config.ctdc_synthetic is not None and config.ctdc_synthetic.enabled)
        or config.multilingual_mteb is not None
    )
    models = (
        [build_model(model_config, device=run.device) for model_config in config.models]
        if retrieves
        else []
    )

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

        if config.msmarco.enabled:
            run_mteb_retrieval(
                config=config,
                tasks=tasks,
                models=models,
                dataset_name="msmarco",
                benchmark=benchmark,
            )

        if config.msmarco.reranking:
            run_mteb_reranking(
                config=config,
                tasks=tasks,
                dataset_name="msmarco",
                reranking=config.msmarco.reranking,
            )

    # CTDC Synthetic evaluation
    if config.ctdc_synthetic:
        loader = CzechTextDocumentDatasetLoader(
            synthetic_path=config.ctdc_synthetic.synthetic_path,
            corpus_dir=config.ctdc_synthetic.corpus_dir,
        )
        data = loader.load(
            query_limit=config.ctdc_synthetic.query_limit,
            corpus_limit=config.ctdc_synthetic.corpus_limit,
        )
        print(
            f"Loaded CTDC synthetic set with {len(data.queries)} queries "
            f"and {len(data.corpus)} corpus documents."
        )

        tasks = [CTDCSyntheticRetrievalTask(dataset_loader=data)]
        benchmark = Benchmark(name="CTDC Synthetic (cz)", tasks=tasks)

        if config.ctdc_synthetic.enabled:
            run_mteb_retrieval(
                config=config,
                tasks=tasks,
                models=models,
                dataset_name="ctdc_synthetic",
                benchmark=benchmark,
            )

        if config.ctdc_synthetic.reranking:
            run_mteb_reranking(
                config=config,
                tasks=tasks,
                dataset_name="ctdc_synthetic",
                reranking=config.ctdc_synthetic.reranking,
            )

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
