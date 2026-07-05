"""MS MARCO MTEB Retrieval evaluation task definition."""

from datasets import Dataset
from evaluation.config import MSMarcoConfig
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from typing import Any, cast


class MSMarcoRetrievalTask(AbsTaskRetrieval):
    metadata = TaskMetadata(
        dataset={
            "path": "local",
            "revision": "main",
        },  # Arbitrary, we are overriding dataset loading
        name="msmarco_cz_retrieval",
        description="MSMARCO Retrieval dataset translated to Czech",
        type="Retrieval",
        category="t2t",
        eval_langs=["ces-Latn"],
        main_score="mrr_at_10",
    )

    def __init__(
        self, dataset_loader: Dataset, dataset_config: MSMarcoConfig, **kwargs
    ):
        self.dataset_loader = dataset_loader
        self.dataset_config = dataset_config
        super().__init__(**kwargs)

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        split = "test"  # We are only testing models, so each split is a test
        queries: dict[str, str] = {}
        corpus: dict[str, dict[str, str]] = {}
        relevant_docs: dict[str, dict[str, int]] = {}

        for raw_record in self.dataset_loader:
            record = cast(dict[str, Any], raw_record)
            query_id = str(record["query_id"])
            query = str(record[self.dataset_config.query_field]).strip()
            passages = record["passages"]
            passage_texts = passages[self.dataset_config.passage_text_field]
            selected = passages["is_selected"]

            if not query:
                continue

            queries[query_id] = query
            relevant_docs[query_id] = {}

            for passage_index, passage_text in enumerate(passage_texts):
                document_id = f"{query_id}:{passage_index}"
                corpus[document_id] = {
                    "title": "",
                    "text": str(passage_text).strip(),
                }

                if selected[passage_index]:
                    relevant_docs[query_id][document_id] = 1

        self.queries = {split: queries}
        self.corpus = {split: corpus}
        self.relevant_docs = {split: relevant_docs}
        self.data_loaded = True
