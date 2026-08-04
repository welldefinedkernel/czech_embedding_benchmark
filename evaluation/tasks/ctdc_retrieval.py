"""CTDC synthetic MTEB Retrieval evaluation task definition."""

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

from dataloaders.czech_text_document import CTDCRetrievalData


class CTDCSyntheticRetrievalTask(AbsTaskRetrieval):
    # Keep the document-side fallback instruction empty for models that resolve prompts by task name.
    abstask_prompt = ""

    def __init__(self, dataset_loader: CTDCRetrievalData, **kwargs):
        self.dataset_loader = dataset_loader

        self.metadata = TaskMetadata(
            dataset={
                "path": "local",
                "revision": "main",
            },  # Arbitrary, we are overriding dataset loading.
            name="ctdc_synthetic_retrieval",
            description=(
                "Retrieval over the Czech Text Document Corpus with synthetic "
                "Czech questions generated from its news articles"
            ),
            type="Retrieval",
            category="t2t",
            eval_langs=["ces-Latn"],
            main_score="mrr_at_10",
            prompt={
                "query": "Given a question in Czech, retrieve the news article that answers it"
            },
        )
        super().__init__(**kwargs)

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        split = "test"  # We are only testing models, so each split is a test
        corpus = {
            document_id: {"title": "", "text": text}
            for document_id, text in self.dataset_loader.corpus.items()
        }

        self.queries = {split: dict(self.dataset_loader.queries)}
        self.corpus = {split: corpus}
        self.relevant_docs = {split: dict(self.dataset_loader.relevant_docs)}
        self.data_loaded = True
