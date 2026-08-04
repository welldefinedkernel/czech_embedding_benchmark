"""MS MARCO MTEB Retrieval evaluation task definition."""

from typing import Any, cast

from datasets import Dataset
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

from evaluation.utils.config import MSMarcoConfig

# Language codes supported by the translated MS MARCO dataset, mapped to the
# dataset field holding their text and their MTEB eval_langs code.
_LANGUAGE_FIELDS: dict[str, dict[str, str]] = {
    "en": {
        "query_field": "query",
        "passage_field": "passage_text",
        "eval_lang": "eng-Latn",
    },
    "cz": {
        "query_field": "query_cz",
        "passage_field": "passage_text_cz",
        "eval_lang": "ces-Latn",
    },
}


class MSMarcoRetrievalTask(AbsTaskRetrieval):
    # Fallback instruction used when a model resolves prompts via a registry
    # lookup by task name (e.g. F2LLM, whose registered "document" prompt is an
    # empty string). Its model card requires no instruction on documents, so we
    # override the AbsTaskRetrieval default ("Retrieve text based on user
    # query.") with an empty string to avoid prepending anything to passages.
    abstask_prompt = ""

    def __init__(
        self,
        dataset_loader: Dataset,
        dataset_config: MSMarcoConfig,
        dataset_split: str,
        query_lang: str,
        passage_lang: str,
        **kwargs,
    ):
        self.dataset_loader = dataset_loader
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.query_lang = query_lang
        self.passage_lang = passage_lang

        eval_langs = sorted(
            {
                _LANGUAGE_FIELDS[query_lang]["eval_lang"],
                _LANGUAGE_FIELDS[passage_lang]["eval_lang"],
            }
        )
        self.metadata = TaskMetadata(
            dataset={
                "path": "local",
                "revision": "main",
            },  # Arbitrary, we are overriding dataset loading.
            name=f"msmarco_{dataset_split}_{query_lang}-{passage_lang}_retrieval",
            description=(
                f"MSMARCO Retrieval dataset ({dataset_split} split) with "
                f"{query_lang} queries and {passage_lang} passages"
            ),
            type="Retrieval",
            category="t2t",
            eval_langs=eval_langs,
            main_score="mrr_at_10",
            # Same instruction as the official MSMARCO task. Instruction-tuned
            # models (e.g. Harrier) read this, without it mteb falls back to a
            # registry lookup by task name, which fails for custom tasks.
            prompt={
                "query": "Given a web search query, retrieve relevant passages that answer the query"
            },
        )
        super().__init__(**kwargs)

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        query_field = _LANGUAGE_FIELDS[self.query_lang]["query_field"]
        passage_field = _LANGUAGE_FIELDS[self.passage_lang]["passage_field"]

        split = "test"  # We are only testing models, so each split is a test
        queries: dict[str, str] = {}
        corpus: dict[str, dict[str, str]] = {}
        relevant_docs: dict[str, dict[str, int]] = {}

        for raw_record in self.dataset_loader:
            record = cast(dict[str, Any], raw_record)
            query_id = str(record["query_id"])
            query = str(record[query_field]).strip()
            passages = record["passages"]
            passage_texts = passages[passage_field]
            selected = passages["is_selected"]

            # An empty query (e.g. missing translation) excludes the query from
            # evaluation, but its passages stay in the corpus so that all
            # language-pair tasks share an identical corpus.
            has_query = bool(query)
            if has_query:
                queries[query_id] = query
                relevant_docs[query_id] = {}

            for passage_index, passage_text in enumerate(passage_texts):
                document_id = f"{query_id}:{passage_index}"
                corpus[document_id] = {
                    "title": "",
                    "text": str(passage_text).strip(),
                }

                if has_query and selected[passage_index]:
                    relevant_docs[query_id][document_id] = 1

        self.queries = {split: queries}
        self.corpus = {split: corpus}
        self.relevant_docs = {split: relevant_docs}
        self.data_loaded = True
