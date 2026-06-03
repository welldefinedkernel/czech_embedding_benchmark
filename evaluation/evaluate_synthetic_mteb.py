import json
import mteb
import os

from collections.abc import Iterable, Mapping, Sequence
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import EncodeKwargs
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
DEFAULT_OUTPUT_FOLDER = Path("results/synthetic_mteb")

class SyntheticRetrievalTask(AbsTaskRetrieval):
    """MTEB retrieval task backed by a local synthetic QA dataset."""

    metadata = TaskMetadata(
        name="CzechSyntheticRetrieval",
        description="Local synthetic Czech QA retrieval dataset generated from source documents.",
        dataset={"path": "local/synthetic", "revision": "local"},
        type="Retrieval",
        category="t2t",
        eval_splits=["train"],
        eval_langs=["ces-Latn"],
        main_score="ndcg_at_10",
        domains=["News", "Written"],
        task_subtypes=["Question Answering Retrieval"],
        license="not specified",
        annotations_creators="LM-generated",
        is_public=False,
    )

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        limit: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.limit = limit

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Load local synthetic rows into MTEB's v1 retrieval task format."""
        if self.data_loaded:
            return

        records = load_synthetic_records(self.dataset_path, limit=self.limit)
        corpus, queries, relevant_docs = build_retrieval_data(records)
        if not queries:
            raise ValueError(f"No usable synthetic records found in {self.dataset_path}")

        self.corpus = {self.split: corpus}
        self.queries = {self.split: queries}
        self.relevant_docs = {self.split: relevant_docs}
        self.data_loaded = True


def evaluate_synthetic_dataset(
    dataset_path: str | Path,
    model_name: str,
    output_folder: str | Path = DEFAULT_OUTPUT_FOLDER,
    split: str = "train",
    limit: int | None = None,
    device: str | None = None,
    use_safetensors: bool = False,
    overwrite_results: bool = True,
    encode_kwargs: EncodeKwargs | None = None,
) -> list[Any]:
    """Evaluate a Hugging Face embedding model on a local synthetic dataset."""
    task = SyntheticRetrievalTask(dataset_path=dataset_path, split=split, limit=limit)
    model = load_embedding_model(model_name, device=device, use_safetensors=use_safetensors)
    benchmark = mteb.MTEB(tasks=[task])
    return benchmark.run(
        model,
        output_folder=str(output_folder),
        eval_splits=[split],
        overwrite_results=overwrite_results,
        encode_kwargs=encode_kwargs,
    )


def load_embedding_model(model_name: str, device: str | None = None, use_safetensors: bool = False) -> SentenceTransformer:
    """Load a Hugging Face SentenceTransformer."""
    return SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"use_safetensors": use_safetensors},
    )


def load_synthetic_records(dataset_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load DeepEval synthetic records from JSONL."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {path}")

    if path.suffix == ".jsonl":
        records = _load_jsonl(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if limit is None:
        return records
    return records[:limit]


def build_retrieval_data(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """Convert synthetic QA rows into MTEB corpus, query and qrels mappings."""
    corpus: dict[str, dict[str, str]] = {}
    queries: dict[str, str] = {}
    relevant_docs: dict[str, dict[str, int]] = {}
    document_ids_by_text: dict[tuple[str, str], str] = {}

    for index, record in enumerate(records):
        query = str(record.get("input") or "").strip()
        contexts = _normalise_contexts(record.get("context"))
        if not query or not contexts:
            continue

        query_id = f"q{index}"
        queries[query_id] = query
        source_file = str(record.get("source_file") or "")
        relevant_docs[query_id] = {}

        for context in contexts:
            document_key = (context, source_file)
            document_id = document_ids_by_text.get(document_key)
            if document_id is None:
                document_id = f"d{len(document_ids_by_text)}"
                document_ids_by_text[document_key] = document_id
                corpus[document_id] = {"title": source_file, "text": context}
            relevant_docs[query_id][document_id] = 1

    return corpus, queries, relevant_docs


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} in {path}")
            records.append(row)
    return records

def _normalise_contexts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        contexts: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                contexts.append(text)
        return contexts
    text = str(value).strip()
    return [text] if text else []
