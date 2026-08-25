"""Loader for the Czech Text Document Corpus and its synthetic query set."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CTDCRetrievalData:
    """Retrieval data built from the synthetic queries and the CTDC corpus."""

    corpus: dict[str, str]
    queries: dict[str, str]
    relevant_docs: dict[str, dict[str, int]]


@dataclass
class CzechTextDocumentDatasetLoader:
    """Build a retrieval dataset from CTDC `.txt` files and synthetic goldens.

    The corpus holds every document referenced by the selected queries plus
    `corpus_limit` further documents used as distractors.
    """

    synthetic_path: str | Path = Path("data/synthetic/ctdc_synthetic.json")
    corpus_dir: str | Path = Path("data/czech_text_document_corpus_v20")
    encoding: str = "utf-8"

    def load(
        self,
        query_limit: int | None = None,
        corpus_limit: int | None = None,
    ) -> CTDCRetrievalData:
        """Parse the synthetic goldens and the source documents."""
        queries, relevant_docs, gold_document_ids = self._load_synthetic(query_limit)
        corpus = self._load_corpus(gold_document_ids, corpus_limit)
        return CTDCRetrievalData(
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs,
        )

    def _load_synthetic(
        self, query_limit: int | None
    ) -> tuple[dict[str, str], dict[str, dict[str, int]], set[str]]:
        """Read synthetic goldens into queries and their relevance judgements."""
        synthetic_path = Path(self.synthetic_path)
        if not synthetic_path.exists():
            raise FileNotFoundError(f"Synthetic dataset not found: {synthetic_path}")

        goldens = json.loads(synthetic_path.read_text(encoding=self.encoding))

        queries: dict[str, str] = {}
        relevant_docs: dict[str, dict[str, int]] = {}
        gold_document_ids: set[str] = set()

        for golden_index, golden in enumerate(goldens):
            if query_limit is not None and len(queries) >= query_limit:
                break

            query = str(golden["input"]).strip()
            document_id = str(golden["source_file"]).strip()
            if not query or not document_id:
                continue

            query_id = f"q{golden_index}"
            queries[query_id] = query
            relevant_docs[query_id] = {document_id: 1}
            gold_document_ids.add(document_id)

        return queries, relevant_docs, gold_document_ids

    def _load_corpus(
        self,
        gold_document_ids: set[str],
        corpus_limit: int | None,
    ) -> dict[str, str]:
        """Read the gold documents plus `corpus_limit` extra distractor documents."""
        sources_dir = Path(self.corpus_dir) / "sources"
        if not sources_dir.is_dir():
            raise NotADirectoryError(f"Corpus directory not found: {sources_dir}")

        corpus: dict[str, str] = {}
        distractor_count = 0

        for path in sorted(sources_dir.glob("*.txt")):
            is_gold = path.stem in gold_document_ids
            reached_limit = (
                corpus_limit is not None and distractor_count >= corpus_limit
            )
            if not is_gold and reached_limit:
                continue

            text = self._clean_text(path.read_text(encoding=self.encoding))
            if not text:
                continue

            corpus[path.stem] = text
            if not is_gold:
                distractor_count += 1

        missing = gold_document_ids - corpus.keys()
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} relevant document(s) missing from {sources_dir}: "
                + ", ".join(sorted(missing)[:5])
            )

        return corpus

    def _clean_text(self, text: str) -> str:
        """Clean the text by stripping whitespace and normalizing spaces."""
        return " ".join(text.split())
