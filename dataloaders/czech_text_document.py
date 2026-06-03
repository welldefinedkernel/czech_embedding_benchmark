"""Loader for the Czech Text Document Corpus `.txt` files."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CzechTextDocumentDatasetLoader:
    """Parse Czech Text Document Corpus text files into Hugging Face datasets."""

    dataset_dir: str | Path = Path("data/czech_text_document_corpus_v20")
    split: str = "train"
    encoding: str = "utf-8"
    dataset: Any | None = field(default=None, init=False, repr=False)

    def load(self, limit: int | None = None) -> Any:
        """Parse source files and store them as a Hugging Face dataset."""
        from datasets import Dataset
        records = self._parse_txt_files(limit=limit)
        self.dataset = Dataset.from_list(records)
        return self.dataset
    
    def _parse_txt_files(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Parse source `.txt` files into a list of records."""
        records: list[dict[str, Any]] = []

        for file_index, path in enumerate(self._iter_txt_files()):
            if limit is not None and file_index >= limit:
                break

            raw_text = path.read_text(encoding=self.encoding)
            text = self._clean_text(raw_text)
            if not text:
                continue

            records.append(
                {
                    "document_id": path.stem,
                    "text": text,
                }
            )

        return records

    def _iter_txt_files(self) -> Sequence[Path]:
        """Yield paths to all `.txt` files in the dataset directory."""
        dataset_path = Path(self.dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")
        return sorted(dataset_path.glob("*.txt"))

    def _clean_text(self, text: str) -> str:
        """Clean the text by stripping whitespace and normalizing spaces."""
        text = text.strip()
        text = " ".join(text.split())
        return text
    
if __name__ == "__main__":
    loader = CzechTextDocumentDatasetLoader()
    dataset = loader.load(limit=10)
    print(dataset)
    print(dataset[0])