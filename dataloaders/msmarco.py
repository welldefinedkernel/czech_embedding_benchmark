import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset


@dataclass
class MSMarcoDatasetLoader:
    """Parse MS MARCO Czech JSONL files into a Hugging Face dataset."""
    
    dataset_dir: str | Path = Path("data/ms_marco")
    dataset: Dataset | None = field(default=None, init=False, repr=False)
    encoding: str = "utf-8"
    split: str = "train"

    def load(self, limit: int | None = None) -> Dataset:
        """Parse source records file and store it as a Hugging Face dataset."""
        records = self._parse_json_file(limit=limit)
        self.dataset = Dataset.from_list(records)
        return self.dataset

    def _parse_json_file(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Parse source JSONL file into a list of records."""
        dataset_path = Path(self.dataset_dir) / f"{self.split}.jsonl"
        records: list[dict[str, Any]] = []

        with dataset_path.open(encoding=self.encoding) as f:
            for line_index, line in enumerate(f):
                if limit is not None and line_index >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                records.append(json.loads(line))
                
        return records

if __name__ == "__main__":
    loader = MSMarcoDatasetLoader(split="validation")
    dataset = loader.load(limit=10)
    print(dataset)
    print(dataset[0])
