import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from evaluation.evaluate_synthetic_mteb import evaluate_synthetic_dataset
from pathlib import Path

DATASET_PATH = Path("data/synthetic/czech_text_document_synthetic_100_train.jsonl")
OUTPUT_FOLDER = Path("results/synthetic_mteb")
MODELS = [
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-small",
]

if __name__ == "__main__":
    for model_name in MODELS:
        print(
            evaluate_synthetic_dataset(
                dataset_path=DATASET_PATH,
                model_name=model_name,
                output_folder=OUTPUT_FOLDER,
                split="train",
                overwrite_results=True,
            )
        )
