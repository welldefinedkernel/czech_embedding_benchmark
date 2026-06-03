from data.create_synthetic import create_synthetic_dataset
from models.azure_llm import AzureLLM


if __name__ == "__main__":
    create_synthetic_dataset(
        dataset_name="czech_text_document_synthetic_100",
        llm_model=AzureLLM,
        document_limit=50,
        max_goldens_per_context=2,
        include_expected_output=True,
        file_type="jsonl",
        async_mode=True,
        quiet=False,
    )
