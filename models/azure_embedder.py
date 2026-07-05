import os

from deepeval.models import DeepEvalBaseEmbeddingModel
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()


class AzureEmbedder(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        model_name = os.getenv("AZURE_EMBEDDING_NAME", "")
        self.endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT")
        self.api_key = SecretStr(os.getenv("AZURE_EMBEDDING_KEY", ""))
        self.api_version = os.getenv("AZURE_API_VERSION", "2024-06-01")
        super().__init__(model=model_name)

    def load_model(self):
        return AzureOpenAIEmbeddings(
            azure_deployment=self.name,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def embed_text(self, text: str, **kwargs) -> list[float]:
        try:
            return self.model.embed_query(text)
        except Exception as e:
            print(f"[AzureEmbedder embed_text ERROR]: {e}")
            raise e

    async def a_embed_text(self, text: str, **kwargs) -> list[float]:
        try:
            return await self.model.aembed_query(text)
        except Exception as e:
            print(f"[AzureEmbedder a_embed_text ERROR]: {e}")
            raise e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.model.aembed_documents(texts)

    def get_model_name(self) -> str:
        return self.name
