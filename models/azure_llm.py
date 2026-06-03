import os
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

load_dotenv()

class AzureLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.deployment_name = os.getenv("AZURE_LLM_NAME", "") 
        self.endpoint = os.getenv("AZURE_LLM_ENDPOINT")
        self.api_key = SecretStr(os.getenv("AZURE_LLM_KEY", ""))
        self.api_version = os.getenv("AZURE_API_VERSION", "2024-06-01")
        
        self.model = self.load_model()
        super().__init__(model_name=self.deployment_name)

    def load_model(self):
        return AzureChatOpenAI(
            azure_deployment=self.deployment_name,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        if "schema" in kwargs:
            raise TypeError("schema not supported, trigger DeepEval's JSON fallback")
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"[AzureLLM generate ERROR]: {e}")
            raise e

    async def a_generate(self, prompt: str, **kwargs) -> str:
        if "schema" in kwargs:
            raise TypeError("schema not supported, trigger DeepEval's JSON fallback")
        try:
            response = await self.model.ainvoke(prompt)
            return response.content
        except Exception as e:
            print(f"[AzureLLM a_generate ERROR]: {e}")
            raise e

    def get_model_name(self) -> str:
        return self.deployment_name
