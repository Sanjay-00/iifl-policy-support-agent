import os

from google import genai
from google.genai import types

from src.config import EMBEDDING_MODEL, GENERATION_MODEL


class MissingApiKey(RuntimeError):
    pass


class Gemini:
    """Thin wrapper so the rest of the code (and the tests) only depend on two methods."""

    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise MissingApiKey("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        # The SDK retries 408/429/5xx with exponential backoff, so rate limits and brief outages don't fail a query.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30_000,
                retry_options=types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=30.0),
            ),
        )
        self.model = GENERATION_MODEL

    def generate_json(self, system: str, prompt: str, schema: type) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                response_mime_type="application/json",
                response_schema=schema,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return response.text or ""

    def embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return list(result.embeddings[0].values)
