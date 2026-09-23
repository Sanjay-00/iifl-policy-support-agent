import json

import pytest

from src.policies import load_chunks
from src.retrieval import Hit


class FakeLLM:
    """Returns queued responses instead of calling Gemini. An Exception in the queue is raised."""

    model = "fake-model"

    def __init__(self, responses=None, embed_error=False):
        self.responses = list(responses or [])
        self.prompts = []
        self.embed_error = embed_error

    def generate_json(self, system, prompt, schema):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)

    def embed(self, text):
        if self.embed_error:
            raise ConnectionError("embedding API down")
        # Deterministic bag-of-letters vector; good enough to exercise the fusion code path.
        vector = [0.0] * 26
        for ch in text.lower():
            if "a" <= ch <= "z":
                vector[ord(ch) - 97] += 1
        return vector


@pytest.fixture
def chunks():
    return {c.chunk_id: c for c in load_chunks()}


@pytest.fixture
def hits(chunks):
    return [
        Hit(chunks["POL-FC-001#foreclosure-charges"], 1.0, 0.8),
        Hit(chunks["POL-FC-001#eligibility-for-foreclosure"], 0.5, 0.7),
    ]
