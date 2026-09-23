import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Callable

from rank_bm25 import BM25Okapi

from src.config import EMBEDDING_CACHE, EMBEDDING_MODEL
from src.policies import Chunk

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "be", "it",
    "i", "my", "me", "you", "your", "we", "our", "can", "do", "does", "how", "what", "when", "will",
    "with", "at", "by", "from", "this", "that", "if", "any", "there", "as", "has", "have", "been",
}


@dataclass
class Hit:
    chunk: Chunk
    score: float
    similarity: float | None  # cosine similarity; None when ranked by BM25


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# Prefix formats recommended for gemini-embedding-2 asymmetric retrieval.
def query_text(question: str) -> str:
    return f"task: search result | query: {question}"


def document_text(chunk: Chunk) -> str:
    return f"title: {chunk.title} - {chunk.section} | text: {chunk.text}"


class Retriever:
    """Ranks policy sections by embedding similarity, falling back to BM25 if embeddings are unavailable.

    BM25+embedding fusion (RRF, k=60) was tried and scored below embeddings alone on the tuning set; see docs/EVALUATION.md.
    """

    def __init__(self, chunks: list[Chunk], embed: Callable[[str], list[float]] | None = None):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])
        self.embed = embed
        self.vectors = None
        if embed:
            try:
                self.vectors = self._chunk_vectors()
            except Exception:
                self.vectors = None

    def search(self, query: str, k: int, mode: str = "embedding") -> tuple[list[Hit], str]:
        similarities = self._similarities(query) if mode == "embedding" else None
        if similarities is not None:
            order = sorted(range(len(self.chunks)), key=lambda i: similarities[i], reverse=True)
            return [Hit(self.chunks[i], similarities[i], similarities[i]) for i in order[:k]], "embedding"

        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [Hit(self.chunks[i], float(scores[i]), None) for i in order[:k]], "bm25"

    def _similarities(self, query: str) -> list[float] | None:
        if not self.vectors:
            return None
        try:
            query_vector = self.embed(query_text(query))
        except Exception:
            return None
        return [cosine(query_vector, v) for v in self.vectors]

    def _chunk_vectors(self) -> list[list[float]]:
        # Cache keyed by model + exact text, so editing one policy section only re-embeds that section.
        cache = json.loads(EMBEDDING_CACHE.read_text()) if EMBEDDING_CACHE.exists() else {}
        vectors = []
        for chunk in self.chunks:
            text = document_text(chunk)
            key = hashlib.sha256(f"{EMBEDDING_MODEL}\n{text}".encode()).hexdigest()
            if key not in cache:
                cache[key] = self.embed(text)
            vectors.append(cache[key])
        EMBEDDING_CACHE.parent.mkdir(exist_ok=True)
        EMBEDDING_CACHE.write_text(json.dumps(cache))
        return vectors
