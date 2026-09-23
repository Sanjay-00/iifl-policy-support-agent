from src.policies import load_chunks
from src.retrieval import Retriever
from tests.conftest import FakeLLM


def test_bm25_ranks_by_document_content():
    retriever = Retriever(load_chunks())
    hits, mode = retriever.search("How much is the EMI bounce charge?", 3, "bm25")
    assert mode == "bm25"
    assert hits[0].chunk.chunk_id == "POL-EMI-002#emi-bounce-and-penal-charges"


def test_embedding_mode_reports_similarity(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retrieval.EMBEDDING_CACHE", tmp_path / "emb.json")
    retriever = Retriever(load_chunks(), embed=FakeLLM().embed)
    hits, mode = retriever.search("foreclosure charges", 3)
    assert mode == "embedding"
    assert all(h.similarity is not None for h in hits)
    assert (tmp_path / "emb.json").exists()


def test_falls_back_to_bm25_when_embeddings_fail(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retrieval.EMBEDDING_CACHE", tmp_path / "emb.json")
    retriever = Retriever(load_chunks(), embed=FakeLLM(embed_error=True).embed)
    hits, mode = retriever.search("foreclosure charges", 3)
    assert mode == "bm25"
    assert hits[0].similarity is None
