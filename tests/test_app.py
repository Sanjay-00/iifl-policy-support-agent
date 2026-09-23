import json

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.pipeline import SupportAgent
from tests.conftest import FakeLLM
from tests.test_pipeline import GOOD, QUESTION


def test_app_displays_exactly_what_the_pipeline_returns(tmp_path, monkeypatch):
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr("src.retrieval.EMBEDDING_CACHE", tmp_path / "emb.json")
    monkeypatch.setattr(SupportAgent.__init__, "__defaults__", ("embedding", log))
    monkeypatch.setattr("src.llm.Gemini", lambda: FakeLLM([GOOD]))
    st.cache_resource.clear()

    expected = SupportAgent(FakeLLM([GOOD])).answer(QUESTION)

    at = AppTest.from_file("../app.py", default_timeout=30).run()
    at.text_area[0].input(QUESTION)
    at.button[0].click().run()

    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert expected["answer"] in rendered
    assert f"Action: {expected['action']}" in rendered
    assert f"Confidence: {expected['confidence']}" in rendered
    assert f"Category: {expected['category']}" in rendered
    # The app's call went through the pipeline's own audit logging: one line per answer() call.
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert {k: records[1][k] for k in expected if k != "debug"} == {k: v for k, v in expected.items() if k != "debug"}
