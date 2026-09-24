"""Demo UI. Presentation only: every answer comes from SupportAgent.answer(), the same call the CLI makes.

    streamlit run app.py
"""
import json

import streamlit as st

from src.config import ROOT
from src.llm import Gemini, MissingApiKey
from src.pipeline import SupportAgent

ACTION_COLOURS = {"respond": "green", "escalate": "orange"}
CONFIDENCE_COLOURS = {"high": "green", "medium": "orange", "low": "red"}
# Same file the CLI's --batch example uses, so the samples live in one place.
SAMPLE_QUESTIONS = json.loads((ROOT / "data" / "sample_questions.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="Policy support agent", page_icon=":material/support_agent:")
st.title("Policy support agent")
st.caption("Answers come from three synthetic policy documents written for this exercise. "
           "They are not official IIFL policies.")


@st.cache_resource
def get_agent() -> SupportAgent:
    # One agent per server process: loads policy sections and their cached embeddings once.
    return SupportAgent(Gemini())


def use_sample() -> None:
    if st.session_state.sample:
        st.session_state.question = st.session_state.sample


st.selectbox("Try a sample question", SAMPLE_QUESTIONS, index=None, key="sample",
             placeholder="Pick one to fill the box below", on_change=use_sample)

with st.form("ask"):
    question = st.text_area(
        "Customer question",
        key="question",
        placeholder="e.g. What are the foreclosure charges on a fixed-rate personal loan?",
    )
    submitted = st.form_submit_button("Submit", type="primary")

if submitted:
    try:
        agent = get_agent()
    except MissingApiKey as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Checking the policy documents..."):
        result = agent.answer(question)

    with st.container(border=True):
        st.markdown(result["answer"])

    with st.container(horizontal=True):
        st.badge(f"Action: {result['action']}", color=ACTION_COLOURS[result["action"]])
        st.badge(f"Confidence: {result['confidence']}", color=CONFIDENCE_COLOURS[result["confidence"]])
        st.badge(f"Category: {result['category']}", color="gray")
    st.markdown(f"**Source:** {result['source'] or 'none'}")

    st.caption("Structured result, the same six fields the CLI prints")
    st.json({key: value for key, value in result.items() if key != "debug"})

    with st.expander("Evidence / debug"):
        st.caption("Query as sent to the model (after PII redaction)")
        st.code(result["query"], language=None, wrap_lines=True)
        st.json(result["debug"])
