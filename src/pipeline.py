import json
from datetime import datetime, timezone

from pydantic import ValidationError

from src import config
from src.decide import decide
from src.generate import PROMPT_VERSION, draft_answer
from src.grounding import check_grounding
from src.policies import Chunk, load_chunks
from src.redact import redact
from src.retrieval import Retriever

# Cosine thresholds for gemini-embedding-2 (768 dims). Top-section similarity on the eval set:
#   expected respond:              0.686-0.856 (lowest: held-out "stop marketing messages")
#   not covered, off-topic:        0.575-0.645 (admin-prompt injection, credit card, unexplained debit)
#   not covered, near the domain:  0.652-0.678 (fixed deposit, gold loan)
#   partially answerable:          0.676-0.790 (waiver request, complaint status, moratorium)
# The groups overlap, so similarity cannot decide answerability. The gate only drops clearly
# off-topic questions to save an LLM call; coverage and grounding decide the rest. STRONG only
# separates high from medium confidence: it sits above every question the policies don't cover
# (max 0.678), but partially answerable questions can exceed it and still escalate.
MIN_SIMILARITY = 0.65
STRONG_SIMILARITY = 0.75

UNVERIFIED_MESSAGE = (
    "I could not give you a reliable answer from our policy documents, so I have passed your "
    "question to a support specialist who will get back to you."
)
HANDOFF_NOTE = " A support specialist will follow up on the rest of your query."


def not_found_message(chunks: list[Chunk]) -> str:
    # Built from the loaded documents so an off-topic reply always states the real scope.
    titles = list(dict.fromkeys(c.title for c in chunks))
    topics = titles[0] if len(titles) == 1 else ", ".join(titles[:-1]) + " and " + titles[-1]
    return (
        f"I can help with questions covered by our {topics}. I could not find this in those documents, "
        "so I have passed your question to a support specialist who will get back to you."
    )


class SupportAgent:
    def __init__(self, llm, retrieval_mode: str = "embedding", audit_log=config.AUDIT_LOG):
        self.llm = llm
        self.retrieval_mode = retrieval_mode
        self.audit_log = audit_log
        embed = llm.embed if retrieval_mode == "embedding" else None
        chunks = load_chunks()
        self.retriever = Retriever(chunks, embed=embed)
        self.not_found_message = not_found_message(chunks)

    def answer(self, question) -> dict:
        if not isinstance(question, str) or not question.strip():
            return self._finish(str(question or ""), "invalid_input",
                                "Please type your question about your loan.", None, "low", "respond",
                                {"reasons": ["empty_input"]})
        if len(question) > config.MAX_QUESTION_CHARS:
            # Redact before truncating: cutting first can split an identifier so the pattern misses it.
            return self._finish(redact(question)[0][:200] + "...", "invalid_input",
                                f"Please keep your question under {config.MAX_QUESTION_CHARS} characters.",
                                None, "low", "respond", {"reasons": ["input_too_long"]})

        query, redacted = redact(question.strip())
        hits, mode = self.retriever.search(query, config.TOP_K, self.retrieval_mode)
        top_similarity = hits[0].similarity
        debug = {
            "redacted": redacted,
            "retrieval_mode": mode,
            "retrieved": [
                {"chunk_id": h.chunk.chunk_id, "version": h.chunk.version,
                 "score": round(h.score, 4),
                 "similarity": None if h.similarity is None else round(h.similarity, 4)}
                for h in hits
            ],
        }

        # Evidence gate: skip the LLM when nothing relevant was retrieved.
        if mode == "embedding":
            no_evidence = top_similarity < MIN_SIMILARITY
        else:
            no_evidence = hits[0].score <= 0  # BM25 only: no query term appears in any section
        if no_evidence:
            debug["reasons"] = ["no_relevant_policy_found"]
            return self._finish(query, "unclassified", self.not_found_message, None, "low", "escalate", debug)

        draft, problems, attempts = None, [], 0
        debug["failed_attempts"] = []
        try:
            for attempts in (1, 2):
                try:
                    draft = draft_answer(self.llm, query, hits, feedback=problems or None)
                except ValidationError as exc:
                    draft, problems = None, [f"response did not match the schema: {exc.error_count()} errors"]
                else:
                    problems = check_grounding(draft, hits)
                if not problems:
                    break
                debug["failed_attempts"].append({"attempt": attempts, "problems": problems})
        except Exception as exc:  # network errors, quota, timeouts after SDK retries: never guess an answer
            debug["reasons"] = ["llm_unavailable", type(exc).__name__]
            return self._finish(query, "unclassified", UNVERIFIED_MESSAGE, None, "low", "escalate", debug)

        debug["attempts"] = attempts
        if draft is None:
            debug["reasons"] = ["invalid_llm_output", *problems]
            return self._finish(query, "unclassified", UNVERIFIED_MESSAGE, None, "low", "escalate", debug)

        strong = mode == "embedding" and top_similarity >= STRONG_SIMILARITY
        confidence, action, reasons = decide(draft, problems, strong)
        debug.update({
            "reasons": reasons,
            "coverage": draft.coverage,
            "conflict": draft.conflict,
            "cited_chunk_ids": draft.cited_chunk_ids,
            "supporting_quotes": draft.supporting_quotes,
        })

        cited = [h.chunk for h in hits if h.chunk.chunk_id in draft.cited_chunk_ids]
        if problems:
            # An answer that failed grounding twice is not shown to the customer.
            return self._finish(query, "unclassified", UNVERIFIED_MESSAGE, None, confidence, action, debug)

        answer = draft.answer
        if action == "escalate" and draft.coverage == "partial":
            answer += HANDOFF_NOTE
        category = cited[0].category if cited else "unclassified"
        source = "; ".join(c.source for c in cited) or None
        return self._finish(query, category, answer, source, confidence, action, debug)

    def _finish(self, query, category, answer, source, confidence, action, debug) -> dict:
        result = {
            "query": query,
            "category": category,
            "answer": answer,
            "source": source,
            "confidence": confidence,
            "action": action,
        }
        self._audit(result, debug)
        return {**result, "debug": debug}

    def _audit(self, result: dict, debug: dict) -> None:
        # The query here is already redacted; raw customer text is never written to disk.
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": getattr(self.llm, "model", None),
            "prompt_version": PROMPT_VERSION,
            **result,
            "debug": debug,
        }
        self.audit_log.parent.mkdir(exist_ok=True)
        with self.audit_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
