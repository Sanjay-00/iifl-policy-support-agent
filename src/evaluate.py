"""Runs data/eval_set.jsonl and prints per-question results plus summary metrics.

    python -m src.evaluate --retrieval-only [--retriever bm25]   # retrieval only, no generation
    python -m src.evaluate                                      # full pipeline with Gemini
"""
import argparse
import json

from src import config
from src.llm import Gemini
from src.pipeline import SupportAgent
from src.policies import load_chunks
from src.redact import redact
from src.retrieval import Retriever

EVAL_FILE = config.ROOT / "data" / "eval_set.jsonl"


def load_cases() -> list[dict]:
    with open(EVAL_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_of(case: dict) -> str:
    # Held-out cases were written after the prompt and thresholds were tuned; the first run is the reported one.
    return "holdout" if case["type"] == "holdout" else "tuning"


def evaluate_retrieval(cases: list[dict], mode: str) -> None:
    retriever = Retriever(load_chunks(), embed=Gemini().embed if mode == "embedding" else None)
    scores = {"tuning": [], "holdout": []}  # (recall@1, recall@3, reciprocal rank) per labelled case
    print(f"{'id':4} {'type':17} {'rank':>4} {'top_sim':>7}  top retrieved")
    for case in cases:
        if not case["question"].strip():
            continue
        hits, used_mode = retriever.search(redact(case["question"])[0], len(retriever.chunks), mode)
        ids = [h.chunk.chunk_id for h in hits]
        expected = set(case["expected_chunks"])
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in expected), None)
        sim = f"{hits[0].similarity:.3f}" if hits[0].similarity is not None else "-"
        print(f"{case['id']:4} {case['type']:17} {rank or '-':>4} {sim:>7}  {ids[0]}")
        if expected:
            scores[split_of(case)].append((
                len(expected & set(ids[:1])) / len(expected),
                len(expected & set(ids[:3])) / len(expected),
                1 / rank if rank else 0.0,
            ))
    print(f"\nmode={used_mode}")
    for split, rows in scores.items():
        r1, r3, mrr = (sum(column) / len(rows) for column in zip(*rows))
        print(f"{split:8} labelled={len(rows):2}  Recall@1={r1:.2f}  Recall@3={r3:.2f}  MRR={mrr:.2f}")


def evaluate_pipeline(cases: list[dict], mode: str) -> None:
    agent = SupportAgent(Gemini(), retrieval_mode=mode, audit_log=config.ROOT / "logs" / "eval_audit.jsonl")
    passed = {"tuning": [], "holdout": []}
    for case in cases:
        result = agent.answer(case["question"])
        failures = []
        if case["expected_action"] != "any" and result["action"] != case["expected_action"]:
            failures.append(f"action={result['action']}")
        if "expected_category" in case and result["category"] != case["expected_category"]:
            failures.append(f"category={result['category']}")
        for text in case.get("must_not_contain", []):
            if text.lower() in result["answer"].lower():
                failures.append(f"answer contains {text!r}")
        passed[split_of(case)].append(not failures)
        debug = result["debug"]
        print(f"\n[{'PASS' if not failures else 'FAIL'}] {case['id']} ({case['type']}) {case['question'][:80]!r}")
        print(f"  expected: {case['expected_chunks'] or '-'} -> {case['expected_action']}")
        print(f"  got:      {result['confidence']}/{result['action']}  source={result['source']}")
        print(f"  reasons:  {debug.get('reasons')}  attempts={debug.get('attempts', 0)}")
        for failed in debug.get("failed_attempts", []):
            print(f"  attempt {failed['attempt']} failed: {failed['problems']}")
        print(f"  answer:   {result['answer']}")
        if failures:
            print(f"  failures: {failures}")
    print()
    for split, results in passed.items():
        print(f"{split:8} {sum(results)}/{len(results)} passed")
    print(f"total    {sum(map(sum, passed.values()))}/{len(cases)} passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["embedding", "bm25"], default="embedding")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    cases = load_cases()
    if args.retrieval_only:
        evaluate_retrieval(cases, args.retriever)
    else:
        evaluate_pipeline(cases, args.retriever)


if __name__ == "__main__":
    main()
