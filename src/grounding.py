import re

from src.generate import Draft
from src.retrieval import Hit


def normalize(text: str) -> str:
    text = text.lower().replace("₹", "rs ").replace("inr", "rs ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 2,00,000 -> 200000
    text = re.sub(r"[^a-z0-9%.\s]", " ", text)
    text = re.sub(r"\.(?!\d)", " ", text)  # drop sentence dots but keep decimals like 2.5
    return " ".join(text.split())


def numbers_in(text: str) -> set[str]:
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in re.findall(r"\d+(?:\.\d+)?", normalize(text))}


def check_grounding(draft: Draft, hits: list[Hit]) -> list[str]:
    """Returns the list of failed checks; empty means the answer is backed by the retrieved evidence."""
    retrieved = {h.chunk.chunk_id: h.chunk for h in hits}
    problems = []

    unknown = [cid for cid in draft.cited_chunk_ids if cid not in retrieved]
    if unknown:
        problems.append(f"cited sections that were not provided: {unknown}")
    cited = [retrieved[cid] for cid in draft.cited_chunk_ids if cid in retrieved]
    evidence = normalize(" ".join(f"{c.section} {c.text}" for c in cited))

    if draft.coverage != "none":
        if not cited:
            problems.append("answer has no valid citation")
        if not draft.supporting_quotes:
            problems.append("answer has no supporting quote")

    for quote in draft.supporting_quotes:
        if normalize(quote) not in evidence:
            problems.append(f"quote not found in cited sections: {quote!r}")

    # Figures are checked strictly: a wrong fee or timeline is the most harmful mistake here.
    unsupported = numbers_in(draft.answer) - numbers_in(evidence)
    if unsupported:
        problems.append(f"figures not found in cited sections: {sorted(unsupported)}")

    return problems
