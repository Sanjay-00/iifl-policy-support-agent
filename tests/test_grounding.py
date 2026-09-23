from src.decide import decide
from src.generate import Draft
from src.grounding import check_grounding

CHARGES = "POL-FC-001#foreclosure-charges"


def draft(answer, quotes, cited=(CHARGES,), coverage="full", conflict=False):
    return Draft(answer=answer, cited_chunk_ids=list(cited), supporting_quotes=list(quotes),
                 coverage=coverage, conflict=conflict)


def test_paraphrased_answer_with_real_quote_passes(hits):
    d = draft("If you close a fixed-rate personal loan early, you pay 4% of what you still owe, plus GST.",
              ["Fixed-rate personal loans: 4% of the principal outstanding, plus GST."])
    assert check_grounding(d, hits) == []


def test_quote_matching_ignores_case_punctuation_and_currency_format(hits, chunks):
    from src.retrieval import Hit
    process = [Hit(chunks["POL-FC-001#foreclosure-process"], 1.0, 0.8)]
    d = draft("Cash above ₹2,00,000 is not accepted.", ["cash payments above rs 200000 are NOT accepted"],
              cited=["POL-FC-001#foreclosure-process"])
    assert check_grounding(d, process) == []


def test_changed_figure_is_rejected(hits):
    d = draft("The foreclosure charge on a fixed-rate personal loan is 0%.",
              ["Fixed-rate personal loans: 4% of the principal outstanding, plus GST."])
    problems = check_grounding(d, hits)
    assert any("figures not found" in p and "0" in p for p in problems)


def test_invented_quote_is_rejected(hits):
    d = draft("There is no charge.", ["Foreclosure is always free of charge."])
    assert any("quote not found" in p for p in check_grounding(d, hits))


def test_citation_outside_retrieved_sections_is_rejected(hits):
    d = draft("There is a charge.", ["4% of the principal outstanding"], cited=["POL-EMI-002#payment-modes"])
    problems = check_grounding(d, hits)
    assert any("not provided" in p for p in problems)
    assert any("no valid citation" in p for p in problems)


def test_answer_without_evidence_is_rejected(hits):
    d = draft("You pay 4%.", [], cited=[])
    assert len(check_grounding(d, hits)) >= 2


def test_not_covered_answer_needs_no_citation(hits):
    d = draft("The policy documents do not cover gold loans.", [], cited=[], coverage="none")
    assert check_grounding(d, hits) == []


def test_decision_table():
    full = draft("a", ["b"])
    assert decide(full, [], strong_evidence=True) == ("high", "respond", ["grounded_and_strong_retrieval"])
    assert decide(full, [], strong_evidence=False)[:2] == ("medium", "respond")
    assert decide(draft("a", ["b"], coverage="partial"), [], True)[:2] == ("medium", "escalate")
    assert decide(draft("a", ["b"], conflict=True), [], True)[:2] == ("medium", "escalate")
    assert decide(draft("a", [], coverage="none"), [], True)[:2] == ("low", "escalate")
    assert decide(full, ["figures not found"], True)[:2] == ("low", "escalate")
