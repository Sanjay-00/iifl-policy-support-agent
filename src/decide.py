from src.generate import Draft


def decide(draft: Draft, grounding_problems: list[str], strong_evidence: bool) -> tuple[str, str, list[str]]:
    """Maps evidence to (confidence, action, reasons). The model supplies the coverage and conflict
    labels, but never sets confidence or action directly."""
    if grounding_problems:
        return "low", "escalate", ["grounding_failed", *grounding_problems]
    if draft.coverage == "none":
        return "low", "escalate", ["not_covered_by_policy"]
    if draft.coverage == "partial":
        return "medium", "escalate", ["partially_covered_or_needs_account_data"]
    if draft.conflict:
        return "medium", "escalate", ["conflicting_policy_sections"]
    if not strong_evidence:
        return "medium", "respond", ["grounded_but_retrieval_evidence_moderate"]
    return "high", "respond", ["grounded_and_strong_retrieval"]
