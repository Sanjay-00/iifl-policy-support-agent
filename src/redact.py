import re

# Deliberately narrow: only identifiers with a recognisable shape. Amounts, dates, percentages and
# tenures must survive, because the policy question often depends on them.
PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)),
    ("AADHAAR", re.compile(r"(?<![\d,])\d{4}[ -]?\d{4}[ -]?\d{4}(?![\d,])")),
    ("PHONE", re.compile(r"(?<![\d,])(?:\+91[ -]?)?[6-9]\d{9}(?![\d,])")),
]
ACCOUNT_NUMBER = re.compile(
    r"\b((?:loan|account|acct)\s*(?:no\.?|number|num|id|#)\s*(?:is\s+)?[:\-]?\s*|a/c\s*(?:no\.?)?\s*[:\-]?\s*)"
    r"([A-Z0-9-]*\d[A-Z0-9-]{4,})",
    re.IGNORECASE,
)


def redact(text: str) -> tuple[str, list[str]]:
    found = []
    text, count = ACCOUNT_NUMBER.subn(r"\1[ACCOUNT_NO]", text)
    if count:
        found.append("ACCOUNT_NO")
    for label, pattern in PATTERNS:
        text, count = pattern.subn(f"[{label}]", text)
        if count:
            found.append(label)
    return text, found
