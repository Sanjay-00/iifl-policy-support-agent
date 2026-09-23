import pytest

from src.redact import redact


@pytest.mark.parametrize("text, label", [
    ("my PAN is ABCDE1234F", "PAN"),
    ("pan abcde1234f", "PAN"),
    ("aadhaar 1234 5678 9012", "AADHAAR"),
    ("call me on 9876543210", "PHONE"),
    ("call me on +91 9876543210", "PHONE"),
    ("mail me at a.customer@example.com", "EMAIL"),
    ("my loan no LN20240012345 is overdue", "ACCOUNT_NO"),
    ("my loan no is LN2024001234", "ACCOUNT_NO"),
    ("account number: 50100234567", "ACCOUNT_NO"),
    ("a/c 50100234567", "ACCOUNT_NO"),
])
def test_identifiers_are_redacted(text, label):
    redacted, found = redact(text)
    assert f"[{label}]" in redacted
    assert label in found


@pytest.mark.parametrize("text", [
    "Cash payments above Rs. 2,00,000 are not accepted.",
    "Foreclosure charge is 4% of the principal outstanding.",
    "NOC is issued within 15 working days.",
    "My EMI is due on 2026-04-05 and is Rs. 12,500.",
    "I have paid 6 EMIs on my loan before the tenure ends.",
    "Can I change my loan account to NACH auto-debit?",
    "I paid Rs 1200000 last year",
])
def test_policy_language_and_amounts_are_untouched(text):
    assert redact(text) == (text, [])
