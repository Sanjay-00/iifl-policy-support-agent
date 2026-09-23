from datetime import date

from src.policies import load_chunks

POLICY = """---
document_id: POL-T-{n}
title: Test Policy {n}
category: test
version: {n}.0
effective_date: {effective}
status: {status}
---

> synthetic notice that must not become a chunk

## Charges

Fee is Rs. {n}00.
"""


def write(dir, name, **fields):
    (dir / name).write_text(POLICY.format(**fields), encoding="utf-8")


def test_real_policies_are_split_into_sections(chunks):
    assert "POL-FC-001#foreclosure-charges" in chunks
    charges = chunks["POL-FC-001#foreclosure-charges"]
    assert charges.source == "loan_foreclosure_policy.md > Foreclosure Charges"
    assert charges.category == "foreclosure_prepayment"
    assert "4% of the principal outstanding" in charges.text
    assert not any("SYNTHETIC" in c.text for c in chunks.values())


def test_only_active_and_effective_versions_are_loaded(tmp_path):
    write(tmp_path, "old.md", n=1, effective="2025-01-01", status="superseded")
    write(tmp_path, "current.md", n=2, effective="2026-01-01", status="active")
    write(tmp_path, "future.md", n=3, effective="2027-01-01", status="active")

    loaded = load_chunks(tmp_path, today=date(2026, 6, 1))

    assert [c.version for c in loaded] == ["2.0"]
    assert loaded[0].text == "Fee is Rs. 200."
