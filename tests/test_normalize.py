from __future__ import annotations

from llm_harness.eval.normalize import normalize_output


def test_normalize_device_iphone_specific_model() -> None:
    actual = {
        "category": "Hardware",
        "priority": "P3",
        "device": "iPhone 13",
        "needs_clarification": False,
        "missing_fields": [],
        "summary": "Phone cannot connect.",
    }
    normalized, warnings = normalize_output(actual, input_text="My iPhone 13 keeps dropping calls")

    assert normalized["device"] == "iPhone"
    assert warnings == []


def test_normalize_does_not_force_vpn_from_input_text() -> None:
    actual = {
        "category": "Software",
        "priority": "P2",
        "device": "Windows",
        "needs_clarification": False,
        "missing_fields": [],
        "summary": "Remote access issue.",
    }
    normalized, _ = normalize_output(actual, input_text="VPN login fails on Windows laptop")

    assert normalized["category"] == "Software"


def test_normalize_authentication_to_access() -> None:
    actual = {
        "category": "Authentication",
        "priority": "P3",
        "device": "Unknown",
        "needs_clarification": True,
        "missing_fields": ["employee_email"],
        "summary": "Need email.",
    }
    normalized, warnings = normalize_output(actual, input_text="Unable to login to payroll")

    assert normalized["category"] == "Access"
    assert "normalized_category_synonym" in warnings
