from __future__ import annotations

from llm_harness.eval.rules import (
    build_output_from_signals,
    infer_category,
    infer_device,
    infer_missing_fields,
    infer_priority,
)


def test_vpn_ticket_inferrs_vpn_category() -> None:
    signals = {"mentions_vpn": True}
    category = infer_category(signals, "VPN fails to connect from home")
    assert category == "VPN"


def test_wifi_outage_inferrs_network_and_p1_for_multiple_users() -> None:
    signals = {
        "mentions_wifi_or_network": True,
        "scope": "multiple_users",
        "security_incident": False,
        "mentions_vpn": False,
        "access_request": False,
        "urgency_words": False,
        "mentions_email": False,
        "mentions_laptop_issue": False,
        "summary": "",
    }
    assert infer_category(signals, "Wi-Fi down for whole floor") == "Network"
    assert infer_priority(signals, "Wi-Fi down for whole floor") == "P1"


def test_priority_security_incident_defaults_to_p1() -> None:
    signals = {
        "security_incident": True,
        "scope": "single_user",
        "mentions_wifi_or_network": False,
        "mentions_email": False,
        "mentions_laptop_issue": False,
        "mentions_vpn": False,
        "access_request": False,
        "urgency_words": False,
        "summary": "",
    }
    assert infer_priority(signals, "Possible phishing link clicked") == "P1"


def test_iphone_hint_inferrs_iphone_device() -> None:
    assert infer_device("iphone", "Mail app not syncing on phone") == "iPhone"


def test_access_missing_fields_are_canonical() -> None:
    signals = {
        "access_request": True,
        "mentions_vpn": False,
        "mentions_email": False,
        "mentions_wifi_or_network": False,
        "mentions_printer": False,
        "mentions_software_app": False,
        "mentions_laptop_issue": False,
        "security_incident": False,
    }
    text = "New employee starts Monday. Please provide Jira and Confluence access."
    missing = infer_missing_fields(signals, text)
    assert "employee_name" in missing
    assert "employee_email" in missing
    assert "team" in missing
    assert "access_level" in missing


def test_build_output_forces_needs_clarification_false_when_missing_fields_empty() -> None:
    signals = {
        "device_hint": "windows",
        "mentions_vpn": True,
        "mentions_email": False,
        "mentions_wifi_or_network": False,
        "mentions_printer": False,
        "mentions_software_app": False,
        "mentions_laptop_issue": False,
        "access_request": False,
        "security_incident": False,
        "scope": "single_user",
        "error_codes": ["809"],
        "urgency_words": False,
        "summary": "VPN issue is clear.",
    }

    output, _warnings = build_output_from_signals(signals, "VPN fails with error 809 on Windows since 09:00")

    assert output["missing_fields"] == []
    assert output["needs_clarification"] is False
