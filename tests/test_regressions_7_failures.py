from __future__ import annotations

from llm_harness.eval.rules import build_output_from_signals


def _base_signals() -> dict:
    return {
        "device_hint": "unknown",
        "mentions_vpn": False,
        "mentions_email": False,
        "mentions_wifi_or_network": False,
        "mentions_printer": False,
        "mentions_software_app": False,
        "mentions_laptop_issue": False,
        "access_request": False,
        "security_incident": False,
        "scope": "single_user",
        "error_codes": [],
        "urgency_words": False,
        "summary": "",
    }


def test_t003_access_starting_monday_stays_p3() -> None:
    signals = _base_signals()
    signals["access_request"] = True
    text = "New employee starts Monday. Please provide Jira and Confluence access. Manager: Lara."

    output, _ = build_output_from_signals(signals, text)

    assert output["category"] == "Access"
    assert output["priority"] == "P3"
    assert output["needs_clarification"] is True
    assert output["missing_fields"] == ["employee_name", "employee_email", "team", "access_level"]


def test_t011_vpn_hotspot_vs_home_wifi_device_unknown_and_exact_missing_fields() -> None:
    signals = _base_signals()
    signals["mentions_vpn"] = True
    signals["device_hint"] = "iphone"
    text = "VPN works from my phone hotspot but not from home Wi-Fi."

    output, _ = build_output_from_signals(signals, text)

    assert output["category"] == "VPN"
    assert output["priority"] == "P3"
    assert output["device"] == "Unknown"
    assert output["needs_clarification"] is True
    assert output["missing_fields"] == ["device_os", "vpn_client_name", "home_router_model"]


def test_t015_monitor_flicker_priority_p3() -> None:
    signals = _base_signals()
    text = "My monitor keeps flickering when connected via HDMI."

    output, _ = build_output_from_signals(signals, text)

    assert output["category"] == "Hardware"
    assert output["priority"] == "P3"
    assert output["needs_clarification"] is True
    assert output["missing_fields"] == ["laptop_model", "monitor_model", "cable_or_port_tested"]


def test_t020_slack_notifications_after_update_is_p4() -> None:
    signals = _base_signals()
    signals["mentions_software_app"] = True
    text = "Slack notifications stopped working on Android after the last update."

    output, _ = build_output_from_signals(signals, text)

    assert output["category"] == "Software"
    assert output["priority"] == "P4"
    assert output["device"] == "Android"
    assert output["needs_clarification"] is True
    assert output["missing_fields"] == ["slack_version", "when_started"]


def test_t021_lost_device_company_email_forces_security_triage_fields() -> None:
    signals = _base_signals()
    signals["mentions_email"] = True
    signals["security_incident"] = True
    signals["device_hint"] = "android"
    text = "I lost my phone and it has company email on it."

    output, _ = build_output_from_signals(signals, text)

    assert output["category"] == "Security"
    assert output["priority"] == "P1"
    assert output["device"] == "Unknown"
    assert output["needs_clarification"] is True
    assert output["missing_fields"] == ["device_type", "phone_number_or_asset_id", "last_known_time"]


def test_t023_slow_internet_connected_priority_p3_and_exact_missing_fields() -> None:
    signals = _base_signals()
    signals["mentions_wifi_or_network"] = True
    text = "Internet is very slow in the office today but Wi-Fi still connects."

    output, _ = build_output_from_signals(signals, text)

    assert output["category"] == "Network"
    assert output["priority"] == "P3"
    assert output["needs_clarification"] is True
    assert output["missing_fields"] == ["location_floor", "speed_test_result", "start_time"]


def test_t026_bitlocker_sets_windows_and_exact_recovery_fields() -> None:
    signals = _base_signals()
    text = "Forgot my BitLocker recovery key and now my laptop asks for it."

    output, _ = build_output_from_signals(signals, text)

    assert output["category"] == "Laptop"
    assert output["priority"] == "P1"
    assert output["device"] == "Windows"
    assert output["needs_clarification"] is True
    assert output["missing_fields"] == ["asset_id", "username", "is_company_managed"]
