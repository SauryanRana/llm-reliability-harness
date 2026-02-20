from __future__ import annotations

from llm_harness.eval.rules import (
    build_output_from_signals,
    infer_category,
    infer_priority,
    normalize_missing_fields_to_canonical,
)


def test_category_overrides_security_access_software_network_printer_vpn() -> None:
    base = {
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

    security = dict(base, mentions_email=True, security_incident=True)
    assert infer_category(security, "Outlook issue after suspicious link") == "Security"

    access = dict(base, mentions_email=True, access_request=True)
    assert infer_category(access, "Need onboarding access for Okta and Jira") == "Access"

    printer = dict(base, mentions_printer=True, mentions_software_app=True)
    assert infer_category(printer, "Printer spooler fails and paper jam") == "Printer"

    software = dict(base, mentions_software_app=True, mentions_wifi_or_network=False, scope="single_user")
    assert infer_category(software, "Teams is stuck loading and freezing") == "Software"

    network = dict(base, mentions_software_app=True, mentions_wifi_or_network=True, scope="multiple_users")
    assert infer_category(network, "Teams outage because wifi down for all users") == "Network"

    vpn = dict(base, mentions_vpn=True)
    assert infer_category(vpn, "AnyConnect error 809 on home internet") == "VPN"


def test_teams_stuck_loading_is_software_even_without_signal() -> None:
    signals = {
        "mentions_software_app": False,
        "mentions_wifi_or_network": False,
        "access_request": False,
        "security_incident": False,
        "mentions_printer": False,
        "mentions_vpn": False,
        "scope": "single_user",
    }
    assert infer_category(signals, "Teams is stuck loading and not opening") == "Software"


def test_wifi_outage_multiple_users_is_network() -> None:
    signals = {
        "mentions_software_app": False,
        "mentions_wifi_or_network": True,
        "access_request": False,
        "security_incident": False,
        "mentions_printer": False,
        "mentions_vpn": False,
        "scope": "multiple_users",
    }
    assert infer_category(signals, "Wi-Fi is down on 3rd floor affecting multiple users") == "Network"


def test_access_request_with_email_text_is_access_not_email() -> None:
    signals = {
        "mentions_email": True,
        "access_request": True,
        "security_incident": False,
        "mentions_printer": False,
        "mentions_vpn": False,
        "mentions_software_app": False,
        "mentions_wifi_or_network": False,
        "scope": "single_user",
    }
    assert infer_category(signals, "Please grant access to Jira for user@corp.com") == "Access"


def test_access_wins_when_security_signal_true_but_no_security_keywords() -> None:
    signals = {
        "mentions_email": True,
        "access_request": True,
        "security_incident": True,
        "mentions_printer": False,
        "mentions_vpn": False,
        "mentions_software_app": False,
        "mentions_wifi_or_network": False,
        "scope": "single_user",
    }
    assert infer_category(signals, "Need onboarding access to Confluence for new employee") == "Access"


def test_security_keyword_present_maps_to_security() -> None:
    signals = {
        "mentions_email": True,
        "access_request": True,
        "security_incident": False,
        "mentions_printer": False,
        "mentions_vpn": False,
        "mentions_software_app": False,
        "mentions_wifi_or_network": False,
        "scope": "single_user",
    }
    assert infer_category(signals, "Possible phishing and credential theft incident") == "Security"


def test_missing_fields_normalization_uses_dataset_union() -> None:
    allowed = {"error_message_or_screenshot", "speed_test_result"}
    fields = ["exact_error_message", "speed_test", "unknown_custom_field"]
    normalized = normalize_missing_fields_to_canonical(fields, allowed_missing_fields=allowed, strict_missing_fields=True)
    assert normalized == ["error_message_or_screenshot", "speed_test_result"]


def test_needs_clarification_requires_missing_fields() -> None:
    actionable_signals = {
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
        "summary": "VPN failure with clear error details.",
    }
    output_ok, _ = build_output_from_signals(
        actionable_signals,
        "VPN error 809 on Windows since 09:00 from home office",
    )
    assert output_ok["missing_fields"] == []
    assert output_ok["needs_clarification"] is False

    access_signals = dict(actionable_signals, mentions_vpn=False, access_request=True)
    output_access, _ = build_output_from_signals(
        access_signals,
        "Please provide access to SAP for a new employee joining Monday.",
    )
    assert output_access["missing_fields"] != []
    assert output_access["needs_clarification"] is True


def test_wifi_outage_actionable_has_no_clarification_needed() -> None:
    signals = {
        "device_hint": "unknown",
        "mentions_vpn": False,
        "mentions_email": False,
        "mentions_wifi_or_network": True,
        "mentions_printer": False,
        "mentions_software_app": False,
        "mentions_laptop_issue": False,
        "access_request": False,
        "security_incident": False,
        "scope": "multiple_users",
        "error_codes": [],
        "urgency_words": False,
        "summary": "",
    }
    output, _ = build_output_from_signals(
        signals,
        "Wi-Fi is down on 3rd floor affecting multiple users since 10:20",
    )
    assert output["needs_clarification"] is False
    assert output["missing_fields"] == []


def test_lost_phone_with_company_email_is_security_p1_with_required_clarification() -> None:
    signals = {
        "device_hint": "iphone",
        "mentions_vpn": False,
        "mentions_email": True,
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
    text = "I lost my phone and it has company email on it."
    assert infer_category(signals, text) == "Security"
    assert infer_priority(signals, text) == "P1"
    output, _ = build_output_from_signals(signals, text)
    assert output["needs_clarification"] is True
    assert output["missing_fields"] == ["device_type", "phone_number_or_asset_id", "last_known_time"]


def test_access_keywords_beat_hardware_words() -> None:
    signals = {
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
    text = "Need Jira access for onboarding; also my keyboard is old."
    assert infer_category(signals, text) == "Access"


def test_bsod_or_bitlocker_maps_to_laptop_high_priority() -> None:
    signals = {
        "device_hint": "windows",
        "mentions_vpn": False,
        "mentions_email": False,
        "mentions_wifi_or_network": False,
        "mentions_printer": False,
        "mentions_software_app": True,
        "mentions_laptop_issue": True,
        "access_request": False,
        "security_incident": False,
        "scope": "single_user",
        "error_codes": [],
        "urgency_words": False,
        "summary": "",
    }
    text = "BitLocker recovery key prompt and boot loop after startup repair."
    assert infer_category(signals, text) == "Laptop"
    assert infer_priority(signals, text) in {"P1", "P2"}


def test_shared_mailbox_or_calendar_issue_maps_to_email() -> None:
    signals = {
        "device_hint": "unknown",
        "mentions_vpn": False,
        "mentions_email": False,
        "mentions_wifi_or_network": True,
        "mentions_printer": False,
        "mentions_software_app": False,
        "mentions_laptop_issue": False,
        "access_request": False,
        "security_incident": False,
        "scope": "multiple_users",
        "error_codes": [],
        "urgency_words": False,
        "summary": "",
    }
    text = "Shared mailbox calendar invitation is missing for the whole team."
    assert infer_category(signals, text) == "Email"


def test_needs_clarification_invariant_true_requires_non_empty_missing_fields() -> None:
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
        "summary": "",
    }
    output, _ = build_output_from_signals(signals, "VPN error 809 on Windows")
    if output["needs_clarification"]:
        assert output["missing_fields"]
    if not output["missing_fields"]:
        assert output["needs_clarification"] is False


def test_github_timeout_from_company_network_is_p2() -> None:
    signals = {
        "device_hint": "unknown",
        "mentions_vpn": False,
        "mentions_email": False,
        "mentions_wifi_or_network": True,
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
    text = "I can't access GitHub from the company network; it times out."
    assert infer_category(signals, text) == "Network"
    assert infer_priority(signals, text) == "P2"
