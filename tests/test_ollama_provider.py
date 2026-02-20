from __future__ import annotations

from llm_harness.providers.ollama import (
    OUTPUT_JSON_SCHEMA,
    OllamaProvider,
    _OllamaHTTPError,
    extract_first_json_object,
    parse_json_object_from_text,
    parse_json_object_with_repair,
)


def test_extract_first_json_object_from_prose() -> None:
    text = (
        "Here is the result:\n"
        '{"category":"Access","priority":"P2","device":"Laptop","needs_clarification":false,'
        '"missing_fields":[],"summary":"VPN fails"}\n'
        "thanks"
    )
    candidate = extract_first_json_object(text)
    assert candidate is not None
    assert candidate.startswith("{")
    assert candidate.endswith("}")


def test_parse_json_object_from_text_salvages_embedded_json() -> None:
    text = (
        "I found this.\n"
        '{"category":"Hardware","priority":"P3","device":"Desktop","needs_clarification":true,'
        '"missing_fields":["employee_email"],"summary":"Need more details."}\n'
        "done."
    )
    parsed, ok = parse_json_object_from_text(text)
    assert ok is True
    assert parsed is not None
    assert parsed["category"] == "Hardware"
    assert parsed["missing_fields"] == ["employee_email"]


def test_parse_json_object_from_text_rejects_non_json() -> None:
    parsed, ok = parse_json_object_from_text("no json here")
    assert ok is False
    assert parsed is None


def test_parse_json_object_with_repair_classifies_extraction_failure() -> None:
    parsed, ok, error_type = parse_json_object_with_repair("preface text without braces")
    assert ok is False
    assert parsed is None
    assert error_type == "ExtractionFailure"


def test_output_schema_has_required_enums_and_keys() -> None:
    assert OUTPUT_JSON_SCHEMA["type"] == "object"
    assert OUTPUT_JSON_SCHEMA["additionalProperties"] is False
    required = set(OUTPUT_JSON_SCHEMA["required"])
    assert required == {
        "device_hint",
        "mentions_vpn",
        "mentions_email",
        "mentions_wifi_or_network",
        "mentions_printer",
        "mentions_software_app",
        "mentions_laptop_issue",
        "access_request",
        "security_incident",
        "scope",
        "error_codes",
        "urgency_words",
        "summary",
    }
    props = OUTPUT_JSON_SCHEMA["properties"]
    assert "enum" in props["device_hint"]
    assert "enum" in props["scope"]


def test_provider_generate_parses_chat_json_with_schema_format(monkeypatch) -> None:
    provider = OllamaProvider()
    seen_payloads: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict) -> dict:
        seen_payloads.append((path, payload))
        return {
            "message": {
                "content": (
                    '{"device_hint":"windows","mentions_vpn":true,"mentions_email":false,'
                    '"mentions_wifi_or_network":false,"mentions_printer":false,'
                    '"mentions_software_app":false,"mentions_laptop_issue":false,'
                    '"access_request":false,"security_incident":false,"scope":"single_user",'
                    '"error_codes":["809"],"urgency_words":true,"summary":"VPN login failing."}'
                )
            },
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

    monkeypatch.setattr(provider, "_post_json", fake_post_json)

    result = provider.generate({"id": "T001", "input_text": "VPN login fails on Windows"}, "qwen2.5:4b")

    assert result.actual is not None
    assert result.actual["mentions_vpn"] is True
    assert result.actual["device_hint"] == "windows"
    assert result.usage is not None
    assert seen_payloads
    path, payload = seen_payloads[0]
    assert path == "/api/chat"
    assert payload["format"] == OUTPUT_JSON_SCHEMA


def test_provider_retry_keeps_ticket_context(monkeypatch) -> None:
    provider = OllamaProvider()
    prompts: list[str] = []

    def fake_call_model(model: str, user_prompt: str) -> tuple[str, dict | None]:
        del model
        prompts.append(user_prompt)
        if len(prompts) == 1:
            return "not json", None
        return (
            '{"device_hint":"windows","mentions_vpn":true,"mentions_email":false,'
            '"mentions_wifi_or_network":false,"mentions_printer":false,'
            '"mentions_software_app":false,"mentions_laptop_issue":false,'
            '"access_request":false,"security_incident":false,"scope":"single_user",'
            '"error_codes":[],"urgency_words":false,"summary":"VPN issue."}',
            None,
        )

    monkeypatch.setattr(provider, "_call_model", fake_call_model)

    result = provider.generate({"id": "T001", "input_text": "VPN fails on Windows laptop"}, "qwen2.5:4b")

    assert result.actual is not None
    assert len(prompts) == 2
    assert "Ticket: VPN fails on Windows laptop" in prompts[0]
    assert "Ticket: VPN fails on Windows laptop" in prompts[1]
    assert "IMPORTANT: Return exactly one single-line minified JSON object." in prompts[1]


def test_provider_json_mode_falls_back_to_schema_on_unsupported_format(monkeypatch) -> None:
    provider = OllamaProvider(json_mode=True)
    seen_formats: list[object] = []

    def fake_call_model_with_format(model: str, user_prompt: str, output_format: object) -> tuple[str, dict | None]:
        del model, user_prompt
        seen_formats.append(output_format)
        if output_format == "json":
            raise _OllamaHTTPError(400, "unsupported format")
        return (
            '{"device_hint":"windows","mentions_vpn":true,"mentions_email":false,'
            '"mentions_wifi_or_network":false,"mentions_printer":false,'
            '"mentions_software_app":false,"mentions_laptop_issue":false,'
            '"access_request":false,"security_incident":false,"scope":"single_user",'
            '"error_codes":[],"urgency_words":false,"summary":"VPN issue."}',
            None,
        )

    monkeypatch.setattr(provider, "_call_model_with_format", fake_call_model_with_format)

    result = provider.generate({"id": "T001", "input_text": "VPN fails on Windows laptop"}, "qwen2.5:4b")

    assert result.actual is not None
    assert seen_formats[0] == "json"
    assert seen_formats[1] == OUTPUT_JSON_SCHEMA
