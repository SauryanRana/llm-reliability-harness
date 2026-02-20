from __future__ import annotations

from typing import Any

ALLOWED_CATEGORIES = [
    "VPN",
    "Email",
    "Access",
    "Laptop",
    "Network",
    "Printer",
    "Software",
    "Security",
    "Hardware",
]
ALLOWED_DEVICES = ["Windows", "Mac", "iPhone", "Android", "Unknown"]
ALLOWED_PRIORITIES = ["P1", "P2", "P3", "P4"]

_CATEGORY_SYNONYMS: dict[str, str] = {
    "auth": "Access",
    "authentication": "Access",
    "login": "Access",
    "password": "Access",
    "account": "Access",
    "wifi": "Network",
    "wi-fi": "Network",
    "internet": "Network",
    "outlook": "Email",
    "mail": "Email",
    "calendar": "Email",
    "zoom": "Software",
    "teams": "Software",
    "slack": "Software",
    "docker": "Software",
    "bitlocker": "Laptop",
    "blue screen": "Laptop",
    "bsod": "Laptop",
}


def normalize_output(actual: dict, input_text: str) -> tuple[dict, list[str]]:
    normalized = dict(actual)
    warnings: list[str] = []
    text = input_text.lower()

    category = _ensure_text_key(normalized, "category", "Software", warnings)
    priority = _ensure_text_key(normalized, "priority", "P3", warnings)
    _ensure_text_key(normalized, "summary", "", warnings)
    if "needs_clarification" not in normalized:
        normalized["needs_clarification"] = False
        warnings.append("defaulted_needs_clarification")
    if "missing_fields" not in normalized:
        normalized["missing_fields"] = []
        warnings.append("defaulted_missing_fields")

    normalized["category"] = _normalize_category(category, text, warnings)
    normalized["priority"] = _normalize_priority(priority, warnings)
    normalized["device"] = _normalize_device(normalized.get("device"), text, warnings)
    normalized["summary"] = str(normalized.get("summary", "")).strip()
    normalized["missing_fields"] = _normalize_missing_fields(normalized.get("missing_fields"), warnings)
    normalized["needs_clarification"] = _normalize_bool(normalized.get("needs_clarification"))

    # Keep the clarification fields internally consistent.
    if normalized["missing_fields"] and normalized["needs_clarification"] is False:
        normalized["needs_clarification"] = True
        warnings.append("coerced_needs_clarification_true")
    if normalized["needs_clarification"] is True and not normalized["missing_fields"]:
        warnings.append("needs_clarification_without_missing_fields")

    return normalized, _dedupe(warnings)


def _ensure_text_key(data: dict[str, Any], key: str, default: str, warnings: list[str]) -> str:
    value = data.get(key)
    if value is None:
        data[key] = default
        warnings.append(f"defaulted_{key}")
        return default
    text = str(value).strip()
    if not text:
        data[key] = default
        warnings.append(f"defaulted_{key}")
        return default
    data[key] = text
    return text


def _normalize_category(category: str, input_text_lower: str, warnings: list[str]) -> str:
    clean = category.strip()
    lowered = clean.lower()

    if clean in ALLOWED_CATEGORIES:
        return clean

    for key, mapped in _CATEGORY_SYNONYMS.items():
        if key in lowered:
            warnings.append("normalized_category_synonym")
            return mapped

    if "vpn" in lowered:
        warnings.append("normalized_category_synonym")
        return "VPN"

    warnings.append("category_out_of_set")
    return "Software"


def _normalize_device(device_value: object, input_text_lower: str, warnings: list[str]) -> str:
    inferred = _infer_device(input_text_lower)
    if isinstance(device_value, str):
        raw = device_value.strip()
    else:
        raw = ""

    if not raw:
        if inferred is not None:
            warnings.append("defaulted_device_from_text")
            return inferred
        warnings.append("defaulted_device")
        return "Unknown"

    lowered = raw.lower()
    if "iphone" in lowered or "ios" in lowered:
        return "iPhone"
    if "android" in lowered:
        return "Android"
    if "mac" in lowered or "macbook" in lowered:
        return "Mac"
    if "windows" in lowered:
        return "Windows"

    if raw == "Laptop" and inferred in {"Windows", "Mac"}:
        warnings.append("mapped_laptop_to_os_device")
        return inferred

    if raw in ALLOWED_DEVICES:
        return raw

    if inferred is not None:
        warnings.append("device_out_of_set_mapped_from_text")
        return inferred

    warnings.append("device_out_of_set")
    return "Unknown"


def _normalize_priority(priority: str, warnings: list[str]) -> str:
    clean = priority.strip().upper()
    if clean in ALLOWED_PRIORITIES:
        return clean
    if clean in {"HIGH", "URGENT"}:
        warnings.append("normalized_priority")
        return "P1"
    if clean in {"MEDIUM", "NORMAL"}:
        warnings.append("normalized_priority")
        return "P3"
    if clean in {"LOW"}:
        warnings.append("normalized_priority")
        return "P4"
    if clean.endswith(("1", "2", "3", "4")):
        maybe = f"P{clean[-1]}"
        if maybe in ALLOWED_PRIORITIES:
            warnings.append("normalized_priority")
            return maybe
    warnings.append("priority_out_of_set")
    return "P3"


def _normalize_missing_fields(value: object, warnings: list[str]) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            out.append(text)
        return _dedupe(out)
    warnings.append("defaulted_missing_fields")
    return []


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return bool(value)


def _infer_device(input_text_lower: str) -> str | None:
    if "windows" in input_text_lower:
        return "Windows"
    if "macbook" in input_text_lower or "mac" in input_text_lower:
        return "Mac"
    if "iphone" in input_text_lower or "ios" in input_text_lower:
        return "iPhone"
    if "android" in input_text_lower:
        return "Android"
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
