from __future__ import annotations

import json
import time
from urllib import error, request

from .base import ProviderResult

DEFAULT_BASE_URL = "http://localhost:11434"

DEVICE_HINT_ENUM = ["windows", "mac", "iphone", "android", "unknown"]
SCOPE_ENUM = ["single_user", "multiple_users", "unknown"]

OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "device_hint": {"type": "string", "enum": DEVICE_HINT_ENUM},
        "mentions_vpn": {"type": "boolean"},
        "mentions_email": {"type": "boolean"},
        "mentions_wifi_or_network": {"type": "boolean"},
        "mentions_printer": {"type": "boolean"},
        "mentions_software_app": {"type": "boolean"},
        "mentions_laptop_issue": {"type": "boolean"},
        "access_request": {"type": "boolean"},
        "security_incident": {"type": "boolean"},
        "scope": {"type": "string", "enum": SCOPE_ENUM},
        "error_codes": {"type": "array", "items": {"type": "string"}},
        "urgency_words": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": [
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
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You extract TicketSignals from support tickets.\n"
    "Return exactly one single-line minified JSON object and nothing else.\n"
    "No markdown, no code fences, no commentary, and no extra leading/trailing text.\n"
    "Do not output final category/priority labels. Output only the requested signal fields.\n"
    "Always include all required TicketSignals fields exactly once.\n"
    "Field rules:\n"
    '- device_hint must be one of ["windows","mac","iphone","android","unknown"]\n'
    '- scope must be one of ["single_user","multiple_users","unknown"]\n'
    "- error_codes must be an array of strings (empty array if none)\n"
    "- summary must be concise and factual (max ~20 tokens)\n"
)


class OllamaProvider:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: int = 60,
        temperature: float = 0.0,
        num_predict: int = 320,
        num_ctx: int = 2048,
        json_mode: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.json_mode = json_mode

    def generate(self, item: dict, model: str) -> ProviderResult:
        ticket_text = str(item.get("input_text", ""))
        prompt_chars = len(ticket_text)
        user_prompt = _build_user_prompt(ticket_text)

        start = time.perf_counter()
        try:
            raw_text, usage = self._call_model(model=model, user_prompt=user_prompt)
        except _OllamaConnectionError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            raise RuntimeError(_unreachable_msg(self.base_url, model)) from exc
        except _OllamaHTTPError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            body = f"HTTP {exc.status_code}: {exc.response_body}".strip()
            return ProviderResult(
                actual=None,
                latency_ms=latency_ms,
                raw_text=body[:500],
                usage=None,
                status="error",
                error_type="HTTPError",
                error_msg=body[:300],
                prompt_chars=prompt_chars,
            )
        except _OllamaResponseShapeError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            raw_payload = json.dumps(exc.payload, ensure_ascii=False)[:500]
            return ProviderResult(
                actual=None,
                latency_ms=latency_ms,
                raw_text=raw_payload,
                usage=None,
                status="error",
                error_type="ResponseShapeError",
                error_msg=str(exc),
                prompt_chars=prompt_chars,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return ProviderResult(
                actual=None,
                latency_ms=latency_ms,
                usage=None,
                status="error",
                error_type=type(exc).__name__,
                error_msg=str(exc),
                prompt_chars=prompt_chars,
            )

        actual, json_valid, parse_error_type = parse_json_object_with_repair(raw_text)
        if not json_valid:
            retry_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Return exactly one single-line minified JSON object. "
                "No extra text. Include all required fields."
            )
            retry_error_type = parse_error_type
            retry_error_msg = _parse_error_message(parse_error_type)
            try:
                retry_text, retry_usage = self._call_model(model=model, user_prompt=retry_prompt)
                retry_actual, retry_ok, retry_parse_error = parse_json_object_with_repair(retry_text)
                if retry_ok:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    return ProviderResult(
                        actual=retry_actual,
                        latency_ms=latency_ms,
                        usage=retry_usage or usage,
                        status="ok",
                        prompt_chars=prompt_chars,
                        response_chars=len(retry_text),
                    )
                raw_text = retry_text
                usage = retry_usage or usage
                retry_error_type = retry_parse_error
                retry_error_msg = _parse_error_message(retry_parse_error)
            except _OllamaHTTPError as exc:
                raw_text = f"HTTP {exc.status_code}: {exc.response_body}".strip()
                retry_error_type = "HTTPError"
                retry_error_msg = raw_text[:300]
            except _OllamaResponseShapeError as exc:
                raw_text = json.dumps(exc.payload, ensure_ascii=False)
                retry_error_type = "ResponseShapeError"
                retry_error_msg = str(exc)

            latency_ms = (time.perf_counter() - start) * 1000.0
            return ProviderResult(
                actual=None,
                latency_ms=latency_ms,
                raw_text=raw_text[:500],
                usage=usage,
                status="error",
                error_type=retry_error_type,
                error_msg=retry_error_msg,
                prompt_chars=prompt_chars,
                response_chars=len(raw_text),
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return ProviderResult(
            actual=actual,
            latency_ms=latency_ms,
            usage=usage,
            status="ok",
            prompt_chars=prompt_chars,
            response_chars=len(raw_text),
        )

    def _call_model(self, model: str, user_prompt: str) -> tuple[str, dict | None]:
        preferred_format: object = "json" if self.json_mode else OUTPUT_JSON_SCHEMA
        try:
            return self._call_model_with_format(model=model, user_prompt=user_prompt, output_format=preferred_format)
        except _OllamaHTTPError as exc:
            if not _should_fallback_from_json_mode(self.json_mode, exc):
                raise
            return self._call_model_with_format(model=model, user_prompt=user_prompt, output_format=OUTPUT_JSON_SCHEMA)

    def _call_model_with_format(self, model: str, user_prompt: str, output_format: object) -> tuple[str, dict | None]:
        chat_payload = {
            "model": model,
            "stream": False,
            "format": output_format,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        try:
            chat_response = self._post_json("/api/chat", chat_payload)
            raw_text = _extract_chat_text(chat_response)
            usage = _extract_usage(chat_response)
            return raw_text, usage
        except (_OllamaEndpointNotFound, _OllamaResponseShapeError):
            generate_payload = {
                "model": model,
                "stream": False,
                "format": output_format,
                "prompt": f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}",
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                    "num_ctx": self.num_ctx,
                },
            }
            generate_response = self._post_json("/api/generate", generate_payload)
            raw_text = _extract_generate_text(generate_response)
            usage = _extract_usage(generate_response)
            return raw_text, usage

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            if exc.code == 404:
                raise _OllamaEndpointNotFound(path) from exc
            msg = _read_http_error(exc)
            raise _OllamaHTTPError(status_code=exc.code, response_body=msg[:500]) from exc
        except error.URLError as exc:
            raise _OllamaConnectionError(str(exc)) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama API returned invalid JSON at {url}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(f"Ollama API returned non-object payload at {url}")
        return parsed


def parse_json_object_from_text(text: str) -> tuple[dict | None, bool]:
    parsed, ok, _ = parse_json_object_with_repair(text)
    return parsed, ok


def parse_json_object_with_repair(text: str) -> tuple[dict | None, bool, str]:
    raw = text.strip()
    if not raw:
        return None, False, "EmptyOutput"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        candidate = extract_first_json_object(raw)
        if candidate is None:
            return None, False, "ExtractionFailure"
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None, False, "InvalidJSON"

    if not isinstance(parsed, dict):
        return None, False, "InvalidJSON"
    return parsed, True, ""


def extract_first_json_object(text: str) -> str | None:
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : idx + 1]

    return None


def _build_user_prompt(ticket_text: str) -> str:
    required_fields = ", ".join(OUTPUT_JSON_SCHEMA["required"])
    return (
        "Extract TicketSignals from this support ticket.\n"
        f"Ticket: {ticket_text}\n"
        "Return exactly one single-line minified JSON object only.\n"
        "No markdown, no code fences, and no commentary.\n"
        f"Include all required fields: {required_fields}."
    )


def _extract_chat_text(response: dict) -> str:
    message = response.get("message")
    if not isinstance(message, dict):
        raise _OllamaResponseShapeError("Ollama /api/chat response missing message object", response)
    content = message.get("content")
    if not isinstance(content, str):
        raise _OllamaResponseShapeError("Ollama /api/chat response missing message.content", response)
    return content


def _extract_generate_text(response: dict) -> str:
    text = response.get("response")
    if not isinstance(text, str):
        raise _OllamaResponseShapeError("Ollama /api/generate response missing response text", response)
    return text


def _extract_usage(response: dict) -> dict | None:
    prompt_tokens = _first_int(response, ["prompt_eval_count", "input_eval_count"])
    completion_tokens = _first_int(response, ["eval_count", "output_eval_count"])
    total_tokens = _first_int(response, ["total_tokens"])
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _first_int(payload: dict, keys: list[str]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


def _read_http_error(exc: error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    if body.strip():
        return body.strip()[:200]
    return str(exc)


def _unreachable_msg(base_url: str, model: str) -> str:
    return (
        f"Ollama not reachable at {base_url}. Start it with 'ollama serve' and ensure model exists: "
        f"'ollama pull {model}'"
    )


class _OllamaConnectionError(RuntimeError):
    pass


class _OllamaEndpointNotFound(RuntimeError):
    pass


class _OllamaHTTPError(RuntimeError):
    def __init__(self, status_code: int, response_body: str) -> None:
        super().__init__(f"Ollama HTTP error {status_code}")
        self.status_code = status_code
        self.response_body = response_body


class _OllamaResponseShapeError(RuntimeError):
    def __init__(self, message: str, payload: dict) -> None:
        super().__init__(message)
        self.payload = payload


def _should_fallback_from_json_mode(json_mode: bool, exc: _OllamaHTTPError) -> bool:
    if not json_mode:
        return False
    return exc.status_code in {400, 404, 422}


def _parse_error_message(error_type: str) -> str:
    if error_type == "EmptyOutput":
        return "Model returned empty output"
    if error_type == "ExtractionFailure":
        return "Could not extract a JSON object from model output"
    return "Model did not return a valid JSON object"
