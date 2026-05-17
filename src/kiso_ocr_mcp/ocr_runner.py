"""Image OCR core — cloud-only MCP wrapper.

Two backends are supported, selected by ``KISO_OCR_BACKEND``:

- ``openrouter`` (default) — direct OpenRouter calls. Requires
  ``OPENROUTER_API_KEY``.
- ``litellm`` — calls go through the consumer's local LiteLLM gateway
  at ``LITELLM_BASE_URL``. Optional ``LITELLM_API_KEY`` (when the
  gateway requires auth). Use this when image traffic must flow through
  the same governance layer as other LLM calls (cost tracking, PII
  filter, residency tag, quota, fallback).

Dimension detection for PNG and JPEG is dependency-free (parses file
headers directly) so the server stays Pillow-free.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import time
import unicodedata
from pathlib import Path


_MAX_OUTPUT_CHARS = 50_000
_MAX_FILE_SIZE = 20 * 1024 * 1024  # OpenAI-compat inline image limit
_EMPTY_RETRIES = 2
_RETRY_BACKOFF = (1, 2)
_SUPPORTED_BACKENDS = {"openrouter", "litellm"}

_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif",
})

_EXTRACT_PROMPT = (
    "Extract ALL text from this image exactly as written. "
    "Preserve the original layout, line breaks, and formatting as much as possible. "
    "Return only the extracted text, no commentary or description."
)

_DESCRIBE_PROMPT = (
    "Describe what is in this image. Include: main subject, text content if any, "
    "layout, colors, and any notable visual elements. Be concise but thorough."
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "google/gemini-2.0-flash-001"


def ocr_image(*, file_path: str) -> dict:
    return _dispatch_image(file_path=file_path, mode="ocr")


def describe_image(*, file_path: str) -> dict:
    return _dispatch_image(file_path=file_path, mode="describe")


def image_info(*, file_path: str) -> dict:
    path = Path(file_path).expanduser()
    if not path.is_file():
        return {
            "success": False,
            "file_name": None,
            "size_bytes": None,
            "format": None,
            "width": None,
            "height": None,
            "stderr": f"file not found: {file_path}",
        }
    dims = _get_dimensions(path)
    return {
        "success": True,
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "format": path.suffix.lower().lstrip("."),
        "width": dims[0] if dims else None,
        "height": dims[1] if dims else None,
        "stderr": "",
    }


def check_health() -> dict:
    issues: list[str] = []
    backend = _backend()
    result: dict = {
        "healthy": False,
        "issues": issues,
        "backend": backend,
    }
    if backend not in _SUPPORTED_BACKENDS:
        issues.append(
            f"KISO_OCR_BACKEND={backend!r} is not supported "
            f"(use one of: {sorted(_SUPPORTED_BACKENDS)})"
        )
        return result
    try:
        _resolve_endpoint(backend)
    except RuntimeError as exc:
        issues.append(str(exc))
    result["healthy"] = not issues
    return result


def _dispatch_image(*, file_path: str, mode: str) -> dict:
    path = Path(file_path).expanduser()
    if not path.is_file():
        return _fail(mode, f"file not found: {file_path}")
    if path.suffix.lower() not in _IMAGE_EXTENSIONS:
        return _fail(mode, f"unsupported image format: {path.suffix}")
    size = path.stat().st_size
    if size > _MAX_FILE_SIZE:
        return _fail(
            mode,
            f"file too large ({_format_size(size)}); limit is "
            f"{_format_size(_MAX_FILE_SIZE)}",
        )

    backend = _backend()
    if backend not in _SUPPORTED_BACKENDS:
        return _fail(
            mode,
            f"KISO_OCR_BACKEND={backend!r} is not supported "
            f"(use one of: {sorted(_SUPPORTED_BACKENDS)})",
        )

    try:
        base_url, api_key, model = _resolve_endpoint(backend)
        prompt = _EXTRACT_PROMPT if mode == "ocr" else _DESCRIBE_PROMPT
        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        text = _call_vision_llm(
            base_url=base_url,
            api_key=api_key,
            model=model,
            image_bytes=path.read_bytes(),
            mime_type=mime_type,
            prompt=prompt,
        )
    except RuntimeError as exc:
        return _fail(mode, str(exc), backend=backend)

    truncated = False
    if len(text) > _MAX_OUTPUT_CHARS:
        shown = text[:_MAX_OUTPUT_CHARS]
        last_nl = shown.rfind("\n")
        if last_nl > 0:
            shown = shown[:last_nl]
        text = shown
        truncated = True

    dims = _get_dimensions(path)
    result = {
        "success": True,
        "format": path.suffix.lower().lstrip("."),
        "width": dims[0] if dims else None,
        "height": dims[1] if dims else None,
        "truncated": truncated,
        "backend": backend,
        "stderr": "",
    }
    if mode == "ocr":
        result["text"] = text
        result["has_text"] = _has_meaningful_content(text)
    else:
        result["description"] = text
    return result


def _backend() -> str:
    return os.environ.get("KISO_OCR_BACKEND", "openrouter").lower()


def _resolve_endpoint(backend: str) -> tuple[str, str, str]:
    """Return ``(base_url, api_key, model)`` for the selected backend.

    Raises ``RuntimeError`` if required env vars are missing.
    """
    model = os.environ.get("KISO_OCR_MODEL", _DEFAULT_MODEL)
    if backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return _OPENROUTER_BASE_URL, api_key, model
    if backend == "litellm":
        base_url = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
        if not base_url:
            raise RuntimeError("LITELLM_BASE_URL is not set")
        return base_url, os.environ.get("LITELLM_API_KEY", ""), model
    raise RuntimeError(f"unknown backend: {backend}")


def _call_vision_llm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
) -> str:
    """POST a multimodal `/chat/completions` request to an OpenAI-compatible
    endpoint and return the text content. Single retry loop on empty
    responses; falls back to the ``reasoning`` field when ``content`` is
    empty (some OpenRouter providers stream into reasoning on first turn).
    """
    import httpx

    encoded = base64.b64encode(image_bytes).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": 8192,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{base_url.rstrip('/')}/chat/completions"

    for attempt in range(_EMPTY_RETRIES + 1):
        response = httpx.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"vision API error ({response.status_code}): {response.text[:500]}"
            )
        result = response.json()
        choices = result.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content", "") or ""
        if not _has_meaningful_content(content):
            reasoning = message.get("reasoning", "") or ""
            if _has_meaningful_content(reasoning):
                content = reasoning
        if _has_meaningful_content(content):
            return content
        if attempt < _EMPTY_RETRIES:
            time.sleep(_RETRY_BACKOFF[attempt])
    return ""


def _has_meaningful_content(text: str, min_chars: int = 3) -> bool:
    count = sum(1 for c in text if unicodedata.category(c)[0] in ("L", "N", "P"))
    return count >= min_chars


def _get_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        data = path.read_bytes()[:32]
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return (
                int.from_bytes(data[16:20], "big"),
                int.from_bytes(data[20:24], "big"),
            )
        if data[:2] == b"\xff\xd8":
            return _jpeg_dimensions(path)
    except OSError:
        return None
    return None


def _jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            h = int.from_bytes(data[i + 5:i + 7], "big")
            w = int.from_bytes(data[i + 7:i + 9], "big")
            return (w, h)
        i += 2 + length
    return None


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _fail(mode: str, message: str, *, backend: str | None = None) -> dict:
    result = {
        "success": False,
        "format": None,
        "width": None,
        "height": None,
        "truncated": False,
        "stderr": message,
    }
    if backend is not None:
        result["backend"] = backend
    if mode == "ocr":
        result["text"] = ""
        result["has_text"] = False
    else:
        result["description"] = ""
    return result
