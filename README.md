# kiso-ocr-mcp

Image OCR exposed as a
[Model Context Protocol](https://modelcontextprotocol.io) server.

A thin cloud-only wrapper around a vision LLM. Two backends:

- **`openrouter`** (default) — direct OpenRouter calls. Single key:
  `OPENROUTER_API_KEY`. Useful for standalone usage.
- **`litellm`** — calls go through the consumer's local LiteLLM
  gateway at `LITELLM_BASE_URL`. Use this when image traffic must flow
  through the same governance layer as other LLM calls (cost tracking,
  PII filter, residency tag, quota, fallback).

Local-OCR engines are explicitly out of scope. Air-gapped consumers
should fork the plugin and add a local backend themselves.

Part of the [`kiso-run`](https://github.com/kiso-run) project.

## Install

```sh
uvx --from git+https://github.com/kiso-run/ocr-mcp@v0.3.0 kiso-ocr-mcp
```

No system dependencies (no Tesseract, no Pillow). Just Python ≥3.11
and an outbound HTTP path to your chosen backend endpoint.

## Required environment

| Variable             | Required (when)                              | Purpose                                                                   |
|----------------------|----------------------------------------------|---------------------------------------------------------------------------|
| `KISO_OCR_BACKEND`   | optional (default `openrouter`)              | Backend selector: `openrouter` or `litellm`                               |
| `OPENROUTER_API_KEY` | required when backend = `openrouter`         | OpenRouter auth                                                           |
| `LITELLM_BASE_URL`   | required when backend = `litellm`            | URL of the consumer's LiteLLM gateway (e.g. `http://litellm:4000/v1`)     |
| `LITELLM_API_KEY`    | optional, used when backend = `litellm`      | Bearer token for the LiteLLM gateway, if the gateway requires auth        |
| `KISO_OCR_MODEL`     | optional (default `google/gemini-2.0-flash-001`) | Model identifier — useful when the consumer registers the vision model in LiteLLM under a different name |

## MCP client config

### Backend `openrouter` (default — single-key usage)

```json
{
  "mcpServers": {
    "ocr": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/kiso-run/ocr-mcp@v0.3.0",
        "kiso-ocr-mcp"
      ],
      "env": { "OPENROUTER_API_KEY": "${env:OPENROUTER_API_KEY}" }
    }
  }
}
```

### Backend `litellm` (route through consumer's LiteLLM gateway)

```json
{
  "mcpServers": {
    "ocr": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/kiso-run/ocr-mcp@v0.3.0",
        "kiso-ocr-mcp"
      ],
      "env": {
        "KISO_OCR_BACKEND": "litellm",
        "LITELLM_BASE_URL": "http://litellm:4000/v1",
        "LITELLM_API_KEY": "${env:LITELLM_API_KEY}",
        "KISO_OCR_MODEL": "vision"
      }
    }
  }
}
```

## Tools

### `ocr_image(file_path)`

Extract text from an image. Returns `{success, text, has_text, format,
width, height, truncated, backend, stderr}`. `has_text` is `false` when
no meaningful characters were detected (blank photo, pure graphics).
Output truncated at 50 000 chars with `truncated: true`.

### `describe_image(file_path)`

Scene description (subject, layout, colors, text). Returns `{success,
description, format, width, height, truncated, backend, stderr}`. Same
code path as `ocr_image` with a different prompt.

### `image_info(file_path)`

File metadata (no LLM call). Returns `{success, file_name, size_bytes,
format, width, height, stderr}`. PNG and JPEG dimensions parsed
directly from the file header.

### `doctor()`

Reports runner health and active configuration:

```json
{
  "healthy": true,
  "issues": [],
  "backend": "openrouter"
}
```

Reports missing env vars per selected backend.

## Supported formats

`png`, `jpg`, `jpeg`, `webp`, `gif`, `bmp`, `tiff`, `tif`.
Max file size: 20 MB (typical OpenAI-compatible inline image limit).

## Reliability

- Empty-response retry up to 2 attempts with 1s/2s backoff.
- Reasoning-field fallback for model variants that route output to
  `reasoning` instead of `content`.
- Output cap: 50 000 chars with `truncated: true` flag.
- HTTP timeout: 120 s per call.

## Development

```sh
uv sync
uv run pytest tests/ -q                          # unit only
OPENROUTER_API_KEY=... uv run pytest tests/ -q   # include live test
```

## License

MIT.
