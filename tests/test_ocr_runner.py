"""Unit tests for kiso_ocr_mcp.ocr_runner — v0.3 cloud-only contract."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kiso_ocr_mcp.ocr_runner import (
    check_health,
    describe_image,
    image_info,
    ocr_image,
)


_PNG_MIN = (
    b"\x89PNG\r\n\x1a\n"
    + b"\x00\x00\x00\x0dIHDR"
    + (64).to_bytes(4, "big")
    + (48).to_bytes(4, "big")
    + b"\x08\x02\x00\x00\x00"
)


@pytest.fixture
def png_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.png"
    f.write_bytes(_PNG_MIN + b"\x00" * 16)
    return f


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Clean any kiso/openrouter/litellm env between tests."""
    for var in (
        "KISO_OCR_BACKEND",
        "OPENROUTER_API_KEY",
        "LITELLM_BASE_URL",
        "LITELLM_API_KEY",
        "KISO_OCR_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


class TestBackendSelection:
    def test_default_backend_is_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        h = check_health()
        assert h["backend"] == "openrouter"

    def test_litellm_backend_selectable(self, monkeypatch):
        monkeypatch.setenv("KISO_OCR_BACKEND", "litellm")
        monkeypatch.setenv("LITELLM_BASE_URL", "http://lite:4000")
        h = check_health()
        assert h["backend"] == "litellm"

    def test_unknown_backend_unhealthy(self, monkeypatch):
        monkeypatch.setenv("KISO_OCR_BACKEND", "bogus")
        h = check_health()
        assert h["healthy"] is False
        assert any("bogus" in i for i in h["issues"])

    def test_tesseract_backend_no_longer_supported(self, monkeypatch, png_file):
        """v0.3 removes the local Tesseract backend entirely."""
        monkeypatch.setenv("KISO_OCR_BACKEND", "tesseract")
        h = check_health()
        assert h["healthy"] is False
        assert any("tesseract" in i.lower() for i in h["issues"])

    def test_gemini_alias_no_longer_supported(self, monkeypatch):
        """v0.3 renames 'gemini' (old opt-in name) to 'openrouter'."""
        monkeypatch.setenv("KISO_OCR_BACKEND", "gemini")
        h = check_health()
        assert h["healthy"] is False
        assert any("gemini" in i.lower() for i in h["issues"])


class TestOcrImageOpenrouter:
    """OCR via the default openrouter backend."""

    def test_missing_api_key_fails(self, monkeypatch, png_file):
        result = ocr_image(file_path=str(png_file))
        assert result["success"] is False
        assert "OPENROUTER_API_KEY" in result["stderr"]

    def test_file_not_found(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        result = ocr_image(file_path=str(tmp_path / "missing.png"))
        assert result["success"] is False
        assert "not found" in result["stderr"].lower()

    def test_unsupported_format(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        f = tmp_path / "x.txt"
        f.write_text("text")
        result = ocr_image(file_path=str(f))
        assert result["success"] is False
        assert "unsupported" in result["stderr"].lower()

    def test_file_too_large(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        huge = tmp_path / "big.png"
        huge.write_bytes(b"\x89PNG" + b"\x00" * (25 * 1024 * 1024))
        result = ocr_image(file_path=str(huge))
        assert result["success"] is False
        assert "too large" in result["stderr"].lower()

    def test_success_with_text(self, monkeypatch, png_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_ocr_mcp.ocr_runner._call_vision_llm",
            return_value="Hello World",
        ):
            result = ocr_image(file_path=str(png_file))
        assert result["success"] is True
        assert result["text"] == "Hello World"
        assert result["has_text"] is True
        assert result["format"] == "png"
        assert result["width"] == 64
        assert result["height"] == 48
        assert result["truncated"] is False
        assert result["backend"] == "openrouter"

    def test_no_text_detected(self, monkeypatch, png_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_ocr_mcp.ocr_runner._call_vision_llm",
            return_value="",
        ):
            result = ocr_image(file_path=str(png_file))
        assert result["success"] is True
        assert result["text"] == ""
        assert result["has_text"] is False

    def test_truncates_long_output(self, monkeypatch, png_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_ocr_mcp.ocr_runner._call_vision_llm",
            return_value="x" * 100_000,
        ):
            result = ocr_image(file_path=str(png_file))
        assert result["truncated"] is True
        assert len(result["text"]) <= 50_000

    def test_api_error_surfaces(self, monkeypatch, png_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_ocr_mcp.ocr_runner._call_vision_llm",
            side_effect=RuntimeError("vision API error (500): boom"),
        ):
            result = ocr_image(file_path=str(png_file))
        assert result["success"] is False
        assert "500" in result["stderr"]


class TestOcrImageLitellm:
    """OCR via the litellm backend — consumer's local LiteLLM gateway."""

    @pytest.fixture(autouse=True)
    def _force_litellm(self, monkeypatch):
        monkeypatch.setenv("KISO_OCR_BACKEND", "litellm")
        monkeypatch.setenv("LITELLM_BASE_URL", "http://lite:4000")

    def test_missing_base_url_fails(self, monkeypatch, png_file):
        monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
        result = ocr_image(file_path=str(png_file))
        assert result["success"] is False
        assert "LITELLM_BASE_URL" in result["stderr"]

    def test_success_uses_litellm_base_url(self, monkeypatch, png_file):
        with patch(
            "kiso_ocr_mcp.ocr_runner._call_vision_llm",
            return_value="text from litellm",
        ) as call:
            result = ocr_image(file_path=str(png_file))
        assert result["success"] is True
        assert result["text"] == "text from litellm"
        assert result["backend"] == "litellm"
        kwargs = call.call_args.kwargs
        assert kwargs["base_url"].startswith("http://lite:4000")

    def test_litellm_api_key_optional(self, monkeypatch, png_file):
        """LITELLM_API_KEY may be empty when gateway has no auth."""
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        with patch(
            "kiso_ocr_mcp.ocr_runner._call_vision_llm",
            return_value="text",
        ):
            result = ocr_image(file_path=str(png_file))
        assert result["success"] is True


class TestDescribeImage:
    """describe_image works on every backend (v0.3: same path as ocr_image)."""

    def test_openrouter_success(self, monkeypatch, png_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_ocr_mcp.ocr_runner._call_vision_llm",
            return_value="A cat on a red sofa.",
        ):
            result = describe_image(file_path=str(png_file))
        assert result["success"] is True
        assert result["description"].startswith("A cat")
        assert "text" not in result

    def test_litellm_success(self, monkeypatch, png_file):
        monkeypatch.setenv("KISO_OCR_BACKEND", "litellm")
        monkeypatch.setenv("LITELLM_BASE_URL", "http://lite:4000")
        with patch(
            "kiso_ocr_mcp.ocr_runner._call_vision_llm",
            return_value="A cat.",
        ):
            result = describe_image(file_path=str(png_file))
        assert result["success"] is True
        assert result["description"] == "A cat."


class TestOcrAndDescribeShareCodePath:
    """v0.3 contract: ocr_image and describe_image both call _call_vision_llm
    with different prompts, never branch by backend."""

    def test_ocr_uses_extract_prompt(self, monkeypatch, png_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_ocr_mcp.ocr_runner._call_vision_llm",
            return_value="x",
        ) as call:
            ocr_image(file_path=str(png_file))
        prompt = call.call_args.kwargs.get("prompt") or call.call_args.args[-1]
        assert "extract" in prompt.lower()

    def test_describe_uses_describe_prompt(self, monkeypatch, png_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_ocr_mcp.ocr_runner._call_vision_llm",
            return_value="x",
        ) as call:
            describe_image(file_path=str(png_file))
        prompt = call.call_args.kwargs.get("prompt") or call.call_args.args[-1]
        assert "describe" in prompt.lower()


class TestImageInfo:
    def test_png_dimensions(self, png_file):
        info = image_info(file_path=str(png_file))
        assert info["success"] is True
        assert info["format"] == "png"
        assert info["width"] == 64
        assert info["height"] == 48

    def test_not_found(self, tmp_path):
        info = image_info(file_path=str(tmp_path / "nope.png"))
        assert info["success"] is False


class TestCallVisionLlm:
    """Direct unit tests for the single _call_vision_llm function."""

    def test_posts_to_base_url(self, png_file):
        from kiso_ocr_mcp import ocr_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello"}}]
        }
        with patch("httpx.post", return_value=mock_response) as post:
            text = ocr_runner._call_vision_llm(
                base_url="http://example/v1",
                api_key="k",
                model="vendor/model",
                image_bytes=png_file.read_bytes(),
                mime_type="image/png",
                prompt="p",
            )
        assert text == "hello"
        url = post.call_args.args[0]
        assert url == "http://example/v1/chat/completions"

    def test_authorization_header_when_key_set(self, png_file):
        from kiso_ocr_mcp import ocr_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        with patch("httpx.post", return_value=mock_response) as post:
            ocr_runner._call_vision_llm(
                base_url="http://x/v1",
                api_key="secret",
                model="m",
                image_bytes=png_file.read_bytes(),
                mime_type="image/png",
                prompt="p",
            )
        headers = post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret"

    def test_no_auth_header_when_key_empty(self, png_file):
        from kiso_ocr_mcp import ocr_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        with patch("httpx.post", return_value=mock_response) as post:
            ocr_runner._call_vision_llm(
                base_url="http://x/v1",
                api_key="",
                model="m",
                image_bytes=png_file.read_bytes(),
                mime_type="image/png",
                prompt="p",
            )
        headers = post.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    def test_empty_then_empty_returns_empty(self, png_file):
        from kiso_ocr_mcp import ocr_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        with patch("httpx.post", return_value=mock_response) as post, \
             patch("kiso_ocr_mcp.ocr_runner.time.sleep"):
            result = ocr_runner._call_vision_llm(
                base_url="http://x/v1",
                api_key="k",
                model="m",
                image_bytes=png_file.read_bytes(),
                mime_type="image/png",
                prompt="p",
            )
        assert result == ""
        assert post.call_count == 3

    def test_reasoning_fallback(self, png_file):
        from kiso_ocr_mcp import ocr_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "", "reasoning": "fell back to here"},
            }],
        }
        with patch("httpx.post", return_value=mock_response):
            result = ocr_runner._call_vision_llm(
                base_url="http://x/v1",
                api_key="k",
                model="m",
                image_bytes=png_file.read_bytes(),
                mime_type="image/png",
                prompt="p",
            )
        assert "fell back" in result

    def test_http_error_raises(self, png_file):
        from kiso_ocr_mcp import ocr_runner

        mock_response = MagicMock(status_code=500, text="server boom")
        with patch("httpx.post", return_value=mock_response), \
             pytest.raises(RuntimeError, match="500"):
            ocr_runner._call_vision_llm(
                base_url="http://x/v1",
                api_key="k",
                model="m",
                image_bytes=png_file.read_bytes(),
                mime_type="image/png",
                prompt="p",
            )


class TestCheckHealth:
    def test_openrouter_healthy(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        h = check_health()
        assert h["healthy"] is True
        assert h["issues"] == []
        assert h["backend"] == "openrouter"

    def test_openrouter_missing_key_unhealthy(self, monkeypatch):
        h = check_health()
        assert h["healthy"] is False
        assert any("OPENROUTER_API_KEY" in i for i in h["issues"])

    def test_litellm_healthy(self, monkeypatch):
        monkeypatch.setenv("KISO_OCR_BACKEND", "litellm")
        monkeypatch.setenv("LITELLM_BASE_URL", "http://lite:4000")
        h = check_health()
        assert h["healthy"] is True
        assert h["backend"] == "litellm"

    def test_litellm_missing_base_url_unhealthy(self, monkeypatch):
        monkeypatch.setenv("KISO_OCR_BACKEND", "litellm")
        h = check_health()
        assert h["healthy"] is False
        assert any("LITELLM_BASE_URL" in i for i in h["issues"])

    def test_no_subprocess_used(self, monkeypatch):
        """v0.3 removes Tesseract — doctor must not probe for any binary."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch("subprocess.run") as run:
            check_health()
        run.assert_not_called()
