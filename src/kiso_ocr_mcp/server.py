"""MCP server exposing image OCR as a tool."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import ocr_runner


mcp = FastMCP("kiso-ocr")


@mcp.tool()
def ocr_image(file_path: str) -> dict:
    """Extract text from an image via a cloud vision LLM.

    Backend is selected by ``KISO_OCR_BACKEND``: ``openrouter`` (default)
    or ``litellm``.

    Args:
        file_path: Path to the image file. Supported formats: png, jpg,
            jpeg, webp, gif, bmp, tiff, tif. Max size 20 MB.

    Returns:
        ``{"success": bool, "text": str, "has_text": bool, "format": str | None,
           "width": int | None, "height": int | None, "truncated": bool,
           "backend": str, "stderr": str}``.

    ``has_text`` is ``false`` when the image contains no meaningful
    characters (blank photo, pure graphics, etc.). Output is truncated
    at 50 000 chars with ``truncated: true``.
    """
    return ocr_runner.ocr_image(file_path=file_path)


@mcp.tool()
def describe_image(file_path: str) -> dict:
    """Describe an image's contents (subject, layout, colors, text) via a cloud vision LLM.

    Same backend selection as ``ocr_image``.

    Args:
        file_path: Path to the image file.

    Returns:
        ``{"success": bool, "description": str, "format": str | None,
           "width": int | None, "height": int | None, "truncated": bool,
           "backend": str, "stderr": str}``.
    """
    return ocr_runner.describe_image(file_path=file_path)


@mcp.tool()
def image_info(file_path: str) -> dict:
    """Return image metadata (format, size, dimensions) without calling an LLM.

    Args:
        file_path: Path to the image file.

    Returns:
        ``{"success": bool, "file_name": str | None, "size_bytes": int | None,
           "format": str | None, "width": int | None, "height": int | None,
           "stderr": str}``.
    """
    return ocr_runner.image_info(file_path=file_path)


@mcp.tool()
def doctor() -> dict:
    """Check backend reachability and credentials. Returns ``{"healthy": bool, "issues": [str], "backend": str}``."""
    return ocr_runner.check_health()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
