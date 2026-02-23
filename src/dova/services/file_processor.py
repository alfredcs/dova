"""
File processor service for handling uploaded files.
"""

import base64
import io

from fastapi import HTTPException, UploadFile

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILES = 5
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".png", ".jpg", ".csv", ".md", ".tex"}


async def process_uploaded_file(file: UploadFile) -> str:
    """Process an uploaded file and return extracted text content.

    Supports .txt (UTF-8 text), .pdf (text extraction), and .png (base64 encoding).
    """
    filename = file.filename or "unknown"
    ext = _get_extension(filename)

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File {filename} exceeds maximum size of 10MB",
        )

    if ext in {".txt", ".csv", ".md", ".tex"}:
        return content.decode("utf-8")
    elif ext == ".pdf":
        return _extract_pdf_text(content)
    elif ext in {".png", ".jpg"}:
        return _encode_image(content, ext)

    return ""


def _get_extension(filename: str) -> str:
    """Extract lowercase file extension."""
    dot_idx = filename.rfind(".")
    if dot_idx == -1:
        return ""
    return filename[dot_idx:].lower()


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2."""
    from PyPDF2 import PdfReader

    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _encode_image(content: bytes, ext: str) -> str:
    """Base64-encode an image and wrap with a description marker."""
    from PIL import Image

    # Validate it's a real image
    Image.open(io.BytesIO(content)).verify()

    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    encoded = base64.b64encode(content).decode("ascii")
    return f"[Attached Image (base64-encoded {ext.lstrip('.')})]\ndata:{mime};base64,{encoded}"
