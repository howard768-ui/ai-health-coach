"""Regression tests for the Anthropic-image downsize helper.

MELD-BACKEND-J / MELD-BACKEND-H fired in production when iOS uploaded a
6.4 MB food photo and the backend forwarded it as-is to Claude Vision,
which rejected it with `image exceeds 5 MB maximum: 6411204 bytes`.

These tests pin the downsize contract:
- Every image is re-encoded to strip EXIF/GPS metadata (Audit P2d), so a
  meal photo's location never reaches Anthropic
- Large images come out under the 4.5 MB safety margin
- Output is always JPEG (smallest format Anthropic accepts)
- Malformed input raises ValueError (becomes a 400 to the client)

Run: cd backend && uv run python -m pytest tests/test_food_image_downsize.py -v
"""

from __future__ import annotations

import base64
import os
from io import BytesIO

import pytest
from PIL import Image

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from app.services.food_recognition import (
    _downsize_for_anthropic,
    _DOWNSIZE_TARGET_BYTES,
)


def _make_image_bytes(
    width: int,
    height: int,
    format: str = "JPEG",
    quality: int = 95,
    mode: str = "RGB",
) -> bytes:
    """Generate a synthetic image of approximately the requested size.

    Uses real CSPRNG bytes for the pixel data so JPEG/PNG can't compress
    the result to a tiny file. Pseudo-random patterns from arithmetic
    formulas have hidden periodicity that PNG palette mode can exploit
    down to ~50 KB, which defeats the size-based assertions in this file.
    """
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    raw_pixels = os.urandom(width * height * channels)
    img = Image.frombytes(mode, (width, height), raw_pixels)
    buf = BytesIO()
    img.save(buf, format=format, quality=quality)
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


# ── Small images: re-encoded to strip metadata (Audit P2d) ───────────────


def test_small_image_is_reencoded_under_margin():
    """An image under the size margin is no longer passed through unchanged —
    it is re-encoded so EXIF/GPS is stripped (defense-in-depth). It must come
    out a valid JPEG still under the safety margin."""
    raw = _make_image_bytes(800, 600)
    assert len(raw) < _DOWNSIZE_TARGET_BYTES

    out_b64, out_media = _downsize_for_anthropic(_b64(raw), "image/jpeg")

    assert out_media == "image/jpeg"
    out_raw = base64.b64decode(out_b64)
    assert len(out_raw) <= _DOWNSIZE_TARGET_BYTES
    Image.open(BytesIO(out_raw)).verify()


def test_exif_and_gps_metadata_is_stripped():
    """EXIF (including GPS — where the meal was eaten) must not survive to
    Anthropic. Embed EXIF tags + a GPS sub-IFD and assert the output has none."""
    img = Image.frombytes("RGB", (400, 400), os.urandom(400 * 400 * 3))
    exif = img.getexif()
    exif[0x010F] = "MeldTestCamera"        # Make
    exif[0x0132] = "2026:06:09 12:00:00"   # DateTime
    gps_ifd = exif.get_ifd(0x8825)         # GPS sub-IFD
    gps_ifd[1] = "N"
    gps_ifd[2] = (40.0, 44.0, 0.0)
    gps_ifd[3] = "W"
    gps_ifd[4] = (73.0, 59.0, 0.0)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92, exif=exif)
    raw = buf.getvalue()

    # Precondition: the input really carries EXIF + GPS.
    in_exif = Image.open(BytesIO(raw)).getexif()
    assert dict(in_exif), "test input should have EXIF"
    assert dict(in_exif.get_ifd(0x8825)), "test input should have GPS"

    out_b64, _ = _downsize_for_anthropic(_b64(raw), "image/jpeg")
    out_exif = Image.open(BytesIO(base64.b64decode(out_b64))).getexif()

    assert not dict(out_exif), "EXIF must be stripped"
    assert not dict(out_exif.get_ifd(0x8825)), "GPS must be stripped"


# ── Downsize the firing case ────────────────────────────────────────────


def test_oversized_image_is_downsized_under_safety_margin():
    """Replicates the MELD-BACKEND-J input (>5 MB binary).

    Output binary must fit under the 4.5 MB safety margin so base64
    transmission (~6 MB) doesn't poke Anthropic's 5 MB ceiling at the
    decoded-binary check.
    """
    # 4000x3000 random-noise JPEG at q=95 reliably lands ~6-8 MB.
    raw = _make_image_bytes(4000, 3000, quality=95)
    assert len(raw) > _DOWNSIZE_TARGET_BYTES, "test input wasn't oversize; pick larger dims"

    out_b64, out_media = _downsize_for_anthropic(_b64(raw), "image/jpeg")

    out_raw = base64.b64decode(out_b64)
    assert len(out_raw) <= _DOWNSIZE_TARGET_BYTES
    assert out_media == "image/jpeg"

    # Sanity: the output must still be a decodable image.
    Image.open(BytesIO(out_raw)).verify()


# ── Format coercion to JPEG ─────────────────────────────────────────────


def test_oversized_png_with_alpha_becomes_jpeg():
    """RGBA PNGs come out as RGB JPEGs (Anthropic doesn't need alpha; JPEG is
    smaller). The alpha channel gets flattened onto white."""
    raw = _make_image_bytes(3500, 2500, format="PNG", quality=100, mode="RGBA")
    assert len(raw) > _DOWNSIZE_TARGET_BYTES

    out_b64, out_media = _downsize_for_anthropic(_b64(raw), "image/png")

    assert out_media == "image/jpeg"
    out_img = Image.open(BytesIO(base64.b64decode(out_b64)))
    assert out_img.format == "JPEG"
    assert out_img.mode == "RGB"


# ── Malformed input ─────────────────────────────────────────────────────


def test_malformed_base64_raises_valueerror():
    """Garbage base64 -> ValueError so the route can return 400."""
    with pytest.raises(ValueError, match="Malformed base64"):
        _downsize_for_anthropic("not-valid-base64!!!", "image/jpeg")


def test_undecodable_image_bytes_raise_valueerror():
    """Valid base64 of large non-image bytes -> ValueError (Anthropic would've
    rejected it anyway, but we fail fast before the API round trip).

    NOTE: the helper short-circuits on inputs already under the safety
    margin, so we deliberately pass >4.5 MB of garbage to force the
    decode-and-resize path.
    """
    big_garbage = os.urandom(_DOWNSIZE_TARGET_BYTES + 1024)
    not_an_image = _b64(big_garbage)

    with pytest.raises(ValueError, match="Could not decode image"):
        _downsize_for_anthropic(not_an_image, "image/jpeg")
