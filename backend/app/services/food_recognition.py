"""Claude Vision food recognition service.

Sends food photos to Claude Vision API for identification and nutrition estimation.
Also supports text-based nutrition estimation as an AI fallback for food search.

All outputs follow the wiki rules:
- Quality as whole/mixed/processed (not moralistic good/bad)
- 4th grade reading level for food names
- Confidence 0-1 for each item
"""

import base64
import json
import logging
from io import BytesIO

import anthropic
from PIL import Image

from app.config import settings

logger = logging.getLogger("meld.food_recognition")

# Anthropic Vision rejects images >5 MB after base64 decode with
# `invalid_request_error: image exceeds 5 MB maximum`. We target a
# generous safety margin so transient compression variations don't
# bump into the ceiling. (MELD-BACKEND-J/H, 2026-05-02.)
_ANTHROPIC_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_DOWNSIZE_TARGET_BYTES = int(4.5 * 1024 * 1024)
_MAX_DIMENSION = 2000  # plenty of detail for food recognition

PHOTO_RECOGNITION_PROMPT = """Identify all food items visible in this photo.

For each item, estimate:
- name: simple, common name (4th grade reading level)
- serving_size: natural serving description (e.g., "1 medium breast", "1 cup", "2 slices")
- calories: estimated kcal for that serving
- protein: grams
- carbs: grams
- fat: grams
- quality: "whole" (unprocessed, natural), "mixed" (some processing), or "processed" (heavily processed)
- confidence: 0.0 to 1.0 (how sure you are about the identification and portions)

Be practical, not perfect. Round to nearest 5 calories. Err on the side of reasonable portions.

Respond with ONLY a JSON array, no other text:
[{"name": "...", "serving_size": "...", "calories": 0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "quality": "...", "confidence": 0.0}]"""

TEXT_RECOGNITION_PROMPT = """Estimate the nutrition for this food: {query}

Assume a standard single serving. Provide your best estimate.

Respond with ONLY a JSON array, no other text:
[{"name": "...", "serving_size": "...", "calories": 0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "quality": "whole|mixed|processed", "confidence": 0.0}]"""


def _downsize_for_anthropic(image_base64: str, media_type: str) -> tuple[str, str]:
    """Return (image_base64, media_type) guaranteed to fit Anthropic's 5 MB limit.

    Modern iPhone cameras shoot 12 MP HEIC at 6-10 MB. Anthropic Vision
    rejects anything >5 MB with `invalid_request_error`, so we have to
    downsize server-side BEFORE the API call. We always re-encode as JPEG
    since it's the smallest format Anthropic accepts and modern phones
    sometimes upload HEIC even when the field is `image/jpeg`.

    Privacy (Audit P2d): photo-library uploads carry EXIF including GPS
    coordinates (where the meal was eaten). The iOS client strips this, but
    we re-encode here as defense-in-depth so a non-stripping client can never
    leak location to Anthropic. We therefore ALWAYS re-encode (no
    pass-through) — PIL's JPEG save drops EXIF/GPS/ICC since we never pass
    `exif=`. Small images are re-encoded at full dimensions; only oversize
    ones are downsized.

    Raises ValueError on malformed base64 or undecodable image bytes —
    the caller surfaces that as a 400 to the iOS client.
    """
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(f"Malformed base64 image payload: {exc}") from exc

    try:
        img = Image.open(BytesIO(raw))
        # Force eager decode — Image.open is lazy and won't surface a bad
        # header until .load() is called (or a method that triggers it).
        img.load()
    except (OSError, Image.UnidentifiedImageError) as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc

    # JPEG can't hold an alpha channel; flatten alpha or palette modes onto
    # white. Image.convert is a no-op on already-RGB inputs.
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Re-encode at full dimensions first: strips metadata while preserving
    # detail. Covers the common case (image already under the size margin).
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    if buf.tell() <= _DOWNSIZE_TARGET_BYTES:
        return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"

    # Still too big — step down dimensions + quality until under the safety
    # margin. 2000 px is plenty of detail for food recognition; quality 25 is
    # the floor before artifacts make food unidentifiable.
    for quality in (85, 70, 55, 40, 25):
        resized = img.copy()
        resized.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.LANCZOS)
        buf = BytesIO()
        resized.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= _DOWNSIZE_TARGET_BYTES:
            logger.info(
                "Downsized food image from %d to %d bytes (q=%d)",
                len(raw), buf.tell(), quality,
            )
            return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"

    # Last-resort: smaller dimensions + lowest quality. Logged as a warning
    # since the image is now small enough Claude may struggle with detail.
    resized = img.copy()
    resized.thumbnail((1200, 1200), Image.LANCZOS)
    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=20, optimize=True)
    logger.warning(
        "Food image required aggressive downsize: %d -> %d bytes",
        len(raw), buf.tell(),
    )
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


class FoodRecognitionService:
    """Identifies food from photos or text using Claude Vision API."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def recognize_from_photo(self, image_base64: str, media_type: str = "image/jpeg") -> list[dict]:
        """Identify food items from a photo using Claude Vision.

        Args:
            image_base64: Base64-encoded image data
            media_type: MIME type (image/jpeg, image/png, etc.)

        Returns:
            List of food item dicts with name, serving_size, calories, macros, quality, confidence
        """
        # Always run through the downsizer — it's a no-op for already-small
        # images and bullet-proofs us against any client (current iOS, future
        # Android, web) that uploads a too-large photo.
        try:
            image_base64, media_type = _downsize_for_anthropic(image_base64, media_type)
        except ValueError as e:
            logger.warning("Food image preprocessing failed: %s", e)
            return []

        try:
            response = self.client.messages.create(
                model=settings.anthropic_model_sonnet,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {"type": "text", "text": PHOTO_RECOGNITION_PROMPT},
                    ],
                }],
            )

            result_text = response.content[0].text
            items = json.loads(result_text)

            # Ensure all items have required fields with defaults
            for item in items:
                item.setdefault("data_source", "ai_estimate")
                item.setdefault("confidence", 0.7)
                item.setdefault("quality", "mixed")

            logger.info("Recognized %d food items from photo", len(items))
            return items

        except json.JSONDecodeError as e:
            logger.error("Failed to parse Vision response as JSON: %s", e)
            return []
        except anthropic.APIError as e:
            logger.error("Food recognition failed: %s", e)
            return []

    def recognize_from_text(self, query: str) -> list[dict]:
        """Estimate nutrition from a text description (AI fallback for search).

        Used when USDA and OFF have no results for a food query.
        """
        try:
            response = self.client.messages.create(
                model=settings.anthropic_model_sonnet,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": TEXT_RECOGNITION_PROMPT.format(query=query),
                }],
            )

            result_text = response.content[0].text
            items = json.loads(result_text)

            for item in items:
                item.setdefault("data_source", "ai_estimate")
                item.setdefault("confidence", 0.6)

            logger.info("AI estimated nutrition for: %s", query)
            return items

        except (anthropic.APIError, json.JSONDecodeError) as e:
            logger.error("Text recognition failed: %s", e)
            return []


# Singleton
food_recognition = FoodRecognitionService()
