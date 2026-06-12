"""Tests for content block parsing.

Covers the three functions in app.services.content_blocks that gate
what the coach sends to the client: sanitize_output, parse_content_blocks,
and flatten_to_markdown.
"""


from app.services.content_blocks import (
    DataCardBlock,
    TextBlock,
    flatten_to_markdown,
    parse_content_blocks,
    sanitize_output,
)


# ========================================================================
# parse_content_blocks
# ========================================================================


class TestParseContentBlocks:
    def test_plain_text_returns_single_text_block(self):
        blocks = parse_content_blocks("Just a regular message with no tags.")
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].value == "Just a regular message with no tags."

    def test_single_tag_alone(self):
        raw = "[[data:sleep_efficiency:91:%:above 7-day avg]]"
        blocks = parse_content_blocks(raw)
        assert len(blocks) == 1
        assert isinstance(blocks[0], DataCardBlock)
        assert blocks[0].metric == "sleep_efficiency"
        assert blocks[0].value == "91"
        assert blocks[0].unit == "%"
        assert blocks[0].subtitle == "above 7-day avg"

    def test_tag_between_text(self):
        raw = "Strong night. [[data:sleep_efficiency:91:%:good]] Train hard."
        blocks = parse_content_blocks(raw)
        assert len(blocks) == 3
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].value == "Strong night."
        assert isinstance(blocks[1], DataCardBlock)
        assert blocks[1].value == "91"
        assert isinstance(blocks[2], TextBlock)
        assert blocks[2].value == "Train hard."

    def test_multiple_tags(self):
        raw = (
            "Here's your recovery picture. "
            "[[data:hrv:45:ms:baseline 42]] "
            "[[data:resting_hr:58:bpm:baseline 60]] "
            "Both trending favorably."
        )
        blocks = parse_content_blocks(raw)
        assert len(blocks) == 4
        assert blocks[1].metric == "hrv"
        assert blocks[2].metric == "resting_hr"

    def test_tag_at_start(self):
        raw = "[[data:steps:8421::close to 10k goal]] Good work."
        blocks = parse_content_blocks(raw)
        assert len(blocks) == 2
        assert isinstance(blocks[0], DataCardBlock)
        assert blocks[0].unit == ""  # Empty unit allowed

    def test_tag_at_end(self):
        raw = "Your sleep. [[data:deep_sleep_minutes:48:min:solid]]"
        blocks = parse_content_blocks(raw)
        assert len(blocks) == 2
        assert isinstance(blocks[-1], DataCardBlock)

    def test_empty_subtitle_allowed(self):
        raw = "[[data:steps:10000:steps:]]"
        blocks = parse_content_blocks(raw)
        assert blocks[0].subtitle == ""

    def test_whitespace_around_text_is_stripped(self):
        raw = "  \n\n  Hello.  \n  [[data:hrv:45:ms:ok]]  \n  Bye."
        blocks = parse_content_blocks(raw)
        # No empty text blocks from the surrounding whitespace.
        text_blocks = [b for b in blocks if isinstance(b, TextBlock)]
        assert all(b.value.strip() == b.value for b in text_blocks)
        assert all(b.value for b in text_blocks)

    def test_malformed_tag_passes_through_as_text(self):
        # Only 3 segments instead of 4 — not a valid tag, stays as text.
        raw = "Text with [[data:hrv:45]] malformed tag."
        blocks = parse_content_blocks(raw)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert "[[data:hrv:45]]" in blocks[0].value

    def test_empty_input_returns_single_empty_block(self):
        blocks = parse_content_blocks("")
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].value == ""

    def test_preserves_markdown_in_text_blocks(self):
        raw = "Your **HRV** is *trending up*. [[data:hrv:50:ms:up]]"
        blocks = parse_content_blocks(raw)
        assert "**HRV**" in blocks[0].value
        assert "*trending up*" in blocks[0].value

    def test_preserves_bullet_lists_in_text_blocks(self):
        raw = "Options:\n- Run 30 min\n- Lift upper body\n- Yoga"
        blocks = parse_content_blocks(raw)
        assert len(blocks) == 1
        assert "- Run" in blocks[0].value
        assert "- Lift" in blocks[0].value


# ========================================================================
# flatten_to_markdown
# ========================================================================


class TestFlattenToMarkdown:
    def test_replaces_tag_with_bolded_value_unit(self):
        raw = "Your sleep was [[data:sleep_efficiency:91:%:above avg]] last night."
        out = flatten_to_markdown(raw)
        assert "**91%**" in out
        assert "[[" not in out
        assert "]]" not in out

    def test_empty_unit_bolds_value_only(self):
        raw = "Walked [[data:steps:8421::close to goal]] today."
        out = flatten_to_markdown(raw)
        assert "**8421**" in out

    def test_multiple_tags_all_flattened(self):
        raw = "HRV [[data:hrv:45:ms:baseline]] RHR [[data:resting_hr:58:bpm:baseline]]"
        out = flatten_to_markdown(raw)
        assert "**45ms**" in out
        assert "**58bpm**" in out
        assert "[[" not in out

    def test_no_tags_unchanged(self):
        raw = "Plain text with **bold** and *italic* markdown."
        out = flatten_to_markdown(raw)
        assert out == raw


# ========================================================================
# sanitize_output (em dash safety net)
# ========================================================================


class TestSanitizeOutput:
    def test_replaces_spaced_em_dash(self):
        raw = "Your sleep was solid — train hard today."
        out = sanitize_output(raw)
        assert "—" not in out
        assert "solid, train hard" in out

    def test_replaces_bare_em_dash(self):
        raw = "HRV up—good recovery."
        out = sanitize_output(raw)
        assert "—" not in out

    def test_leaves_regular_hyphens_alone(self):
        raw = "Your 7-day average is solid. This is a 4th-grade test."
        out = sanitize_output(raw)
        assert out == raw
        assert "7-day" in out
        assert "4th-grade" in out

    def test_leaves_en_dash_alone(self):
        # En dash (–, U+2013) is acceptable for number ranges.
        raw = "Aim for 7–10 reps."
        out = sanitize_output(raw)
        assert "–" in out  # en dash preserved

    def test_collapses_adjacent_commas(self):
        # Adjacent em dashes would produce ",," — collapsed to single comma.
        raw = "A — B — C"
        out = sanitize_output(raw)
        # After replacement: "A, B, C" (not "A,, B,, C")
        assert ",," not in out

    def test_preserves_markdown_and_tags(self):
        raw = "**Bold** with — dash and [[data:hrv:45:ms:ok]] tag."
        out = sanitize_output(raw)
        assert "**Bold**" in out
        assert "[[data:hrv:45:ms:ok]]" in out
        assert "—" not in out


# ========================================================================
# Integration: round-trip through the full pipeline
# ========================================================================


class TestPipelineIntegration:
    def test_full_pipeline_real_coach_response(self):
        """A realistic response goes through sanitize -> parse -> flatten."""
        # Imagine Claude slips an em dash in (breaking the prompt rule).
        raw = (
            "**Strong night.** Take the green light and train hard today.\n\n"
            "- Sleep efficiency [[data:sleep_efficiency:91:%:above 7-day avg of 85%]]\n"
            "- Deep sleep: solid at 1h 18m\n"
            "- HRV and RHR both within baseline — no red flags\n\n"
            "Good options: 45-min run or strength. Protein + carbs before."
        )
        sanitized = sanitize_output(raw)
        assert "—" not in sanitized

        blocks = parse_content_blocks(sanitized)
        # Expected: verdict text, data card, closing text
        assert len(blocks) >= 2
        data_cards = [b for b in blocks if isinstance(b, DataCardBlock)]
        assert len(data_cards) == 1
        assert data_cards[0].metric == "sleep_efficiency"

        flat = flatten_to_markdown(sanitized)
        assert "**91%**" in flat
        assert "[[" not in flat
        assert "—" not in flat
