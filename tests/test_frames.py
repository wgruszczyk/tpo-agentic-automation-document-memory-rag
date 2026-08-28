from __future__ import annotations

from product_memory.ingestion.extractors import _interleave


def test_a_screen_is_placed_before_the_words_spoken_over_it() -> None:
    # The point of a screen is to show what someone was referring to, so it has to sit next to
    # what they said, not in a pile at the end of the transcript.
    transcript = "\n".join(
        [
            "[00:00:10] let me share my screen",
            "[00:05:00] as you can see on the right",
            "[00:09:30] any questions",
        ]
    )
    screens = [(295.0, "[Image text: screen at 00:04:55]\nPricing overview")]

    merged = _interleave(transcript, screens).splitlines()

    assert merged == [
        "[00:00:10] let me share my screen",
        "[Image text: screen at 00:04:55]",
        "Pricing overview",
        "[00:05:00] as you can see on the right",
        "[00:09:30] any questions",
    ]


def test_several_screens_keep_their_order() -> None:
    transcript = "\n".join(["[00:01:00] first", "[00:10:00] second"])
    screens = [
        (30.0, "[Image text: screen at 00:00:30]\nA"),
        (400.0, "[Image text: screen at 00:06:40]\nB"),
    ]

    merged = _interleave(transcript, screens).splitlines()

    assert merged[0] == "[Image text: screen at 00:00:30]"
    assert merged[2] == "[00:01:00] first"
    assert merged[3] == "[Image text: screen at 00:06:40]"
    assert merged[5] == "[00:10:00] second"


def test_a_screen_shown_after_the_last_word_is_still_kept() -> None:
    transcript = "[00:00:05] thanks everyone"
    screens = [(600.0, "[Image text: screen at 00:10:00]\nClosing slide")]

    merged = _interleave(transcript, screens).splitlines()

    assert merged[-2:] == ["[Image text: screen at 00:10:00]", "Closing slide"]


def test_a_transcript_with_no_screens_is_left_alone() -> None:
    transcript = "[00:00:05] nothing was shared"

    assert _interleave(transcript, []) == transcript
