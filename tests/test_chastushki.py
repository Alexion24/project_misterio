from misterio.chastushki import (
    format_issues,
    quality_issues,
    rhyme_score,
)

GOOD_RHYME = """По деревне трактор мчится,
Пыль клубится за трубой.
Петя думал прокатиться —
Еле выбрался домой.

Программист сидел всю ночь,
Всё ловил ошибку.
А она умчалась прочь —
Спряталась в прошивку!"""

BAD_RHYME = """По деревне трактор мчится,
Пыль клубится за трубой.
Петя думал прокатиться —
Еле выбрался из сада.

Программист сидел всю ночь,
Всё ловил ошибку.
А она умчалась прочь —
Спряталась под столом!"""


def test_format_issues_accepts_two_quatrains() -> None:
    assert format_issues(GOOD_RHYME) == []


def test_format_issues_allows_natural_lines_outside_word_count_guide() -> None:
    varied_lines = GOOD_RHYME.replace(
        "По деревне трактор мчится,",
        "По нашей деревне старый трактор быстро к станции мчится,",
    )

    assert format_issues(varied_lines) == []


def test_rhyme_score_prefers_matching_endings() -> None:
    assert rhyme_score(GOOD_RHYME) > rhyme_score(BAD_RHYME)


def test_rhyme_score_compares_from_stressed_vowel() -> None:
    strong = GOOD_RHYME.replace("за трубой.", "в рамке.").replace("домой.", "в замке.")
    weak = GOOD_RHYME.replace("за трубой.", "в песке.").replace("домой.", "в замке.")

    assert rhyme_score(strong) > rhyme_score(weak)


def test_rhyme_score_penalizes_syllable_count_difference() -> None:
    uneven = GOOD_RHYME.replace(
        "Еле выбрался домой.",
        "Еле-еле очень медленно выбирался он домой.",
    )

    assert rhyme_score(GOOD_RHYME) > rhyme_score(uneven)


def test_rhyme_score_penalizes_repeated_content_words() -> None:
    repeated = GOOD_RHYME.replace(
        "Еле выбрался домой.",
        "Пыль клубится домой.",
    )

    assert rhyme_score(GOOD_RHYME) > rhyme_score(repeated)


def test_rhyme_score_rejects_matching_grammatical_endings_only() -> None:
    strong = GOOD_RHYME.replace("за трубой.", "игривого.").replace(
        "домой.", "красивого."
    )
    suffix_only = GOOD_RHYME.replace("за трубой.", "весёлого.").replace(
        "домой.", "красивого."
    )

    assert rhyme_score(strong) > rhyme_score(suffix_only)


def test_rhyme_score_rejects_repeated_last_word() -> None:
    repeated = GOOD_RHYME.replace("домой.", "трубой.")

    assert rhyme_score(repeated) < rhyme_score(GOOD_RHYME)
    assert quality_issues(repeated) == [
        "строки 2 и 4 в каждой строфе должны рифмоваться"
    ]
