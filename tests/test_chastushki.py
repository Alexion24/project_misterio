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


def test_rhyme_score_prefers_matching_endings() -> None:
    assert rhyme_score(GOOD_RHYME) > rhyme_score(BAD_RHYME)


def test_rhyme_score_rejects_repeated_last_word() -> None:
    repeated = GOOD_RHYME.replace("домой.", "трубой.")

    assert rhyme_score(repeated) < rhyme_score(GOOD_RHYME)
    assert quality_issues(repeated) == [
        "строки 2 и 4 в каждой строфе должны рифмоваться"
    ]
