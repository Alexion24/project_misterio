import re

_VOWELS = "аеёиоуыэюя"
_UNSTRESSED_ENDINGS = (
    "ами",
    "ями",
    "ого",
    "его",
    "ому",
    "ему",
    "ою",
    "ею",
    "ая",
    "яя",
    "ое",
    "ее",
    "ую",
    "юю",
    "ые",
    "ие",
    "ка",
    "ку",
    "ки",
    "ке",
    "кой",
    "ком",
)


def format_issues(text: str) -> list[str]:
    stanzas = _stanzas(text)
    lines = [line.strip() for stanza in stanzas for line in stanza.splitlines()]
    issues: list[str] = []
    if len(stanzas) != 2:
        issues.append(f"нужно 2 строфы, получено {len(stanzas)}")
    if any(len(stanza.splitlines()) != 4 for stanza in stanzas):
        issues.append("в каждой строфе должно быть 4 строки")
    normalized = [re.sub(r"\W", "", line.casefold()) for line in lines]
    if len(normalized) != len(set(normalized)):
        issues.append("строки не должны повторяться")
    return issues


def rhyme_score(text: str) -> float:
    stanzas = _stanzas(text)
    if len(stanzas) != 2 or any(len(stanza.splitlines()) != 4 for stanza in stanzas):
        return 0.0
    scores = []
    for stanza in stanzas:
        lines = stanza.splitlines()
        left = _last_word(lines[1])
        right = _last_word(lines[3])
        if not left or not right or left == right:
            scores.append(0.0)
            continue
        tail_score = _tail_similarity(_rhyme_tail(left), _rhyme_tail(right))
        scores.append(
            tail_score
            * _rhythm_similarity(lines[1], lines[3])
            * _repetition_penalty(lines[1], lines[3])
        )
    return sum(scores) / len(scores)


def quality_issues(text: str) -> list[str]:
    issues = format_issues(text)
    if not issues and rhyme_score(text) <= 0.375:
        issues.append("строки 2 и 4 в каждой строфе должны рифмоваться")
    return issues


def _stanzas(text: str) -> list[str]:
    return [part for part in re.split(r"\n\s*\n", text.strip()) if part]


def _last_word(line: str) -> str:
    words = re.findall(r"[а-яё]+", line.casefold())
    return words[-1] if words else ""


def _rhyme_tail(word: str) -> str:
    stress = word.find("ё")
    if stress < 0:
        stem_end = len(word)
        for ending in _UNSTRESSED_ENDINGS:
            if word.endswith(ending) and len(word) > len(ending):
                stem_end -= len(ending)
                break
        stress = max(
            (index for index, char in enumerate(word[:stem_end]) if char in _VOWELS),
            default=-1,
        )
        if stress < 0:
            stress = max(
                (index for index, char in enumerate(word) if char in _VOWELS),
                default=0,
            )
    return word[stress:].replace("ё", "е")


def _tail_similarity(left: str, right: str) -> float:
    length = max(len(left), len(right))
    if not length or left[0] != right[0]:
        return 0.0
    matches = sum(a == b for a, b in zip(left, right, strict=False))
    return matches / length


def _rhythm_similarity(left: str, right: str) -> float:
    difference = abs(_syllable_count(left) - _syllable_count(right))
    return max(0.0, 1 - difference / 6)


def _syllable_count(line: str) -> int:
    return sum(char in _VOWELS for char in line.casefold())


def _repetition_penalty(left: str, right: str) -> float:
    left_words = set(re.findall(r"[а-яё]{4,}", left.casefold()))
    right_words = set(re.findall(r"[а-яё]{4,}", right.casefold()))
    return 0.75 if len(left_words & right_words) >= 2 else 1.0
