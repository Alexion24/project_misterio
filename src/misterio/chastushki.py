import re


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
        common = 0
        for left_char, right_char in zip(reversed(left), reversed(right), strict=False):
            if left_char != right_char:
                break
            common += 1
        scores.append(min(common, 4) / 4)
    return sum(scores) / len(scores)


def quality_issues(text: str) -> list[str]:
    issues = format_issues(text)
    if not issues and rhyme_score(text) < 0.375:
        issues.append("строки 2 и 4 в каждой строфе должны рифмоваться")
    return issues


def _stanzas(text: str) -> list[str]:
    return [part for part in re.split(r"\n\s*\n", text.strip()) if part]


def _last_word(line: str) -> str:
    words = re.findall(r"[а-яё]+", line.casefold())
    return words[-1].replace("ё", "е") if words else ""
