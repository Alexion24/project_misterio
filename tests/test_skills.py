from misterio.skills import list_skills


def test_list_skills_finds_chastushki() -> None:
    skills = list_skills()
    assert len(skills) == 1
    assert skills[0].id == "chastushki"
    assert skills[0].version == "0.1.0"
    assert skills[0].needs_retrieval is False
