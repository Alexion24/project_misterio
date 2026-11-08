from misterio.skills import list_skills


def test_list_skills_finds_chastushki() -> None:
    skills = list_skills()
    assert len(skills) == 1
    assert skills[0].id == "chastushki"
    assert skills[0].version == "0.9.0"
    assert skills[0].needs_retrieval is False
    assert skills[0].temperature == 0.8
    assert skills[0].top_p == 0.8
    assert skills[0].max_tokens == 350
    assert skills[0].candidate_count == 2
    assert skills[0].quality_policy == "chastushki"
