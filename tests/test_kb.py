from pathlib import Path

from misterio.kb import LocalKb


def test_ingest_then_search_returns_seed_citation(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    seed_file = seed_dir / "tractor.md"
    seed_file.write_text("Синий трактор работает на ферме.", encoding="utf-8")
    kb = LocalKb(tmp_path / "kb.sqlite3")

    assert kb.ingest(seed_dir) == 1
    results = kb.search("синий трактор", brand_id=None, k=5)

    assert len(results) == 1
    assert results[0].source == seed_file.as_posix()
    assert results[0].locator == f"{seed_file.as_posix()}#0"
    assert results[0].score == 1.0


def test_search_ignores_unrelated_chunks_and_honors_limit(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "a.md").write_text("трактор поле", encoding="utf-8")
    (seed_dir / "b.txt").write_text("трактор дорога", encoding="utf-8")
    (seed_dir / "c.md").write_text("корабль море", encoding="utf-8")
    kb = LocalKb(tmp_path / "kb.sqlite3")
    kb.ingest(seed_dir)

    results = kb.search("трактор", k=1)

    assert len(results) == 1
    assert results[0].source.endswith("a.md")
