import re
import sqlite3
from pathlib import Path

from misterio.contracts import DocumentChunk


class LocalKb:
    def __init__(self, db_path: Path = Path("data/kb.sqlite3")) -> None:
        self.db_path = db_path

    def ingest(self, seed_dir: Path = Path("kb/seed")) -> int:
        files = sorted(
            path
            for path in seed_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chunks "
                "(source TEXT NOT NULL, locator TEXT NOT NULL, text TEXT NOT NULL)"
            )
            connection.execute("DELETE FROM chunks")
            connection.executemany(
                "INSERT INTO chunks (source, locator, text) VALUES (?, ?, ?)",
                [
                    (
                        path.as_posix(),
                        f"{path.as_posix()}#0",
                        path.read_text(encoding="utf-8"),
                    )
                    for path in files
                ],
            )
        return len(files)

    def search(
        self, query: str, brand_id: str | None = None, k: int = 5
    ) -> list[DocumentChunk]:
        del brand_id
        if k <= 0 or not self.db_path.exists():
            return []

        query_terms = set(_terms(query))
        if not query_terms:
            return []
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute("SELECT text, source, locator FROM chunks").fetchall()

        matches: list[DocumentChunk] = []
        for text, source, locator in rows:
            score = len(query_terms & set(_terms(text))) / len(query_terms)
            if score:
                matches.append(
                    DocumentChunk(
                        text=text, source=source, locator=locator, score=score
                    )
                )
        matches.sort(key=lambda chunk: (-chunk.score, chunk.source, chunk.locator))
        return matches[:k]


def _terms(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.casefold())
