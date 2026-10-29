import json
from pathlib import Path

from misterio.contracts import SkillManifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKILLS_DIR = _REPO_ROOT / "skills"


def _iter_skills(skills_dir: Path | None = None) -> list[tuple[SkillManifest, Path]]:
    root = skills_dir or DEFAULT_SKILLS_DIR
    packs: list[tuple[SkillManifest, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        path = child / "manifest.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        packs.append((SkillManifest.model_validate(data), child))
    return packs


def list_skills(skills_dir: Path | None = None) -> list[SkillManifest]:
    return [manifest for manifest, _ in _iter_skills(skills_dir)]


def load_skill(
    skill_id: str, skills_dir: Path | None = None
) -> tuple[SkillManifest, Path]:
    for manifest, folder in _iter_skills(skills_dir):
        if manifest.id == skill_id:
            return manifest, folder
    raise KeyError(skill_id)
