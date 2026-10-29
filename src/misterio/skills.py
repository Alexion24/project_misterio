import json
from pathlib import Path

from misterio.contracts import SkillManifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKILLS_DIR = _REPO_ROOT / "skills"


def list_skills(skills_dir: Path | None = None) -> list[SkillManifest]:
    root = skills_dir or DEFAULT_SKILLS_DIR
    manifests: list[SkillManifest] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        path = child / "manifest.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        manifests.append(SkillManifest.model_validate(data))
    return manifests
