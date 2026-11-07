# Misterio agent instructions

## Workflow

- Make the smallest change that fully solves the request.
- Preserve existing behavior unless the request explicitly changes it.
- Do not modify unrelated files or reformat unrelated code.
- State material assumptions when requirements are ambiguous.

## Project guidance

- Python requires 3.12+ and dependencies are managed with `uv`.
- Use `docs/architecture.md` when changing component boundaries, contracts,
  skill loading, or LLM integration.
- Treat `skills/*/SKILL.md` and `skills/*/manifest.json` as one public skill
  definition; keep them consistent when changing a skill.
- Do not edit generated artifacts or local data under `data/`.

## Verification

- Run focused tests for the changed behavior first:
  `uv run pytest <relevant-test-file> -q`.
- Run `uv run ruff check .` after Python changes.
- Run the full suite with `uv run pytest -q` when a change can affect multiple
  components.
- Tests marked `model_eval` call a real configured LLM. Run them only when
  explicitly requested or when validating model-dependent behavior, and report
  that they require external model availability.
- Do not claim success when required checks were not run; state what remains
  unverified.

## Code changes

- Match existing project patterns before introducing an abstraction.
- Add or update a regression test for changed behavior when practical.
- Every changed line should be directly attributable to the request.
