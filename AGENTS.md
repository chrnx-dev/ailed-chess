# AILED Chess

- Single-package Python library. Install with `pip install -e ".[dev]"`.
- Requires Python `>=3.11`.

## Commands

- Run the full test suite: `pytest -q`
- Run one test file: `pytest tests/unit/test_move_selector.py -q`
- Build the package: `python -m build`

## Source Of Truth

- Trust `pyproject.toml` and GitHub Actions over README prose.
- The real import/package name is `ailed_chess`. Source and tests use `from ailed_chess...`.
- There is no repo-local lint, formatter, typecheck, tox, nox, make, or task runner config. Do not invent commands that are not defined here.

## Do / Don't

- Do verify commands, imports, and workflows in `pyproject.toml`, tests, and `.github/workflows/` before editing docs or instructions.
- Don't trust README examples if they conflict with executable sources.
- Do start work from `main` in a feature branch such as `feat/`, `fix/`, or `docs/`; don't commit directly on `main`.
- Do run relevant tests before every commit or push; for broad code changes, prefer `pytest -q`.
- Do use subagents in parallel when work is truly independent; don't split parallel work across the same files.
- Do use brainstorming as required planning before non-trivial changes.
- Do version releases with CalVer `YYYY.MM.PATCH`; don't bump versions before the release that ships them.
- Do create a git tag for each release; publishing is driven by `v*` tags.
- Don't merge into `main`, `develop`, or `release/*` without a pull request.

## Layout

- `src/ailed_chess/psyche/`: board evaluation, phase detection, psyche update logic.
- `src/ailed_chess/uci/`: EQ curves and move-probability reshaping/sampling.
- `src/ailed_chess/__init__.py`: public API re-exports.
- `tests/unit/`: main verification surface.

## Change Guidance

- `PsycheConfig` and `SelectionConfig` are frozen Pydantic models; keep config changes immutable.
- Critical behavior lives in `psyche/calculator.py`, `psyche/config.py`, `uci/move_selector.py`, and `uci/eq_curve.py`; pair edits there with targeted tests.
- Keep dependencies minimal. `CONTRIBUTING.md` explicitly says not to add new core dependencies without discussion.

## CI / Release

- CI runs `pip install -e ".[dev]"` then `pytest -v` on Python 3.11, 3.12, and 3.13.
- Publishing is tag-driven (`v*`) and runs tests before `python -m build` and PyPI publish.
