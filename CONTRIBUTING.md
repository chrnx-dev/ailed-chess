# Contributing to AILED

Thank you for your interest in contributing. This document covers how to get started, what kinds of contributions are welcome, and the project's policy on AI-generated code.

---

## Getting Started

```bash
# Fork the repo, then clone your fork
git clone https://github.com/<your-username>/ailed-chess
cd ailed-chess

# Install in development mode
pip install -e ".[dev]"

# Run the test suite
pytest
```

All tests must pass before submitting a PR.

---

## What Contributions Are Welcome

- **Bug fixes** — incorrect psyche calculations, phase detection edge cases, EQ signal chain behavior
- **New EQ personalities** — add a new curve preset to `uci/eq_curve.py`
- **New psyche factors** — extend `PsycheCalculator` with additional positional signals
- **Documentation** — clarifications, examples, corrections in README or docstrings
- **Tests** — additional coverage for existing behavior
- **Integration examples** — showing how to plug the psyche layer into Maia2, Stockfish, or other engines

If you are unsure whether your contribution fits, open an issue first to discuss.

---

## AI-Generated Code Policy

AI-assisted contributions are welcome. However:

1. **You are responsible** for understanding, testing, and validating any code you submit — regardless of how it was written.
2. **Disclose AI assistance** in your PR description (a single line is enough, e.g., _"Written with Claude / Copilot assistance"_).
3. **Critical paths may require human validation.** Changes to the following areas may trigger a request from the maintainer for additional human review before merge:
   - Psyche math (`psyche/calculator.py`, `psyche/config.py`)
   - Game phase detection
   - EQ signal chain (`uci/move_selector.py`, `uci/eq_curve.py`)
   - UCI protocol handling (`uci/engine.py`)
   - Model training logic (`models/`, `scripts/train.py`)

This is not a rejection of AI tooling — it is a quality gate for correctness in components where subtle bugs can silently affect engine behavior.

---

## Commit Style

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short summary

Body explaining what and why (not how).
```

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

Examples:
```
feat(psyche): add king tropism factor to positional evaluation
fix(calculator): clamp material_lost to prevent negative values on promotion
docs: add game phase detection table to README
```

---

## Pull Request Process

1. Branch from `main` with a descriptive name (`feat/`, `fix/`, `docs/` prefix).
2. Keep PRs focused — one logical change per PR.
3. Ensure `pytest` passes with no failures.
4. Fill out the PR template fully, including the AI disclosure checkbox.
5. A maintainer will review and may request changes or additional validation for critical paths.

---

## Code Style

- Python 3.11+
- Type hints on all public functions
- Frozen Pydantic models for all configuration
- No new dependencies without discussion — the core library intentionally keeps its dependency footprint small (`chess`, `pydantic`, `torch`)
