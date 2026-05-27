# Contributing to Meta-Attention

Thank you for your interest in contributing. This repository contains the public **meta-attention** Python library.

## Development setup

```bash
git clone https://github.com/alanferrari/meta-attention.git
cd meta-attention
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

Optional coverage:

```bash
pytest --cov=meta_attention --cov-report=term-missing
```

## Code style

- Python ≥ 3.9; keep code compatible with the version range in `pyproject.toml`.
- Line length: 100 (see `[tool.ruff]` in `pyproject.toml`).
- Add or update tests for behavior changes.
- Preserve the Apache 2.0 license header on new source files (match existing files).

## Pull requests

1. Open an issue for large changes (new experts, API breaks, new integrations) so we can align on design.
2. Keep PRs focused; include a short description and test plan.
3. Update `CHANGELOG.md` under **Unreleased** (or the next version section) for user-visible changes.
4. Ensure CI passes (`pytest` on Python 3.9–3.12).

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Python and PyTorch versions
- Minimal code to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
