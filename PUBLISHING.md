# Publishing this repository on GitHub

This directory is a **ready-to-publish** snapshot of the **meta-attention** library. Copy or push its contents to a new public repository (for example `alanferrari/meta-attention`).

## 1. Create the GitHub repository

1. On GitHub: **New repository** → name `meta-attention` (or your choice).
2. Do **not** initialize with a README if you are pushing this folder as-is (avoid merge conflicts).

## 2. Push the code

From this folder (`github/`):

```bash
git init
git add .
git commit -m "Initial public release v0.2.0"
git branch -M main
git remote add origin git@github.com:YOUR_USER/meta-attention.git
git push -u origin main
```

## 3. Optional: attach the paper PDF

The README references a companion paper. Add it before or after the first push:

```text
paper/meta-attention.pdf
```

See [`paper/README.md`](paper/README.md). If you host the PDF elsewhere (arXiv, project site), update the link in `README.md` and `docs/architecture.md`.

## 4. GitHub repository settings (recommended)

- **About**: short description, link to docs, tags `transformers`, `attention`, `pytorch`
- **Topics**: `meta-attention`, `transformers`, `efficient-inference`, `pytorch`
- **Releases**: tag `v0.2.0`, attach wheels/sdist if publishing to PyPI
- Enable **Issues** and **Discussions** (optional)

## 5. PyPI (optional)

```bash
pip install build twine
python -m build
twine upload dist/*
```

Ensure `meta_attention/_version.py` matches the git tag and `CHANGELOG.md`.

## 6. CI

GitHub Actions workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs `pytest` on push and pull requests.

## What is intentionally excluded

This export omits internal research artifacts (`old/`, `paper_validation/`, training runs). Only the installable library, tests, examples, and user documentation are included.
