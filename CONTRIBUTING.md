# Contributing to Genesis Playground

This document outlines the minimal steps to get started.

---

## Principles and Socpe

- **Typing first**: Prefer explicit types (std `typing`, `jaxtyping`, `pydantic` models for configs & data schemas). Type errors are design feedback.
- **Tensor‑native**:  Training Environments and algorithms operate on PyTorch tensors only (no numpy in hot paths). Convert at boundaries.
- **Modular & composable**: Small, single‑purpose modules with clear interfaces. Favor composition over inheritance.

## Tooling & Baselines

- **Package/Env**: `uv` for lock/venv & scripts. 
- **Lint/Format**: `ruff` with autofix + format (project config). No black/isort separately.
- **Typecheck**: `pyright` strict for `src/*`
- **Pre‑commit**: hooks for `ruff`, pyright (optional), buf (if protobuf), uv‑lock, yamlfmt.
- **CI**: Automatic flow for continuous integrations

## 1. Getting Started 

- Fork the repository and clone your fork.
- Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
# create virtual environments
uv venv
# sync environments
uv sync --package gs-env
# 
pre-commit install
```

## 2. Development Flow

- Create a feature branch:

```bash
git checkout -b feat/my-feature
```

- Run checks locally before committing:

```
pre-commit run --all-files
```

## 3. Pull Request

- Keep PRs focused and under ~500 LOC when possible.
- Use conventional commits for commit messages:
  - [Feat]: add PPO config validation
  - [Fix]: correct reward tensor shape
  - [Docs]: update env README
- Ensure:
  - [ ] Lint, typecheck, and tests pass
  - [ ] New code has tests/docs
  - [ ] Examples run successfully


> **Note:**
> AI-generated code should **NOT** be committed directly into Pull Requests.  
> Please review, refactor, and ensure it follows our coding guidelines and project style before submitting.

## 4. Acknowledgement

Thank you for your interest in contributing! 🎉  