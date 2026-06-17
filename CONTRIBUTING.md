# Contributing to EthicLens

Thanks for your interest! This project values **correctness over breadth** — every fairness
number must be reproducible and validated.

## Development setup

```bash
uv venv
uv pip install -e "packages/fairness-core[validation,viz,cli,reporting]" -e services/api --group dev
pre-commit install
```

## Before you push

```bash
make lint        # ruff check
make type        # mypy --strict on the engine
make cov         # pytest with the 85% coverage gate
make audit-golden  # the golden Disparate Impact must stay in band
```

CI runs the same checks on Python 3.11 and 3.12, plus the golden-audit regression gate and
CodeQL/pip-audit security scans.

## Ground rules

- **Never fabricate a metric.** If something cannot be computed (e.g. Equalized Odds without
  labels), return `INSUFFICIENT_DATA` — don't guess.
- **New metrics** must be implemented from scratch *and* cross-validated against Fairlearn in a
  test, with a Hypothesis property test for their invariants.
- **Conventional Commits** for messages (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Keep `fairness-core` free of web/database dependencies.
- Update `LIMITATIONS.md` if a change alters what the system can or cannot prove.

By contributing you agree your work is licensed under the project's [MIT License](LICENSE).
