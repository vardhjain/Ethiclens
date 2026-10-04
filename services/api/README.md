# ethiclens-api

The EthicLens backend: a FastAPI service that handles sign-in, model uploads, bias audits,
mitigation, PDF reports, governance sign-off, and the hosted agent. All fairness maths is
delegated to the shared [`fairness-core`](../../packages/fairness-core) package — the API never
re-implements a metric.

## Run it locally

No database setup is needed: with no `DATABASE_URL` set, the service uses a local SQLite file.

```bash
uv pip install -e "packages/fairness-core[reporting,validation]" -e "services/api[test]"
uvicorn ethiclens_api.main:app --reload     # interactive docs at http://localhost:8000/docs
```

For PostgreSQL, set `DATABASE_URL` (see [`.env.example`](../../.env.example)) and apply the
migrations from this directory with `alembic upgrade head`.

## Endpoints

| Area | Routes | What it does |
|---|---|---|
| Auth | `POST /api/auth/register`, `/login`, `GET /me` | Accounts and JWT sign-in (rate limited) |
| Models | `POST /api/models/upload`, `GET /api/models` | Upload a model; it is loaded in a sandbox |
| Audits | `POST /api/sessions/create`, `/{id}/run`, `GET /{id}/status`, `/{id}/metrics` | Run a bias audit and read the results |
| Mitigation | `GET /api/sessions/{id}/recommendations`, `POST /{id}/mitigate` | Ranked fixes, then a re-audit as a child session |
| Reports | `GET /api/sessions/{id}/report` | Fairness Scorecard PDF |
| Governance | `POST /api/sessions/{id}/sign-off` | Role-restricted sign-off that locks the session |
| Agent | `POST /api/agent/propose-schema`, `/run-audit`, `/demo-audit`, `/records/{id}/ask` | CSV in, narrated audit and grounded Q&A out |

## Configuration

Settings are read from environment variables (or a `.env` file) — see
[`config.py`](src/ethiclens_api/config.py). The ones that matter most:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Signs login tokens. Use a long random value (32+ characters) in production. |
| `DATABASE_URL` | `postgresql+asyncpg://…` in production; defaults to local SQLite. |
| `GROQ_API_KEY` / `GEMINI_API_KEY` | Optional. Without them the agent returns numbers only, no narrative. |
| `EAGER_TASKS` | `true` runs audits in-process; `false` hands them to the arq/Redis worker. |

## Tests

```bash
pytest services/api/tests
```

Tests run against a throwaway SQLite database and never call a real LLM.
