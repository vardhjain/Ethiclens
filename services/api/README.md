# ethiclens-api

FastAPI service for EthicLens: authentication, model upload (sandboxed ingestion → ONNX),
audit sessions with async processing, mitigation apply / re-audit, Fairness Scorecard PDF, and the
governance sign-off workflow. Persists to PostgreSQL via async SQLAlchemy 2.0.

```bash
uvicorn ethiclens_api.main:app --reload    # docs at http://localhost:8000/docs
```

All fairness computation is delegated to the shared `fairness-core` package — the API never
re-implements a metric.
