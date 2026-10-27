#!/bin/sh
# Cloud Run entrypoint: run migrations against Supabase, then serve on $PORT.
# No worker branch here — the hosted deployment always runs EAGER_TASKS=true,
# so there is nothing to queue (see ethiclens_api.tasks).
set -e

# Hosted/public deployment cannot enable unsafe model-file execution, even if
# an old Cloud Run environment variable was accidentally set to true.
export ENABLE_MODEL_FILE_INGESTION=false

cd /app/services/api
alembic upgrade head
cd /app

exec uvicorn ethiclens_api.main:app --host 0.0.0.0 --port "${PORT:-8080}"
