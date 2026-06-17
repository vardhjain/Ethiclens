#!/bin/sh
# Run database migrations, then start the API (or the arq worker via $APP_CMD).
set -e

cd /app/services/api
alembic upgrade head
cd /app

if [ "$1" = "worker" ]; then
    exec arq ethiclens_api.tasks.WorkerSettings
fi

exec uvicorn ethiclens_api.main:app --host 0.0.0.0 --port 8000
