# Google Cloud Run image for the EthicLens API + agent.
#
# `gcloud run deploy --source .` auto-detects this root Dockerfile and builds
# it via Cloud Build. Cloud Run injects $PORT (defaults to 8080) and routes
# traffic to whatever port the container listens on — no EXPOSE/HEALTHCHECK
# port matters to Cloud Run itself (it uses its own HTTP startup probe against
# the service URL), the values below are just documentation.
#
# This is a copy of infra/docker/Dockerfile.api adapted for that constraint:
# no worker mode — the hosted deployment runs audits in-process via
# EAGER_TASKS=true, so there is no separate worker/Redis to build or run (see
# ethiclens_api.tasks). infra/docker/Dockerfile.api remains the source of
# truth for local docker-compose (which still runs the enterprise arq worker);
# keep both in sync if the build steps change.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml ./
COPY packages ./packages
COPY services ./services

RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
    -e packages/fairness-core -e "services/api[agent]"

# --- runtime ---------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY packages ./packages
COPY services ./services
COPY ml ./ml
COPY models ./models
COPY infra/docker/entrypoint-cloudrun.sh /entrypoint-cloudrun.sh
RUN chmod +x /entrypoint-cloudrun.sh && \
    adduser --disabled-password --gecos "" --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
ENTRYPOINT ["/entrypoint-cloudrun.sh"]
