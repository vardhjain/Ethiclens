# Deploying EthicLens (hosted agent demo)

Free-tier deployment: **Google Cloud Run** for the API, **Supabase** (Postgres)
for storage, **Vercel** for the frontend. Total recurring cost: $0 for
low-traffic use.

Cloud Run's free tier is generous enough for this: 180,000 vCPU-seconds +
360,000 GiB-seconds + 2,000,000 requests per month, and it scales to zero
between requests (no idle cost — a cold start is expected and fine for a
portfolio demo). Originally this targeted Hugging Face Spaces (Docker), but as
of 2026-07-08 HF started gating the Docker SDK behind a paid plan for new
Spaces with no prior notice — Cloud Run is the fallback.

This is a demonstration deployment, not production infrastructure — see
LIMITATIONS.md. No uptime guarantee.

## 1. Supabase (Postgres)

1. Create a project at supabase.com (free tier).
2. Project Settings → Database → Connection string → **URI**, "Session pooler"
   mode (works better than the direct connection from a serverless/container
   client). It looks like:
   `postgresql://postgres.xxxx:PASSWORD@aws-0-region.pooler.supabase.com:5432/postgres`
3. Convert it to the async driver EthicLens expects — replace
   `postgresql://` with `postgresql+asyncpg://`. Save this as `DATABASE_URL`;
   you'll paste it into both the Cloud Run service (step 2) and a GitHub
   Actions secret (step 4).
4. Nothing else to create manually — `alembic upgrade head` runs automatically
   on container start (see `infra/docker/entrypoint-cloudrun.sh`) and creates
   every table from `services/api/src/ethiclens_api/models.py`.
5. Row-Level Security: migration `0002_enable_rls` enables RLS (with no
   policies) on every table and revokes the `anon`/`authenticated` grants, so
   Supabase's auto-generated REST API
   (`https://<project>.supabase.co/rest/v1/`) can't read or write anything.
   EthicLens never uses that API — it connects directly over `DATABASE_URL`
   as the table owner, which RLS doesn't restrict. If Supabase's security
   advisor flagged `rls_disabled_in_public` / `sensitive_columns_exposed` on
   an existing deployment, redeploying (which reruns `alembic upgrade head`)
   clears both findings.

Handle the password carefully: never paste it into a chat, issue tracker, or
anywhere outside your password manager / the actual secret store (Supabase,
`gcloud`, GitHub Actions secrets). If a password was ever pasted somewhere
insecure, rotate it (Project Settings → Database → reset password) before
using it. If the password contains URI-reserved characters
(`: / ? # [ ] @ ! $ & ' ( ) * + , ; =`), URL-encode just the password segment
before building the connection string.

## 2. Google Cloud Run

Prereqs: a Google Cloud account (needs a card on file for identity
verification — this does not mean you'll be charged; Cloud Run simply won't
bill anything within the free-tier limits above) and the `gcloud` CLI
installed and authenticated (`gcloud init`).

1. Create/select a GCP project, then enable the required APIs:
   ```
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```
2. From the repo root, deploy directly from source — Cloud Build reads the
   root `Dockerfile` (adapted from `infra/docker/Dockerfile.api`; that one
   stays for local `docker-compose`, which still runs the enterprise arq
   worker for the non-agent `/sessions` flow):
   ```
   gcloud run deploy ethiclens-api \
     --source . \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars EAGER_TASKS=true \
     --set-env-vars DATABASE_URL="<postgresql+asyncpg://... from step 1>" \
     --set-env-vars SECRET_KEY="<any long random string>" \
     --set-env-vars GROQ_API_KEY="<from console.groq.com>"
   ```
   Add `--set-env-vars GEMINI_API_KEY="..."` too if you have one (optional
   fallback provider). For anything beyond a personal demo, prefer
   `--set-secrets` backed by Secret Manager over `--set-env-vars` for
   `DATABASE_URL`/`SECRET_KEY`/`*_API_KEY` — plain env vars are visible to
   anyone with read access to the Cloud Run service config.
3. The command prints a **Service URL** when it finishes
   (`https://ethiclens-api-xxxxxxxxxx-uc.a.run.app`). Confirm
   `<service-url>/health` returns `{"status":"ok",...}` and `<service-url>/docs`
   loads the FastAPI docs.

### Redeploying after code changes

Re-run the same `gcloud run deploy` command (or just `gcloud run deploy
ethiclens-api --source .` if the env vars are already set — Cloud Run keeps
existing env vars across deploys unless you explicitly change them). No git
remote or manual sync step needed; it builds straight from your local working
tree via Cloud Build.

### Automatic redeploys via GitHub Actions (optional)

`.github/workflows/deploy-api.yml` runs the same `gcloud run deploy
ethiclens-api --source .` on every push to `main` that touches the API's
source (path-filtered — frontend-only or docs-only commits don't trigger it).
Same no-env-vars approach as the manual command above: it only needs
permission to *deploy*, never your `DATABASE_URL`/`SECRET_KEY`/`*_API_KEY`
values, which stay wherever you already set them on the Cloud Run service.

It authenticates via **Workload Identity Federation** — no long-lived GCP key
stored in GitHub. One-time setup (run once, locally, with `gcloud` logged in
as a project owner):

```bash
PROJECT_ID="<your GCP project ID>"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
REPO="<your-github-username>/<repo-name>"   # e.g. octocat/ethiclens

# 1. A dedicated service account the workflow will deploy as.
gcloud iam service-accounts create github-deployer \
  --project "$PROJECT_ID" --display-name "GitHub Actions Cloud Run deployer"

# 2. Just enough to build (Cloud Build) and deploy (Cloud Run) from source.
for ROLE in roles/run.admin roles/iam.serviceAccountUser roles/cloudbuild.builds.editor roles/storage.admin; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:github-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="$ROLE"
done

# 3. A Workload Identity Pool + OIDC provider trusting GitHub's own token issuer.
gcloud iam workload-identity-pools create github-pool \
  --project "$PROJECT_ID" --location global --display-name "GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project "$PROJECT_ID" --location global --workload-identity-pool github-pool \
  --issuer-uri "https://token.actions.githubusercontent.com" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition "assertion.repository=='${REPO}'"

# 4. Let ONLY this specific repo impersonate the service account (no key ever
#    leaves Google — GitHub's OIDC token is exchanged for short-lived creds).
gcloud iam service-accounts add-iam-policy-binding \
  "github-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project "$PROJECT_ID" --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${REPO}"
```

Then, repo → Settings → Secrets and variables → Actions → **Variables** tab,
add:
- `GCP_PROJECT_ID` = `$PROJECT_ID`
- `GCP_WORKLOAD_IDENTITY_PROVIDER` =
  `projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider`
- `GCP_SERVICE_ACCOUNT` = `github-deployer@$PROJECT_ID.iam.gserviceaccount.com`
- `GCP_REGION` (optional — defaults to `us-central1` if unset)

The workflow no-ops (skips every step, exits green) until all three required
variables are set, so merging it before doing this setup is safe — same
pattern as `keep-alive.yml`.

## 3. Vercel (frontend)

1. Import the GitHub repo into Vercel. Root directory: `apps/web`. Framework
   preset: Vite (auto-detected). Build command / output dir: defaults are
   correct (`npm run build`, `dist`).
2. Edit `apps/web/vercel.json` — replace
   `REPLACE-WITH-YOUR-CLOUD-RUN-URL` with your actual Cloud Run service URL
   from step 2 (no trailing slash). This makes `/api/*` requests from the
   browser transparently proxy to Cloud Run, so the existing frontend code
   (`src/api/client.ts`'s relative `/api${path}` fetches) needs **zero
   changes** — same-origin from the browser's point of view, no CORS
   configuration needed.
3. Deploy. Vercel gives you a `*.vercel.app` URL.

## 4. GitHub Actions keep-alive

`.github/workflows/keep-alive.yml` pings the Cloud Run service's `/health`
(mostly just a smoke test / pre-warm — Cloud Run scaling to zero between
requests is expected behavior, not a problem to work around) and runs
`SELECT 1` against Supabase daily, since **Supabase does pause after ~1 week
idle** and that's the one thing actually worth guarding against here.

Repo → Settings → Secrets and variables → Actions:
- Add **variable** `API_URL` = your Cloud Run service URL (not secret — it's
  public anyway).
- Add **secret** `SUPABASE_DATABASE_URL` = the same connection string from
  step 1 (plain `postgresql://`, not `+asyncpg` — `psql` doesn't understand
  the asyncpg driver suffix).

The workflow no-ops on either step until its corresponding secret/variable is
set, so merging it before you've done steps 1-3 is safe.

## 5. Verify end-to-end

1. Visit the Vercel URL, register an account, log in.
2. "Agent audit" → upload a predictions CSV → confirm the proposed schema →
   run. You should get a real narrative (if `GROQ_API_KEY`/`GEMINI_API_KEY`
   are set and valid) or the numeric-only degraded fallback (if not, or if the
   daily LLM budget cap is hit — see `agent_daily_llm_call_cap` in
   `config.py`).
3. Ask a question in the chat panel; confirm it's grounded in the scorecard.

## What's intentionally not enabled here

- **Per-IP rate limiting on the agent audit endpoints** (propose-schema/
  run-audit/ask) — the daily LLM call budget (global + per-user caps, see
  `agent_daily_llm_call_cap*` in `config.py`) is the guard there instead.
  Login/register *are* IP-rate-limited (slowapi — see `rate_limit.py`), since
  those are open, unauthenticated endpoints on a public demo.
- **Model file uploads** — the agent pipeline only ever accepted predictions
  CSVs, never model files, so there's nothing to disable here; it was
  designed this way from the start (see agent/csv_ingest.py).
- **arq/Redis worker** — not deployed at all; `EAGER_TASKS=true` makes it
  unnecessary (see `ethiclens_api/tasks.py`).
