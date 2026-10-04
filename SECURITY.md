# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Instead, use GitHub's
[private vulnerability reporting](https://github.com/vardhjain/Ethiclens/security/advisories/new)
for this repository. Include what you found, how to reproduce it, and the impact you expect.

You can expect an acknowledgement within a few days. Once a fix is released, the report is
credited in the changelog unless you prefer otherwise.

## Scope

- The API service (`services/api`), including authentication, model upload and the hosted agent.
- The web app (`apps/web`).
- The `fairness-core` engine and the CI/deployment configuration in this repository.

The hosted demo runs on free tiers and holds no sensitive data; please do not load-test it.

## What is already in place

- Uploaded model files are deserialised in an isolated sandbox, because loading an untrusted
  pickle file can execute arbitrary code (see
  [ADR 0002](docs/adr/0002-onnx-keystone-and-sandbox.md)).
- Login and registration are rate limited; responses carry standard security headers.
- CodeQL, `pip-audit` and `npm audit` run on every push and weekly.
- Uploaded CSVs are never stored — only the resulting audit record is.

Known boundaries are documented honestly in [LIMITATIONS.md](LIMITATIONS.md).
