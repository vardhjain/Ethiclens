# ADR 0005 — Pin third-party GitHub Actions to commit SHAs

- **Status:** Accepted
- **Date:** 2026-05

## Context
GitHub Actions referenced by a moving tag (`@v4`) resolve to whatever commit that tag currently
points at. Tags are mutable, so a compromised or hijacked action can inject a malicious step into
CI — which runs with repository secrets in scope. This is not hypothetical: tag-mutation supply-chain
incidents against popular actions have leaked CI secrets in the wild.

## Decision
For any **production-hardened** pipeline, pin every third-party action to a **full 40-character
commit SHA**, with the human-readable tag in a trailing comment, e.g.:

```yaml
- uses: actions/checkout@<40-char-sha> # v4.2.2
```

In this repository the workflows currently reference major tags for readability; the hardening step
(SHA pinning + Dependabot updates for the `github-actions` ecosystem) is tracked as a follow-up.
First-party `actions/*` and `github/codeql-action/*` are lower-risk but are pinned under the same
policy when hardening.

## Consequences
- Reproducible, tamper-resistant CI; an upstream tag move cannot silently change our pipeline.
- Slightly more maintenance (SHAs must be bumped deliberately) — delegated to Dependabot.
- Security scanning uses GitHub-native **CodeQL + pip-audit** rather than a third-party scanner, to
  minimise the trusted-action surface.
