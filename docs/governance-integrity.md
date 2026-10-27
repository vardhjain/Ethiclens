# Governance sign-off and audit integrity

EthicLens supports role-restricted sign-off on completed, persisted audit results.
Only a governance approver (or admin) **other than the audit owner** can sign.
Draft, queued, running and failed audits, as well as audits without persisted
metrics, cannot be signed.

Signing off marks the session `SIGNED_OFF` and `locked`, records the approver,
timestamp and note, and stores a SHA-256 fingerprint of a **canonical snapshot**
including the session's configuration, results, protected attributes and
mitigation logs. An atomic status check avoids duplicate sign-offs and helps
prevent concurrent audit workers from overwriting an approved scorecard.
Subsequent audit reruns and mitigations are rejected.

## Verify an approved audit

The session owner, governance approvers and administrators can request:

```http
GET /api/sessions/{session_id}/integrity
Authorization: Bearer <token>
```

The response includes `status`, `expected_digest` and `current_digest`.
Possible statuses:

- `verified`: the current data matches its sign-off fingerprint.
- `mismatch`: the approved data changed after signing.
- `not_signed`: no approved snapshot exists.
- `legacy_unverifiable`: signed before fingerprints were introduced.

The fingerprint also appears in the session response as `signoff_digest`.
Retain a copy **outside the EthicLens database** (for example in an independent
review system) to compare later.

## What this does *not* guarantee

A SHA-256 hash stored in the same database is **not** an immutable audit ledger,
a digital signature, or protection against a database administrator able to
alter both the record and its digest. The application-level lock prevents normal
API writes; it does not prevent direct SQL modifications. For stronger evidence,
export fingerprints to an independently controlled append-only system and
introduce signed events / separate key management. PostgreSQL row locks and
conditional writes defend the normal application state transitions, while
SQLite remains a development/test option.
