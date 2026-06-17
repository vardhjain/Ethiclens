# ADR 0002 — ONNX inference keystone and the untrusted-model sandbox

- **Status:** Accepted
- **Date:** 2026-05

## Context
FR-001 lets users upload trained models as `.h5` (TensorFlow), `.pt` (PyTorch) or `.pkl`
(scikit-learn). Two problems:

1. **Portability.** Running three native ML stacks in the request path is heavy and brittle, and
   makes the live demo image enormous.
2. **Security (critical).** Deserialising any of these formats executes arbitrary code embedded in
   the file (`pickle.load`, `torch.load`, Keras `Lambda` layers). The original spec waved this
   away; it is a textbook **remote-code-execution** vector.

These are **two different concerns** and must not be conflated. Converting to ONNX does **not**
remove the RCE, because you must deserialise the original artefact *first* in order to convert it.

## Decision
- **Portability — ONNX keystone.** On ingestion, convert every accepted model to ONNX and run a
  single `onnxruntime` inference path. The demo worker then ships only `onnxruntime`, not
  torch + tensorflow.
- **Security — a real sandbox, stated honestly.**
  - *Prefer safe formats:* `safetensors`, `skops`, or ONNX-direct upload. Pickle is a **discouraged
    fallback**.
  - *Contain the unavoidable:* deserialise untrusted artefacts inside a separate container with
    `--network none`, a read-only root filesystem, dropped capabilities, CPU/memory/PID limits and
    a hard timeout. Use `torch.load(weights_only=True)` where applicable; reject Keras `Lambda`
    layers; validate size and magic bytes before loading.
  - *Threat model is documented* in `LIMITATIONS.md`: this is **defense-in-depth, not a guarantee**
    against a sandbox-escape.

## Consequences
- "Framework-agnostic" becomes real (one inference path) without three brittle code branches.
- The security posture is honest and demonstrable — a stronger interview signal than an
  unqualified "we safely run untrusted models."
- Conversion has coverage limits (exotic custom layers may not export); these surface as explicit
  ingestion errors rather than silent failures.
