# EthicLens

An **AI Bias Detection & Mitigation Workbench** for regulated decision models — unifying fairness
detection, prescriptive (executable) mitigation, and formal governance sign-off.

## The three things that make it credible

1. **Verifiable numeric correctness** — every metric is implemented from scratch and
   cross-validated against [Fairlearn](https://fairlearn.org/) to 1e-9, with a golden-reference
   model whose Disparate Impact (≈ 0.55) is asserted in CI.
2. **Real, measured mitigation** — a held-out accuracy-vs-fairness Pareto frontier feeds a ranked
   recommender; the re-audit actually crosses the 0.80 threshold.
3. **Security & responsible-AI maturity** — sandboxed model deserialisation, bootstrap confidence
   intervals, Model Cards, Datasheets, and an honest [limitations](limitations.md) statement.

## Start here

- **[Methodology](methodology.md)** — metric definitions, the composite rationale, and the
  methodological fix over the original spec.
- **[STP traceability](traceability-matrix.md)** — every requirement mapped to code + tests.
- **[Limitations](limitations.md)** — what this does *not* prove.
- **[Architecture decisions](adr/0001-composite-weights.md)** — the choices and why.

The original (never-implemented) System Test Plan is preserved at
[`original-stp.pdf`](original-stp.pdf). A generated [sample scorecard](sample-scorecard.pdf),
[Model Card](sample-model-card.md), and [Datasheet](sample-datasheet.md) show the report outputs.

![Impossibility theorem](impossibility-theorem.png)
