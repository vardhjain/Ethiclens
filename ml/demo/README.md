---
title: EthicLens Bias Workbench
emoji: ⚖️
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

> Deploy: this Space = the contents of `ml/demo/` (upload `app.py`, `requirements.txt`, and this
> `README.md` to the Space root). `requirements.txt` installs `fairness-core` straight from GitHub.

# EthicLens — live demo

An interactive walkthrough of the [EthicLens](https://github.com/vardhjain/Ethiclens) bias workbench: train a biased
lending model, audit it for disparate impact with bootstrap confidence intervals, then apply a
**measured** mitigation (Fairlearn `ThresholdOptimizer`) and watch the re-audit cross the 0.80
four-fifths threshold.

It runs the same `fairness_core` engine — implemented from scratch and validated against Fairlearn
to 1e-9 — that powers the full FastAPI service.

## Deploying to Hugging Face Spaces

This `README.md` doubles as the Space card. Create a **Gradio** Space, push this repository, and HF
will install `ml/demo/requirements.txt` and launch `ml/demo/app.py`.

## Run locally

```bash
uv pip install -e "packages/fairness-core[validation,viz]" gradio
python -m ml.demo.app
```
