"""EthicLens live demo + bring-your-own-data auditor (Gradio / Hugging Face Spaces).

Two tabs:

* **Demo** — train a biased model on a synthetic lending dataset, audit it, and apply a
  *measured* mitigation that crosses the 0.80 threshold.
* **Audit your own data** — upload a CSV of your model's predictions, true labels and a
  protected attribute, and get a full fairness scorecard (DI, SPD, Equalized Odds with
  bootstrap CIs) plus a downloadable PDF. We audit your **outputs** and never execute your
  model — safe by design.

Uses the same audited ``fairness_core`` engine as the API and CLI.

Run locally:  python -m ml.demo.app
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fairness_core import mitigate_and_reaudit, run_audit
from fairness_core.audit import AttributeSpec
from fairness_core.cli import render_scorecard
from fairness_core.datasets import make_biased_lending_dataset
from fairness_core.metrics.group import confusion_rates
from fairness_core.types import AuditResult, MetricName

FEATURES = ["income", "credit_score", "debt_ratio"]
_NONE = "(none)"
_TMP = Path(tempfile.gettempdir())


# --------------------------------------------------------------------------- #
# Built-in demo (synthetic biased lending model)
# --------------------------------------------------------------------------- #


def _train(seed: int):
    from sklearn.linear_model import LogisticRegression

    df, target = make_biased_lending_dataset(n=6000, seed=seed)
    model = LogisticRegression(max_iter=2000, random_state=seed).fit(df[FEATURES], df[target])
    return model, df, target


def run_demo(attribute: str, seed: int) -> tuple[pd.DataFrame, Any, str]:
    model, df, target = _train(int(seed))
    result = run_audit(
        model,
        df,
        [AttributeSpec(attribute)],
        target=target,
        feature_columns=FEATURES,
        n_boot=400,
        seed=int(seed),
    )
    return _table(result), _bar(result), _verdict(result)


def run_mitigation(attribute: str, seed: int) -> str:
    model, df, target = _train(int(seed))
    audit = run_audit(
        model,
        df,
        [AttributeSpec(attribute)],
        target=target,
        feature_columns=FEATURES,
        n_boot=200,
        seed=int(seed),
    )
    if not audit.flagged_groups:
        return "No flagged groups to mitigate."
    g = audit.flagged_groups[0]
    res = mitigate_and_reaudit(
        model,
        df,
        g.attribute,
        g.privileged_value,
        g.unprivileged_value,
        target=target,
        feature_columns=FEATURES,
        strategy="threshold_optimizer",
        n_boot=200,
        seed=int(seed),
    )
    verdict = "✅ DI now ≥ 0.80" if res.crossed_threshold else "still below 0.80"
    ci = res.di_after_ci
    return (
        f"### Applied ThresholdOptimizer to `{res.group_label}` (measured on held-out data)\n"
        f"- Disparate Impact: **{res.di_before:.3f} → {res.di_after:.3f}** "
        f"(95% CI [{ci.low:.2f}, {ci.high:.2f}])\n"
        f"- Accuracy: {res.accuracy_before:.3f} → {res.accuracy_after:.3f}\n"
        f"- **{verdict}**"
    )


# --------------------------------------------------------------------------- #
# Bring-your-own-data: audit an uploaded predictions CSV
# --------------------------------------------------------------------------- #


def _to_binary(series: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(series, errors="coerce")
    values = set(pd.unique(raw.dropna()))
    if values <= {0, 1}:
        return raw.fillna(0).astype(int).to_numpy()
    return (raw.fillna(0) >= 0.5).astype(int).to_numpy()  # treat as scores


def audit_csv(
    file: str | None, pred_col: str, label_col: str, attr_col: str, privileged: str
) -> tuple[str, Any, str | None]:
    if not file:
        return "Upload a CSV first.", None, None
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return f"Could not read CSV: {exc}", None, None
    for col in (pred_col, attr_col):
        if col not in df.columns:
            return f"Column '{col}' is not in the CSV.", None, None

    use_label = label_col and label_col != _NONE and label_col in df.columns
    keep = [pred_col, attr_col] + ([label_col] if use_label else [])
    df = df.dropna(subset=keep).copy()
    if len(df) < 50:
        return "Need at least ~50 rows after dropping missing values.", None, None

    y_pred = _to_binary(df[pred_col])

    def predict(_x: pd.DataFrame) -> np.ndarray:  # the user supplied predictions directly
        return y_pred

    spec = AttributeSpec(attr_col, privileged.strip() or None, None)
    try:
        result = run_audit(
            predict,
            df,
            [spec],
            target=(label_col if use_label else None),
            feature_columns=[pred_col],
            n_boot=500,
            seed=42,
        )
    except Exception as exc:
        return f"Audit failed: {exc}", None, None

    summary = (
        f"{_verdict(result)}\n\n```\n"
        + render_scorecard(result, title="EthicLens — your data")
        + "\n```\n"
        + _error_rate_md(df, y_pred, attr_col, label_col if use_label else None, result)
    )
    pdf_path = _write_pdf(result, attr_col)
    return summary, _bar(result), pdf_path


def on_csv_upload(file: str | None):
    import gradio as gr

    if not file:
        return gr.update(), gr.update(), gr.update()
    try:
        cols = list(pd.read_csv(file, nrows=5).columns)
    except Exception:
        return gr.update(), gr.update(), gr.update()
    guess_pred = next((c for c in cols if "pred" in c.lower()), cols[0])
    guess_label = next((c for c in cols if c.lower() in {"label", "y", "target", "true"}), _NONE)
    guess_attr = next(
        (c for c in cols if any(k in c.lower() for k in ("race", "sex", "gender", "group"))),
        cols[0],
    )
    return (
        gr.update(choices=cols, value=guess_pred),
        gr.update(choices=[_NONE, *cols], value=guess_label),
        gr.update(choices=cols, value=guess_attr),
    )


def make_sample_csv() -> str:
    """Generate a small example predictions CSV so users can see the expected format."""
    model, df, target = _train(7)
    out = df[["race", "gender"]].copy()
    out["prediction"] = model.predict(df[FEATURES])
    out["true_label"] = df[target].to_numpy()
    path = _TMP / "ethiclens_sample_predictions.csv"
    out.to_csv(path, index=False)
    return str(path)


# --------------------------------------------------------------------------- #
# Shared rendering helpers
# --------------------------------------------------------------------------- #


def _verdict(result: AuditResult) -> str:
    if result.composite_score is None:
        return "No comparable groups were found."
    flagged = [g.group_label for g in result.flagged_groups]
    tail = f"⚠️ Flagged: {', '.join(flagged)}." if flagged else "✅ No groups flagged."
    score = f"{result.composite_score:.3f}"
    return f"**Composite Bias Score: {score}** ({result.composite_band}). {tail}"


def _table(result: AuditResult) -> pd.DataFrame:
    rows = []
    for g in result.groups:
        di = g.metric(MetricName.DISPARATE_IMPACT)
        spd = g.metric(MetricName.SPD)
        eo = g.metric(MetricName.EQUALIZED_ODDS)
        ci = di.ci if di and di.ci else None
        rows.append(
            {
                "Group": g.group_label,
                "Disparate Impact": round(di.value, 3) if di and di.value is not None else None,
                "95% CI": f"[{ci.low:.2f}, {ci.high:.2f}]" if ci else "—",
                "SPD": round(spd.value, 3) if spd and spd.value is not None else None,
                "Eq. Odds": round(eo.value, 3) if eo and eo.value is not None else "N/A",
                "Status": "FLAG" if g.flagged else "OK",
            }
        )
    return pd.DataFrame(rows)


def _error_rate_md(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    attr_col: str,
    label_col: str | None,
    result: AuditResult,
) -> str:
    if not label_col:
        return "_Equalized Odds and error rates need a true-label column (none selected)._"
    y_true = _to_binary(df[label_col])
    sens = df[attr_col].to_numpy()
    lines = [
        "**Error-rate view (per group):**",
        "",
        "| Group | base rate | PPV | FPR | TPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for v in pd.unique(sens):
        mask = sens == v
        if mask.sum() == 0:
            continue
        tpr, fpr, ppv = confusion_rates(y_true, y_pred, mask)
        base = float(np.mean(y_true[mask]))
        lines.append(f"| {attr_col}:{v} | {base:.3f} | {_f(ppv)} | {_f(fpr)} | {_f(tpr)} |")
    return "\n".join(lines)


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _bar(result: AuditResult):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups, values, colors = [], [], []
    for g in result.groups:
        di = g.metric(MetricName.DISPARATE_IMPACT)
        if di and di.value is not None:
            groups.append(g.group_label)
            values.append(di.value)
            colors.append("#c0392b" if g.flagged else "#27ae60")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if values:
        ax.bar(groups, values, color=colors)
        ax.axhline(0.8, linestyle="--", color="#34495e")
        ax.set_ylim(0, max(1.2, max(values) + 0.1))
    ax.set_ylabel("Disparate Impact")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    return fig


def _write_pdf(result: AuditResult, dataset: str) -> str | None:
    try:
        from fairness_core.reporting import generate_scorecard_pdf
        from fairness_core.reporting.scorecard import ScorecardMeta

        pdf = generate_scorecard_pdf(
            result, ScorecardMeta(model_name="your model", dataset=dataset, seed=42)
        )
        path = _TMP / "EthicLens_Scorecard.pdf"
        path.write_bytes(pdf)
        return str(path)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


def build() -> Any:
    import gradio as gr

    with gr.Blocks(title="EthicLens — AI Bias Workbench") as demo:
        gr.Markdown(
            "# ⚖️ EthicLens — AI Bias Detection & Mitigation\n"
            "Audit a model for disparate impact and error-rate bias with bootstrap confidence "
            "intervals, then apply a **measured** mitigation. "
            "[Source & methodology](https://github.com/vardhjain/Ethiclens)."
        )

        with gr.Tab("Demo (synthetic biased model)"):
            with gr.Row():
                attribute = gr.Dropdown(
                    ["race", "gender"], value="race", label="Protected attribute"
                )
                seed = gr.Slider(0, 100, value=7, step=1, label="Seed")
            run_btn = gr.Button("Run audit", variant="primary")
            verdict = gr.Markdown()
            with gr.Row():
                table = gr.Dataframe(label="Per-group fairness metrics")
                plot = gr.Plot(label="Disparate Impact (red = flagged)")
            mit_btn = gr.Button("Apply top mitigation & re-audit")
            mit_out = gr.Markdown()
            run_btn.click(run_demo, [attribute, seed], [table, plot, verdict])
            mit_btn.click(run_mitigation, [attribute, seed], [mit_out])

        with gr.Tab("Audit your own data (CSV)"):
            gr.Markdown(
                "Upload a **CSV of your model's predictions**. We audit your *outputs* and never "
                "execute your model — so it's safe. Columns you need:\n"
                "- a **prediction** column (0/1 labels, or scores in [0,1]),\n"
                "- a **protected attribute** column (e.g. race, sex),\n"
                "- *(optional)* a **true-label** column — needed for Equalized Odds.\n\n"
                "Tip: in your code, `df['prediction'] = model.predict(X_test)`, add the true label "
                "and the protected attribute, then `df.to_csv(...)`."
            )
            gr.File(value=make_sample_csv, label="📥 Download a sample CSV (the format)")
            upload = gr.File(label="Your predictions CSV", file_types=[".csv"], type="filepath")
            with gr.Row():
                pred_col = gr.Dropdown(label="Prediction column", choices=[])
                label_col = gr.Dropdown(label="True-label column (optional)", choices=[_NONE])
                attr_col = gr.Dropdown(label="Protected attribute column", choices=[])
            privileged = gr.Textbox(
                label="Privileged value (optional; inferred if blank)",
                placeholder="e.g. White",
            )
            audit_btn = gr.Button("Audit my data", variant="primary")
            csv_plot = gr.Plot(label="Disparate Impact (red = flagged)")
            csv_out = gr.Markdown()
            csv_pdf = gr.File(label="Download Fairness Scorecard (PDF)")

            upload.change(on_csv_upload, [upload], [pred_col, label_col, attr_col])
            audit_btn.click(
                audit_csv,
                [upload, pred_col, label_col, attr_col, privileged],
                [csv_out, csv_plot, csv_pdf],
            )

    return demo


def main() -> None:
    build().launch()


if __name__ == "__main__":
    main()
