"""EthicLens live demo (Gradio) — deployable to Hugging Face Spaces.

A self-contained walkthrough of the workbench: train a biased model, audit it,
inspect the Disparate-Impact chart with confidence intervals, then apply a real
mitigation and watch the re-audit cross the 0.80 threshold. Uses the same
``fairness_core`` engine as the API and CLI.

Run locally:  python -m ml.demo.app
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fairness_core import mitigate_and_reaudit, run_audit
from fairness_core.audit import AttributeSpec
from fairness_core.datasets import make_biased_lending_dataset
from fairness_core.types import MetricName

FEATURES = ["income", "credit_score", "debt_ratio"]


def _train(seed: int):
    from sklearn.linear_model import LogisticRegression

    df, target = make_biased_lending_dataset(n=6000, seed=seed)
    model = LogisticRegression(max_iter=2000, random_state=seed).fit(df[FEATURES], df[target])
    return model, df, target


def run_audit_demo(attribute: str, seed: int) -> tuple[pd.DataFrame, Any, str]:
    model, df, target = _train(seed)
    result = run_audit(
        model,
        df,
        [AttributeSpec(attribute)],
        target=target,
        feature_columns=FEATURES,
        n_boot=400,
        seed=seed,
    )
    rows = []
    chart_groups, chart_di, chart_colors = [], [], []
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
        if di and di.value is not None:
            chart_groups.append(g.group_label)
            chart_di.append(di.value)
            chart_colors.append("#c0392b" if g.flagged else "#27ae60")

    fig = _bar(chart_groups, chart_di, chart_colors)
    verdict = (
        f"**Composite Bias Score: {result.composite_score:.3f}** "
        f"({result.composite_band}). "
        + (
            f"⚠️ Flagged: {', '.join(g.group_label for g in result.flagged_groups)}."
            if result.flagged_groups
            else "✅ No groups flagged below 0.80."
        )
    )
    return pd.DataFrame(rows), fig, verdict


def mitigate_demo(attribute: str, seed: int) -> str:
    model, df, target = _train(seed)
    audit = run_audit(
        model,
        df,
        [AttributeSpec(attribute)],
        target=target,
        feature_columns=FEATURES,
        n_boot=200,
        seed=seed,
    )
    flagged = audit.flagged_groups
    if not flagged:
        return "No flagged groups to mitigate."
    g = flagged[0]
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
        seed=seed,
    )
    verdict = "✅ DI now ≥ 0.80" if res.crossed_threshold else "still below 0.80"
    return (
        f"### Applied ThresholdOptimizer to `{res.group_label}` (measured on held-out data)\n"
        f"- Disparate Impact: **{res.di_before:.3f} → {res.di_after:.3f}** "
        f"(95% CI [{res.di_after_ci.low:.2f}, {res.di_after_ci.high:.2f}])\n"
        f"- Accuracy: {res.accuracy_before:.3f} → {res.accuracy_after:.3f}\n"
        f"- **{verdict}**"
    )


def _bar(groups, values, colors):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(groups, values, color=colors)
    ax.axhline(0.8, linestyle="--", color="#34495e")
    ax.set_ylabel("Disparate Impact")
    ax.set_ylim(0, max(1.1, max(values) + 0.1) if values else 1.1)
    fig.tight_layout()
    return fig


def build() -> Any:
    import gradio as gr

    with gr.Blocks(title="EthicLens — AI Bias Workbench") as demo:
        gr.Markdown(
            "# ⚖️ EthicLens — AI Bias Detection & Mitigation\n"
            "Train a biased lending model, audit it for disparate impact with bootstrap "
            "confidence intervals, then apply a **measured** mitigation that crosses the "
            "0.80 four-fifths threshold. "
            "[Source & methodology](https://github.com/vardhjain/Ethiclens)."
        )
        with gr.Row():
            attribute = gr.Dropdown(["race", "gender"], value="race", label="Protected attribute")
            seed = gr.Slider(0, 100, value=7, step=1, label="Seed")
        run_btn = gr.Button("Run audit", variant="primary")
        verdict = gr.Markdown()
        with gr.Row():
            table = gr.Dataframe(label="Per-group fairness metrics")
            plot = gr.Plot(label="Disparate Impact (red = flagged)")
        mit_btn = gr.Button("Apply top mitigation & re-audit")
        mit_out = gr.Markdown()

        run_btn.click(run_audit_demo, [attribute, seed], [table, plot, verdict])
        mit_btn.click(mitigate_demo, [attribute, seed], [mit_out])
    return demo


def main() -> None:
    build().launch()


if __name__ == "__main__":
    main()
