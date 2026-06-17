"""Generate committed sample report artefacts from the golden model.

    python -m ml.cli.generate_sample_report

Writes a Fairness Scorecard PDF, a Model Card, and a Datasheet into ``docs/`` so
the repository showcases real, generated outputs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fairness_core.reporting import (
    generate_datasheet,
    generate_model_card,
    generate_scorecard_pdf,
)
from fairness_core.reporting.datasheet import DatasheetInfo
from fairness_core.reporting.scorecard import ScorecardMeta
from ml.training.train_calibrated_bias_model import FEATURES, measure_black_di, train_model

_DOCS = Path(__file__).resolve().parents[2] / "docs"


def main() -> int:
    from fairness_core import run_audit
    from fairness_core.audit import AttributeSpec

    model, df, target = train_model()
    di = measure_black_di(model, df, target)
    result = run_audit(
        model,
        df,
        [AttributeSpec("race"), AttributeSpec("gender")],
        target=target,
        feature_columns=FEATURES,
    )
    today = date.today().isoformat()

    _DOCS.mkdir(parents=True, exist_ok=True)
    pdf = generate_scorecard_pdf(
        result,
        ScorecardMeta(
            model_name="calibrated_bias_model (golden)",
            dataset="synthetic-lending",
            session_id="SAMPLE-001",
            audit_date=today,
            seed=42,
            notes=[f"Golden race:Black Disparate Impact reproduced at {di:.3f} (pinned band)."],
        ),
    )
    (_DOCS / "sample-scorecard.pdf").write_bytes(pdf)
    (_DOCS / "sample-model-card.md").write_text(
        generate_model_card(
            result,
            model_name="calibrated_bias_model (golden)",
            dataset="synthetic-lending",
            audit_date=today,
        ),
        encoding="utf-8",
    )
    (_DOCS / "sample-datasheet.md").write_text(
        generate_datasheet(
            DatasheetInfo(
                name="synthetic-lending",
                n_rows=8000,
                protected_attributes=["race", "gender"],
                motivation="A controlled, labelled dataset with a known disparate impact, used as "
                "the bias-injection oracle that validates the EthicLens measurement engine.",
                composition="8,000 synthetic loan applications with income, credit score and debt "
                "ratio features; the disadvantaged group has systematically weaker inputs (proxy "
                "discrimination). Labels are a fair function of the features (no race term).",
                collection="Generated deterministically under a fixed seed; see "
                "fairness_core.datasets.make_biased_lending_dataset.",
                uses="Engine validation and demos only; not a substitute for real-data audits.",
            )
        ),
        encoding="utf-8",
    )
    print(f"Wrote sample scorecard, model card, and datasheet to {_DOCS} (golden DI {di:.3f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
