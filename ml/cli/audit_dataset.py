"""Audit a real fairness benchmark end-to-end: load -> train -> audit -> scorecard.

    python -m ml.cli.audit_dataset compas
    python -m ml.cli.audit_dataset german_credit --seed 7

Trains a transparent baseline model on the benchmark's legitimate features (never
on the protected attribute) and audits it on a held-out split using the same
``fairness_core`` engine as the API. For an adverse-outcome dataset like COMPAS,
it also prints the error-rate view (PPV / FPR per group) that reveals the
predictive-parity impossibility result the Disparate-Impact rule alone misses.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fairness_core import run_audit, set_global_seed
from fairness_core.cli import render_scorecard
from fairness_core.metrics.group import confusion_rates
from ml.datasets import available_datasets, load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=available_datasets())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-boot", type=int, default=800)
    args = parser.parse_args(argv)

    set_global_seed(args.seed)
    ds = load_dataset(args.dataset, use_cache=False)
    frame = ds.frame.dropna(subset=[*ds.feature_columns, ds.target]).copy()
    x = frame[ds.feature_columns].astype(float)
    y = frame[ds.target].astype(int)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    model = LogisticRegression(max_iter=5000).fit(x_tr, y_tr)
    test = frame.loc[x_te.index]

    result = run_audit(
        model,
        test,
        ds.protected,
        target=ds.target,
        feature_columns=ds.feature_columns,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    print(f"Dataset : {ds.name} — {ds.description}")
    print(f"Model   : LogisticRegression on {ds.feature_columns}")
    print(f"Audited : {len(test):,} held-out rows | model accuracy {model.score(x_te, y_te):.3f}")
    print(f"Positive (audited) outcome label: {ds.target} == {ds.positive_label}\n")
    print(render_scorecard(result, title=f"EthicLens Audit — {ds.name}"))

    _print_error_rate_view(model, test, x_te, y_te, ds)
    return 0


def _print_error_rate_view(model, test, x_te, y_te, ds) -> None:
    """Per-group base rate / PPV / FPR — the lens that exposes the COMPAS story."""
    y_pred = model.predict(x_te)
    y_true = y_te.to_numpy()
    print("\nError-rate view (per group) — read this for adverse outcomes like recidivism:")
    print(f"{'Group':<26}{'base rate':>10}{'PPV':>8}{'FPR':>8}{'TPR':>8}")
    print("-" * 60)
    for spec in ds.protected:
        sens = test[spec.name].to_numpy()
        values = [spec.privileged_value, *(spec.unprivileged_values or [])]
        for v in values:
            if v is None:
                continue
            mask = sens == v
            if mask.sum() == 0:
                continue
            tpr, fpr, ppv = confusion_rates(y_true, y_pred, mask)
            base = float(np.mean(y_true[mask]))
            print(
                f"{spec.name + ':' + str(v):<26}{base:>10.3f}{_f(ppv):>8}{_f(fpr):>8}{_f(tpr):>8}"
            )
    print(
        "\nWhen base rates differ, a model cannot equalise PPV (predictive parity) AND "
        "FPR/TPR\nat once (Chouldechova 2017). A large FPR gap with similar PPV is the "
        "classic COMPAS\nsignature — and why the 4/5ths Disparate-Impact rule alone can miss it."
    )


def _f(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
