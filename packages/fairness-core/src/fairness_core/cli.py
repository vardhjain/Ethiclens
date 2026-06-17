"""``ethiclens-audit`` — a headless audit that prints a Fairness Scorecard.

Runs the *same* ``fairness_core`` code path as the API and notebooks, so the CLI
is a faithful, dependency-light way to demonstrate the engine end-to-end::

    ethiclens-audit demo          # build a biased model, audit it, print a scorecard
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from fairness_core import __version__, run_audit
from fairness_core.audit import AttributeSpec
from fairness_core.datasets import make_biased_lending_dataset
from fairness_core.seeds import DEFAULT_SEED, set_global_seed
from fairness_core.types import AuditResult, MetricName

# Backwards-compatible alias (kept for existing imports / tests).
make_biased_dataset = make_biased_lending_dataset


def _fmt(x: float | None, nd: int = 3) -> str:
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def render_scorecard(result: AuditResult, title: str = "EthicLens Fairness Scorecard") -> str:
    lines: list[str] = []
    bar = "=" * 78
    lines.append(bar)
    lines.append(title.center(78))
    lines.append(bar)
    lines.append(
        f"Composite Bias Score: {_fmt(result.composite_score)}  "
        f"[{result.composite_band}]   (higher = fairer)"
    )
    lines.append(
        f"Worst-group Disparate Impact: {_fmt(result.min_di)}   "
        f"Labels available: {'yes' if result.has_labels else 'no (Equalized Odds N/A)'}"
    )
    lines.append("-" * 78)
    header = f"{'Group':<22}{'DI':>8}{'95% CI':>16}{'SPD':>8}{'EO':>8}{'Flag':>8}"
    lines.append(header)
    lines.append("-" * 78)
    for g in result.groups:
        di = g.metric(MetricName.DISPARATE_IMPACT)
        spd = g.metric(MetricName.SPD)
        eo = g.metric(MetricName.EQUALIZED_ODDS)
        ci = di.ci if di and di.ci else None
        if ci is not None and not np.isnan(ci.low):
            ci_s = f"[{_fmt(ci.low, 2)},{_fmt(ci.high, 2)}]"
        else:
            ci_s = "      --"
        flag = "FLAG" if g.flagged else "ok"
        lines.append(
            f"{g.group_label:<22}{_fmt(di.value if di else None):>8}{ci_s:>16}"
            f"{_fmt(spd.value if spd else None):>8}"
            f"{_fmt(eo.value if eo and eo.value is not None else None):>8}{flag:>8}"
        )
    lines.append(bar)
    flagged = [g.group_label for g in result.flagged_groups]
    if flagged:
        lines.append(f"[!] {len(flagged)} flagged group(s): {', '.join(flagged)}")
        lines.append("    A group is flagged only when its DI confidence interval is below 0.80.")
    else:
        lines.append("[OK] No groups flagged below the 0.80 four-fifths threshold.")
    lines.append(bar)
    return "\n".join(lines)


FEATURES = ["income", "credit_score", "debt_ratio"]


def cmd_demo(args: argparse.Namespace) -> int:
    from sklearn.ensemble import GradientBoostingClassifier

    set_global_seed(args.seed)
    df, target = make_biased_dataset(n=args.n, seed=args.seed)
    model = GradientBoostingClassifier(random_state=args.seed).fit(df[FEATURES], df[target])

    result = run_audit(
        model,
        df,
        protected_attrs=[AttributeSpec("race"), AttributeSpec("gender")],
        target=target,
        feature_columns=FEATURES,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(render_scorecard(result))
    return 0


def cmd_mitigate(args: argparse.Namespace) -> int:
    from sklearn.linear_model import LogisticRegression

    from fairness_core import get_recommendations, mitigate_and_reaudit
    from fairness_core.mitigation import AVAILABLE_STRATEGIES

    set_global_seed(args.seed)
    df, target = make_biased_dataset(n=args.n, seed=args.seed)
    model = LogisticRegression(max_iter=2000, random_state=args.seed).fit(df[FEATURES], df[target])

    result = run_audit(
        model,
        df,
        [AttributeSpec("race"), AttributeSpec("gender")],
        target=target,
        feature_columns=FEATURES,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(render_scorecard(result))

    recs = get_recommendations(result)
    if not recs:
        print("\n[OK] No flagged groups — nothing to mitigate.")
        return 0

    group_label, group_recs = next(iter(recs.items()))
    print(f"\nRanked mitigation recommendations for {group_label}:")
    for r in group_recs:
        name = r.strategy_name
        print(f"  #{r.rank} {name:<34} +{r.estimated_di_improvement:.2f} DI [{r.stage}]")

    runnable = next((r for r in group_recs if r.strategy in AVAILABLE_STRATEGIES), group_recs[0])
    flagged = next(g for g in result.flagged_groups if g.group_label == group_label)
    res = mitigate_and_reaudit(
        model,
        df,
        flagged.attribute,
        flagged.privileged_value,
        flagged.unprivileged_value,
        target=target,
        feature_columns=FEATURES,
        strategy=runnable.strategy,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    ci = res.di_after_ci
    ci_s = f"  CI [{ci.low:.2f}, {ci.high:.2f}]" if ci else ""
    verdict = "PASS - DI now >= 0.80" if res.crossed_threshold else "still below 0.80"
    di_line = f"{res.di_before:.3f} -> {res.di_after:.3f}{ci_s}"
    acc_line = f"{res.accuracy_before:.3f} -> {res.accuracy_after:.3f}"
    print(f"\nApplied '{res.strategy}' ({res.stage}) and re-audited on a held-out split:")
    print(f"  {res.group_label} Disparate Impact:  {di_line}")
    print(f"  Accuracy:  {acc_line}")
    print(f"  [{verdict}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ethiclens-audit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"fairness-core {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Audit a freshly trained biased model.")
    demo.add_argument("--n", type=int, default=4000)
    demo.add_argument("--n-boot", type=int, default=1000)
    demo.add_argument("--seed", type=int, default=DEFAULT_SEED)
    demo.set_defaults(func=cmd_demo)

    mit = sub.add_parser("mitigate", help="Audit, recommend, apply a fix, and re-audit.")
    mit.add_argument("--n", type=int, default=4000)
    mit.add_argument("--n-boot", type=int, default=600)
    mit.add_argument("--seed", type=int, default=DEFAULT_SEED)
    mit.set_defaults(func=cmd_mitigate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
