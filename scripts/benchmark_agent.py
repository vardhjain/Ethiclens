"""Reproducible, no-API-cost evidence for EthicLens's three bundled agent demos.

Runs the deterministic CSV -> planning -> fairness-core -> held-out mitigation
-> scorecard path. No LLM, web service, database, or cloud credentials are used.
Timings are observational on the runner, not a performance SLA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from ethiclens_api.agent.audit_planner import plan_audit
from ethiclens_api.agent.demo_datasets import DEMO_DATASETS, DemoDataset
from ethiclens_api.agent.executor import execute_plan, measure_top_mitigation
from ethiclens_api.agent.narrative import build_scorecard

_EXPECTED_PATHS = {
    "compas": (True, False, False),
    "adult_income": (True, True, True),
    "synthetic_hiring": (False, True, True),
}


def validate_evidence(dataset: DemoDataset, data: pd.DataFrame, scorecard: dict) -> None:
    """Guard against regressions and independently check the reported DI ratio."""
    expected_labels, expected_no_measurement, favorable = _EXPECTED_PATHS[dataset.key]
    if scorecard["has_labels"] is not expected_labels:
        raise ValueError(f"{dataset.key}: ground-truth availability changed")
    if dataset.proposal.outcome_direction.positive_is_favorable is not favorable:
        raise ValueError(f"{dataset.key}: outcome polarity changed")
    if (scorecard["measured_mitigation"] is None) is not expected_no_measurement:
        raise ValueError(f"{dataset.key}: measured-mitigation availability changed")
    if not scorecard["groups"]:
        raise ValueError(f"{dataset.key}: no cohort comparisons were produced")

    for group in scorecard["groups"]:
        metrics = group["metrics"]
        eo = metrics.get("equalized_odds")
        if not expected_labels and eo is not None and eo["value"] is not None:
            raise ValueError(f"{dataset.key}: reported Equalized Odds without labels")

        # Independent calculation directly from CSV outcome rates. This catches
        # polarity mistakes even if the main engine produces internally plausible
        # numbers. Skip underpowered/null results without treating them as zero.
        di = metrics.get("disparate_impact")
        if di is None or di["value"] is None:
            continue
        attr = group["attribute"]
        values = data[attr].astype(str)
        raw = data[dataset.proposal.outcome_column].astype(int)
        selected = raw if favorable else 1 - raw
        privileged = selected[values == group["privileged_value"]]
        unprivileged = selected[values == group["unprivileged_value"]]
        if privileged.empty or unprivileged.empty:
            raise ValueError(f"{dataset.key}: reported unknown comparison group")
        reference_rate = float(privileged.mean())
        if reference_rate == 0:
            raise ValueError(f"{dataset.key}: DI reference rate is zero")
        independent_di = float(unprivileged.mean()) / reference_rate
        if abs(independent_di - di["value"]) > 0.015:
            raise ValueError(
                f"{dataset.key}: independent DI {independent_di:.4f} "
                f"disagrees with reported {di['value']:.4f}"
            )



def _metric_supported(metric: dict | None) -> bool:
    """A diagnostic point estimate is not sufficient statistical evidence."""
    return (
        metric is not None
        and metric.get("value") is not None
        and metric.get("classification") != "INSUFFICIENT_DATA"
    )


def _missing_reason(metric: dict | None) -> str | None:
    if _metric_supported(metric):
        return None
    if metric is None:
        return "Metric was not produced."
    return metric.get("details", {}).get("reason") or "Insufficient metric evidence."


def describe_group_evidence(groups: list[dict]) -> list[dict]:
    """Preserve per-group availability, reasons and bootstrap CI provenance.

    Unlike group point estimates, supported values must pass the engine's
    evidence floors. Underpowered diagnostic point estimates remain separate.
    """
    evidence = []
    for group in groups:
        metrics = group["metrics"]
        di = metrics.get("disparate_impact")
        spd = metrics.get("spd")
        eo = metrics.get("equalized_odds")
        di_supported = _metric_supported(di)
        eo_supported = _metric_supported(eo)
        spd_supported = _metric_supported(spd)
        di_interval = di.get("ci") if di_supported and di is not None else None
        eo_interval = eo.get("ci") if eo_supported and eo is not None else None
        evidence.append(
            {
                "group_label": group["group_label"],
                "reference_group": group["privileged_value"],
                "comparison_group": group["unprivileged_value"],
                "reference_rows": group["n_privileged"],
                "comparison_rows": group["n_unprivileged"],
                "flagged": bool(group["flagged"]),
                "di_supported": di_supported,
                "di_value": di["value"] if di_supported and di is not None else None,
                "di_raw_point_estimate": di.get("value") if di is not None else None,
                "di_ci": di_interval,
                "di_classification": di.get("classification") if di is not None else None,
                "di_reason": _missing_reason(di),
                "spd_supported": spd_supported,
                "spd_reason": _missing_reason(spd),
                "eo_supported": eo_supported,
                "eo_value": eo["value"] if eo_supported and eo is not None else None,
                "eo_raw_point_estimate": eo.get("value") if eo is not None else None,
                "eo_ci": eo_interval,
                "eo_classification": eo.get("classification") if eo is not None else None,
                "eo_reason": _missing_reason(eo),
                "composite_eligible": di_supported and spd_supported and eo_supported,
            }
        )
    return evidence


def validate_coverage(scorecard: dict, evidence: list[dict]) -> None:
    """Fail CI when a headline hides an excluded cohort or selects a subset."""
    if not evidence:
        if scorecard["min_disparate_impact"] is not None:
            raise ValueError("minimum DI published with no group comparisons")
        if scorecard["composite_score"] is not None:
            raise ValueError("composite published with no group comparisons")
        return
    supported = [g["di_value"] for g in evidence if g["di_supported"]]
    full_di = len(supported) == len(evidence)
    reported_di = scorecard["min_disparate_impact"]
    if not full_di and reported_di is not None:
        raise ValueError("minimum DI must be withheld when a cohort is underpowered")
    if full_di and (reported_di is None or abs(reported_di - min(supported)) > 1e-8):
        raise ValueError("minimum DI does not cover all supported cohorts")

    full_composite = all(g["composite_eligible"] for g in evidence)
    if (scorecard["composite_score"] is not None) != full_composite:
        raise ValueError("composite availability contradicts per-cohort evidence")
    if not full_composite and not scorecard.get("composite_reason"):
        raise ValueError("unavailable composite must include an explanation")


def _interval_label(interval: dict | None) -> str:
    if interval is None:
        return "Unavailable"
    low = interval.get("low")
    high = interval.get("high")
    level = interval.get("level")
    if low is None or high is None or level is None:
        return "Unavailable"
    if not (0 < level < 1 and low <= high):
        raise ValueError("invalid confidence interval metadata")
    method = interval.get("method") or "unspecified method"
    return f"[{low:.3f}, {high:.3f}] ({level:.0%}, {method})"


def _coverage_reason(evidence: list[dict], metric: str) -> str | None:
    if not evidence:
        return "No group comparisons were produced."
    omitted = sum(not g[f"{metric}_supported"] for g in evidence)
    if omitted:
        return (
            f"{omitted} of {len(evidence)} group comparisons have insufficient "
            f"{metric.upper()} evidence; see cohort-level reasons below."
        )
    return None


def benchmark_case(dataset: DemoDataset, repeats: int) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    timings: list[dict[str, float]] = []
    last_scorecard: dict = {}
    last_data: pd.DataFrame | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        frame = pd.read_csv(dataset.csv_path)
        parsed = time.perf_counter()
        plan = plan_audit(dataset.proposal, frame)
        result = execute_plan(plan, frame, dataset.proposal.outcome_direction)
        audited = time.perf_counter()
        measurement = measure_top_mitigation(
            plan, frame, dataset.proposal.outcome_direction, result
        )
        measured = time.perf_counter()
        scorecard, _ = build_scorecard(result, plan, measurement)
        assembled = time.perf_counter()
        validate_evidence(dataset, frame, scorecard)
        timings.append(
            {
                "csv_load_ms": round((parsed - started) * 1000, 2),
                "audit_ms": round((audited - parsed) * 1000, 2),
                "heldout_mitigation_ms": round((measured - audited) * 1000, 2),
                "scorecard_validation_ms": round((assembled - measured) * 1000, 2),
                "total_ms": round((assembled - started) * 1000, 2),
            }
        )
        last_scorecard = scorecard
        last_data = frame

    assert last_data is not None
    group_evidence = describe_group_evidence(last_scorecard["groups"])
    validate_coverage(last_scorecard, group_evidence)
    output = {
        "key": dataset.key,
        "title": dataset.title,
        "source_sha256": hashlib.sha256(dataset.csv_path.read_bytes()).hexdigest(),
        "rows": len(last_data),
        "labels_available": last_scorecard["has_labels"],
        "positive_is_favorable": dataset.proposal.outcome_direction.positive_is_favorable,
        "continuous_score_available": dataset.proposal.score_column is not None,
        "cohort_comparisons": len(last_scorecard["groups"]),
        "computable_di_cohorts": sum(g["di_supported"] for g in group_evidence),
        "computable_eo_cohorts": sum(g["eo_supported"] for g in group_evidence),
        "composite_eligible_cohorts": sum(g["composite_eligible"] for g in group_evidence),
        "di_coverage_reason": _coverage_reason(group_evidence, "di"),
        "group_evidence": group_evidence,
        "flagged_cohorts": sum(bool(g["flagged"]) for g in last_scorecard["groups"]),
        "inconclusive_cohorts": sum(
            any(m.get("classification") == "INSUFFICIENT_DATA" for m in g["metrics"].values())
            for g in last_scorecard["groups"]
        ),
        "min_disparate_impact": last_scorecard["min_disparate_impact"],
        "composite_score": last_scorecard["composite_score"],
        "composite_reason": last_scorecard["composite_reason"],
        "heldout_mitigation": last_scorecard["measured_mitigation"],
        "timing_ms": {
            name: {
                "median": round(statistics.median(row[name] for row in timings), 2),
                "min": min(row[name] for row in timings),
                "max": max(row[name] for row in timings),
            }
            for name in timings[0]
        },
    }
    return output


def render_markdown(report: dict) -> str:
    rows = [
        "# Offline EthicLens agent evidence",
        "",
        "Reproducible evaluation of the **bundled demo CSVs**, not a production "
        "benchmark or regulatory certification. The human-confirmed demo "
        "schemas are used as-is; LLM inference, narratives, Q&A, HTTP, database "
        "and cloud are intentionally excluded.",
        "",
        f"Python {report['environment']['python']} · "
        f"pandas {report['environment']['pandas']} · "
        f"scikit-learn {report['environment']['scikit_learn']} · "
        f"{report['repeats']} run(s)/dataset.",
        "",
        "| Dataset | Rows | Labels | Favorable polarity | Flagged groups | "
        "Overall DI | DI comparisons available | Measured mitigation | Median wall time |",
        "|---|---:|---|---|---:|---:|---:|---|---:|",
    ]
    for result in report["cases"]:
        di = result["min_disparate_impact"]
        di_text = f"{di:.3f}" if di is not None else "not assessed"
        mm = result["heldout_mitigation"]
        measured = f"yes (held-out n={mm['n_test']})" if mm is not None else "not available"
        polarity = "yes" if result["positive_is_favorable"] else "no (inverted)"
        rows.append(
            f"| {result['key']} | {result['rows']} | "
            f"{'yes' if result['labels_available'] else 'no'} | {polarity} | "
            f"{result['flagged_cohorts']} | {di_text} | "
            f"{result['computable_di_cohorts']} / {result['cohort_comparisons']} | "
            f"{measured} | {result['timing_ms']['total_ms']['median']:.0f} ms |"
        )
    rows.extend(
        [
            "",
            "## Per-cohort evidence and confidence intervals",
            "",
            "| Dataset | Comparison (reference → cohort) | n (ref / cohort) | "
            "Supported DI | DI confidence interval | EO evidence | Exclusion reason |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    for result in report["cases"]:
        for group in result["group_evidence"]:
            di_text = (
                f"{group['di_value']:.3f}"
                if group["di_supported"]
                else "Not assessed"
            )
            reason_parts = []
            if group["di_reason"]:
                reason_parts.append(f"DI: {group['di_reason']}")
            if group["spd_reason"]:
                reason_parts.append(f"SPD: {group['spd_reason']}")
            if group["eo_reason"]:
                reason_parts.append(f"EO: {group['eo_reason']}")
            reasons = "; ".join(reason_parts) if reason_parts else "None"
            # Group names/reasons originate from reviewed demo inputs and the
            # deterministic engine; escape table syntax in output.
            reasons = reasons.replace("|", "\\|").replace("\\n", " ")
            label = group["group_label"].replace("|", "\\|")
            ref = group["reference_group"].replace("|", "\\|")
            rows.append(
                f"| {result['key']} | {label} (ref: {ref}) | "
                f"{group['reference_rows']} / {group['comparison_rows']} | "
                f"{di_text} | {_interval_label(group['di_ci'])} | "
                f"{'Supported' if group['eo_supported'] else 'Not assessed'} | "
                f"{reasons} |"
            )
        rows.append("")
        if result["di_coverage_reason"]:
            rows.append(f"**{result['key']} DI headline withheld:** {result['di_coverage_reason']}")
        if result["composite_reason"]:
            rows.append(
                f"**{result['key']} composite unavailable:** {result['composite_reason']}"
            )
        rows.append("")

    rows += [
        "",
        "Timings are measured end-to-end on the current runner, include the "
        "fairness-core bootstrap calculations, and are **not comparable across "
        "different machines or GitHub Actions runner loads**. No throughput, "
        "p95 latency, inference accuracy, dollar cost, or fairness improvement "
        "can be inferred from them.",
        "",
        "Only cohorts meeting the deterministic minimum-sample policy receive "
        "a supported DI estimate (100 reference and 100 comparison rows), and "
        "Equalized Odds needs 30 positive and 30 negative labels on BOTH sides. "
        "CIs are the fairness engine's per-cohort bootstrap intervals, not CIs "
        "for the all-groups minimum, and can be unavailable independently of a "
        "supported point estimate. A metric point estimate for an underpowered "
        "cohort is diagnostic only; it must never be treated as a finding. "
        "The overall minimum DI requires every comparison to be eligible.",
        "",
        "See `evidence.json` for source-file SHA-256 hashes, per-stage median/min/max "
        "timings and each dataset's actual held-out mitigation fields. Null values "
        "mean insufficient data or unavailable measurement, **not zero impact**.",
        "",
    ]
    return "\n".join(rows)


def run(repeats: int) -> dict:
    return {
        "schema_version": 2,
        "purpose": "offline deterministic-agent evaluation, not LLM evaluation",
        "repeats": repeats,
        "environment": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "platform": platform.system(),
        },
        "cases": [benchmark_case(dataset, repeats) for dataset in DEMO_DATASETS.values()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/agent-evidence"))
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.repeats > 20:
        parser.error("--repeats must be between 1 and 20")
    report = run(args.repeats)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "README.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
