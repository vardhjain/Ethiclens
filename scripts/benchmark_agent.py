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
    group_evidence = [
        {
            "group_label": g["group_label"],
            "reference_rows": g["n_privileged"],
            "comparison_rows": g["n_unprivileged"],
            "flagged": g["flagged"],
            "di_value": (g["metrics"].get("disparate_impact") or {}).get("value"),
            "di_ci": (g["metrics"].get("disparate_impact") or {}).get("ci"),
            "di_classification": (g["metrics"].get("disparate_impact") or {}).get(
                "classification"
            ),
            "eo_value": (g["metrics"].get("equalized_odds") or {}).get("value"),
            "eo_classification": (g["metrics"].get("equalized_odds") or {}).get(
                "classification"
            ),
        }
        for g in last_scorecard["groups"]
    ]
    output = {
        "key": dataset.key,
        "title": dataset.title,
        "source_sha256": hashlib.sha256(dataset.csv_path.read_bytes()).hexdigest(),
        "rows": len(last_data),
        "labels_available": last_scorecard["has_labels"],
        "positive_is_favorable": dataset.proposal.outcome_direction.positive_is_favorable,
        "continuous_score_available": dataset.proposal.score_column is not None,
        "cohort_comparisons": len(last_scorecard["groups"]),
        "computable_di_cohorts": sum(g["di_value"] is not None for g in group_evidence),
        "group_evidence": group_evidence,
        "flagged_cohorts": sum(bool(g["flagged"]) for g in last_scorecard["groups"]),
        "inconclusive_cohorts": sum(
            any(m.get("classification") == "INSUFFICIENT_DATA" for m in g["metrics"].values())
            for g in last_scorecard["groups"]
        ),
        "min_disparate_impact": last_scorecard["min_disparate_impact"],
        "composite_score": last_scorecard["composite_score"],
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
    rows += [
        "",
        "Timings are measured end-to-end on the current runner, include the "
        "fairness-core bootstrap calculations, and are **not comparable across "
        "different machines or GitHub Actions runner loads**. No throughput, "
        "p95 latency, inference accuracy, dollar cost, or fairness improvement "
        "can be inferred from them.",
        "",
        "Overall DI may be inconclusive even if some group-level DI estimates are "
        "computable: see the group_evidence records instead of treating the "
        "missing aggregate as zero.",
        "",
        "See `evidence.json` for source-file SHA-256 hashes, per-stage median/min/max "
        "timings and each dataset's actual held-out mitigation fields. Null values "
        "mean insufficient data or unavailable measurement, **not zero impact**.",
        "",
    ]
    return "\n".join(rows)


def run(repeats: int) -> dict:
    return {
        "schema_version": 1,
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
