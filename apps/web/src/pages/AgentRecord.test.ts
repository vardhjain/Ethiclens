import { describe, expect, it } from "vitest";
import type { AgentRecord, ScorecardMetric } from "../api/client";
import { buildMarkdownReport, formatMetricMarkdown } from "./AgentRecord";

describe("formatMetricMarkdown", () => {
  it("formats a metric with a classification", () => {
    const metric: ScorecardMetric = {
      name: "di",
      value: 0.566,
      classification: "FAIL",
      ci: null,
      n: 100,
    };
    expect(formatMetricMarkdown(metric)).toBe("0.566 (FAIL)");
  });

  it("formats a metric without a classification", () => {
    const metric: ScorecardMetric = {
      name: "di",
      value: 0.9,
      classification: null,
      ci: null,
      n: 100,
    };
    expect(formatMetricMarkdown(metric)).toBe("0.900");
  });

  it("reports insufficient data instead of a bogus number", () => {
    const metric: ScorecardMetric = {
      name: "di",
      value: null,
      classification: "INSUFFICIENT_DATA",
      ci: null,
      n: 5,
    };
    expect(formatMetricMarkdown(metric)).toBe("Insufficient data");
  });

  it("falls back to N/A for a missing metric", () => {
    expect(formatMetricMarkdown(undefined)).toBe("N/A");
  });
});

function _record(overrides: Partial<AgentRecord> = {}): AgentRecord {
  return {
    record_id: "r1",
    created_at: "2026-07-17T00:00:00Z",
    grounded: true,
    degraded: false,
    narrative: "The audit found a disparity.",
    mitigation_summary: "Apply a threshold optimizer.",
    plan: {
      include_equalized_odds: true,
      metrics_reason: "labels present",
      protected_attribute_columns: ["race"],
      outcome_column: "flag",
      true_label_column: "label",
      score_column: "score",
      feature_columns: [],
      excluded_subgroups: [],
    },
    scorecard: {
      composite_score: 0.62,
      composite_band: "MEDIUM RISK",
      min_disparate_impact: 0.566,
      has_labels: true,
      metrics_reason: "labels present",
      excluded_subgroups: [],
      groups: [
        {
          attribute: "race",
          group_label: "race:Black",
          privileged_value: "White",
          unprivileged_value: "Black",
          n_privileged: 146,
          n_unprivileged: 928,
          flagged: true,
          metrics: {
            disparate_impact: {
              name: "di",
              value: 0.566,
              classification: "FAIL",
              ci: null,
              n: 1074,
            },
          },
        },
      ],
      recommendations: {},
      measured_mitigation: null,
    },
    ...overrides,
  };
}

describe("buildMarkdownReport", () => {
  it("includes the risk band, composite score, and narrative", () => {
    const md = buildMarkdownReport(_record());
    expect(md).toContain("**Risk band:** MEDIUM RISK");
    expect(md).toContain("**Composite score:** 0.620");
    expect(md).toContain("The audit found a disparity.");
  });

  it("labels the narrative as numeric-only when not grounded", () => {
    const md = buildMarkdownReport(_record({ grounded: false }));
    expect(md).toContain("**Narrative:** Numeric-only");
  });

  it("lists excluded subgroups when present", () => {
    const md = buildMarkdownReport(
      _record({
        scorecard: {
          ..._record().scorecard,
          excluded_subgroups: [
            {
              attribute: "race",
              group_value: "Asian",
              n: 8,
              reason: "too small",
            },
          ],
        },
      }),
    );
    expect(md).toContain("Excluded race:Asian (n=8) — too small");
  });

  it("includes a measured-mitigation section when present", () => {
    const md = buildMarkdownReport(
      _record({
        scorecard: {
          ..._record().scorecard,
          measured_mitigation: {
            strategy: "threshold_optimizer",
            stage: "postprocessing",
            group_label: "race:Black",
            di_before: 0.566,
            di_after: 0.85,
            di_after_ci: null,
            accuracy_before: 0.9,
            accuracy_after: 0.888,
            di_improvement: 0.284,
            accuracy_cost: 0.012,
            crossed_threshold: true,
            n_test: 323,
          },
        },
      }),
    );
    expect(md).toContain(
      "## Mitigation: before vs. after (measured on held-out data)",
    );
    expect(md).toContain("0.566 → 0.850 (crosses 0.80 threshold)");
  });
});
