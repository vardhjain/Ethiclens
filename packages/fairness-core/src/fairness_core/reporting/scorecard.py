"""Fairness Scorecard PDF generation (FR-007).

Produces a self-contained PDF: an executive verdict in plain language (for the
non-technical Compliance Officer), a per-group metrics table with confidence
intervals, a Disparate-Impact chart against the 0.80 line, and an honesty banner
stating exactly what the audit did and did not measure.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from fairness_core.types import AuditResult, GroupAuditResult, MetricName

__all__ = ["ScorecardMeta", "generate_scorecard_pdf"]


@dataclass
class ScorecardMeta:
    model_name: str = "Unnamed model"
    dataset: str = "synthetic"
    session_id: str = "—"
    audit_date: str = "—"  # caller supplies (the engine never reads the clock)
    organization: str = "EquitaTech Systems"
    threshold: float = 0.80
    seed: int | None = None
    notes: list[str] = field(default_factory=list)


def _di_chart_png(result: AuditResult, threshold: float) -> bytes | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is an optional extra
        return None

    groups, values, colors = [], [], []
    for g in result.groups:
        m = g.metric(MetricName.DISPARATE_IMPACT)
        if m is None or m.value is None:
            continue
        groups.append(g.group_label)
        values.append(m.value)
        colors.append("#c0392b" if g.flagged else "#27ae60")
    if not groups:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 0.5 + 0.45 * len(groups)))
    ax.barh(groups, values, color=colors)
    ax.axvline(threshold, color="#34495e", linestyle="--", linewidth=1)
    ax.text(threshold, -0.6, "  0.80 four-fifths rule", color="#34495e", fontsize=8)
    ax.set_xlabel("Disparate Impact (1.0 = parity)")
    ax.set_xlim(0, max(1.1, max(values) + 0.1))
    ax.invert_yaxis()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _di_of(g: GroupAuditResult) -> float:
    m = g.metric(MetricName.DISPARATE_IMPACT)
    return m.value if m is not None and m.value is not None else 1.0


def _verdict_text(result: AuditResult, threshold: float) -> str:
    flagged = result.flagged_groups
    if not flagged:
        return (
            "All evaluated demographic groups meet the 0.80 four-fifths threshold. "
            "No disparate impact was detected at the configured significance level."
        )
    worst = min(flagged, key=_di_of)
    pct = round(_di_of(worst) * 100)
    return (
        f"<b>{len(flagged)} group(s) flagged.</b> The most affected group "
        f"(<b>{worst.group_label}</b>) is selected at about <b>{pct}%</b> of the favoured "
        f"group's rate — below the 80% regulatory threshold. Review the recommended "
        f"mitigations before this model is cleared for production."
    )


def generate_scorecard_pdf(result: AuditResult, meta: ScorecardMeta) -> bytes:
    """Render the Fairness Scorecard and return the PDF as bytes."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, title="EthicLens Fairness Scorecard")
    flow: list = []

    flow.append(Paragraph("EthicLens Fairness Scorecard", styles["Title"]))
    flow.append(Paragraph(meta.organization, styles["Normal"]))
    flow.append(Spacer(1, 0.15 * inch))

    info = [
        ["Model", meta.model_name, "Dataset", meta.dataset],
        ["Session", meta.session_id, "Audit date", meta.audit_date],
        [
            "Composite",
            f"{_fmt(result.composite_score)} ({result.composite_band})",
            "Worst DI",
            _fmt(result.min_di),
        ],
    ]
    info_table = Table(info, colWidths=[1.0 * inch, 2.2 * inch, 1.0 * inch, 2.0 * inch])
    info_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#ecf0f1")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
            ]
        )
    )
    flow.append(info_table)
    flow.append(Spacer(1, 0.2 * inch))

    flow.append(Paragraph("Executive verdict", styles["Heading2"]))
    flow.append(Paragraph(_verdict_text(result, meta.threshold), styles["Normal"]))
    flow.append(Spacer(1, 0.2 * inch))

    chart = _di_chart_png(result, meta.threshold)
    if chart is not None:
        flow.append(Image(io.BytesIO(chart), width=6.0 * inch, height=None))
        flow.append(Spacer(1, 0.2 * inch))

    flow.append(Paragraph("Per-group metrics", styles["Heading2"]))
    header = ["Group", "DI", "95% CI", "SPD", "Eq. Odds", "Status"]
    rows = [header]
    flag_rows: list[int] = []
    for i, g in enumerate(result.groups, start=1):
        di = g.metric(MetricName.DISPARATE_IMPACT)
        spd = g.metric(MetricName.SPD)
        eo = g.metric(MetricName.EQUALIZED_ODDS)
        ci = di.ci if di and di.ci else None
        ci_s = f"[{_fmt(ci.low, 2)}, {_fmt(ci.high, 2)}]" if ci and not _nan(ci.low) else "—"
        eo_val = eo.value if eo and eo.value is not None else None
        rows.append(
            [
                g.group_label,
                _fmt(di.value if di else None),
                ci_s,
                _fmt(spd.value if spd else None),
                _fmt(eo_val) if eo_val is not None else "N/A",
                "FLAG" if g.flagged else "OK",
            ]
        )
        if g.flagged:
            flag_rows.append(i)

    table = Table(rows, repeatRows=1)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
    ]
    for r in flag_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fdecea")))
        style.append(("TEXTCOLOR", (5, r), (5, r), colors.HexColor("#c0392b")))
    table.setStyle(TableStyle(style))
    flow.append(table)
    flow.append(Spacer(1, 0.25 * inch))

    banner = (
        "<b>What this scorecard does and does not prove.</b> Metrics are statistical "
        "evidence, not a legal clearance. The composite is a triage convenience, not a "
        "standard; always read the per-group values and their confidence intervals. "
        "Equalized Odds requires ground-truth labels and is shown as N/A otherwise. "
        f"Audit reproduced under seed {meta.seed if meta.seed is not None else 'fixed'} "
        f"on dataset '{meta.dataset}'."
    )
    flow.append(Paragraph(banner, styles["Italic"]))
    for note in meta.notes:
        flow.append(Paragraph(f"• {note}", styles["Normal"]))

    doc.build(flow)
    return buf.getvalue()


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None or _nan(x):
        return "—"
    return f"{x:.{nd}f}"


def _nan(x: float) -> bool:
    return x != x
