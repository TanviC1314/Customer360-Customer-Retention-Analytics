"""Render static PNG previews of the three intended Power BI pages from the exported data.

These are **Python/Matplotlib renderings of the planned dashboard layout, not Power BI
screenshots** - Power BI Desktop cannot be automated in this environment, so no ``.pbix`` file
is shipped. The previews use exactly the files in ``dashboard/dashboard_data`` so the numbers
match what the Power BI measures in ``dashboard/powerbi_guide.md`` should return.

    python -m src.render_dashboard_preview
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .exploratory_analysis import COLOR_CHURNED, COLOR_NEUTRAL, COLOR_RETAINED, set_style

set_style()
D = config.DASHBOARD_DATA_DIR
OUT = config.PROJECT_ROOT / "docs" / "images"
BG = "#F3F4F6"
USD = chr(92) + "$"  # escaped dollar sign (Matplotlib treats $...$ as math text)
FOOTER = "Python-rendered preview of the planned Power BI layout (not a Power BI screenshot)"


def _frame(title: str, subtitle: str, figsize=(16, 9)):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    fig.text(0.02, 0.965, title, fontsize=20, fontweight="bold", va="top")
    fig.text(0.02, 0.925, subtitle, fontsize=11, color="#555", va="top")
    fig.text(0.98, 0.012, FOOTER, fontsize=8, color="#777", ha="right")
    return fig


def _card(fig, rect, label, value, color="#1F2937"):
    ax = fig.add_axes(rect)
    ax.set_facecolor("white")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=22, fontweight="bold", color=color)
    ax.text(0.5, 0.2, label, ha="center", va="center", fontsize=10, color="#666")
    return ax


def _panel(fig, rect, title):
    ax = fig.add_axes(rect)
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="x", visible=False)
    return ax


def _rate_bars(ax, df, col, order, horizontal=False):
    g = df.groupby(col, observed=True)["churn_flag"].agg(["mean", "size"]).reindex(order)
    overall = df["churn_flag"].mean() * 100
    vals = g["mean"] * 100
    colors = [COLOR_CHURNED if v > overall else COLOR_RETAINED for v in vals]
    if horizontal:
        ax.barh(order, vals, color=colors)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, vals.max() * 1.2)
        ax.tick_params(axis="y", labelsize=8)
    else:
        ax.bar(order, vals, color=colors, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_ylim(0, vals.max() * 1.2)
        ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("")


def page1(fact: pd.DataFrame, kpi: pd.Series) -> None:
    fig = _frame("Customer360 | 1. Executive Overview", "Telco customer base, snapshot of 7,043 customers")
    labels = [("Total customers", f"{int(kpi.total_customers):,}", "#1F2937"),
              ("Churned customers", f"{int(kpi.churned_customers):,}", COLOR_CHURNED),
              ("Churn rate", f"{kpi.churn_rate_pct:.1f}%", COLOR_CHURNED),
              ("Avg monthly charges", f"${kpi.avg_monthly_charges:.2f}", "#1F2937"),
              ("Avg tenure (months)", f"{kpi.avg_tenure_months:.1f}", "#1F2937"),
              ("Active MRR", f"${kpi.active_mrr/1000:,.0f}K", COLOR_RETAINED)]
    w = 0.145
    for i, (lab, val, c) in enumerate(labels):
        _card(fig, [0.02 + i * 0.16, 0.75, w, 0.12], lab, val, c)

    ax = fig.add_axes([0.02, 0.38, 0.28, 0.32], facecolor="white")
    ax.pie([kpi.active_customers, kpi.churned_customers], labels=["Retained", "Churned"],
           colors=[COLOR_RETAINED, COLOR_CHURNED], startangle=90, counterclock=False,
           wedgeprops={"width": 0.4, "edgecolor": "white"}, autopct="%1.1f%%", pctdistance=0.8,
           textprops={"fontsize": 10})
    ax.set_title("Churn distribution", fontsize=11, fontweight="bold", loc="left")

    _rate_bars(_panel(fig, [0.35, 0.38, 0.3, 0.32], "Churn rate by contract type"), fact, "contract", config.CONTRACT_ORDER)
    ax = _panel(fig, [0.70, 0.38, 0.28, 0.32], "Churn rate by tenure group")
    _rate_bars(ax, fact, "tenure_group", config.TENURE_LABELS)

    ax = _panel(fig, [0.07, 0.06, 0.40, 0.25], "Active vs lost monthly recurring revenue (MRR) by contract")
    m = fact.groupby(["contract", "churn_status"])["monthly_charges"].sum().unstack().reindex(config.CONTRACT_ORDER) / 1000
    x = np.arange(3)
    ax.bar(x - 0.2, m["Retained"], 0.4, color=COLOR_RETAINED, label="Active MRR")
    ax.bar(x + 0.2, m["Churned"], 0.4, color=COLOR_CHURNED, label="Lost MRR")
    ax.set_xticks(x, config.CONTRACT_ORDER)
    ax.set_ylabel("$ thousand / month")
    ax.legend(fontsize=9)

    ax = fig.add_axes([0.52, 0.06, 0.46, 0.25], facecolor="white")
    ax.axis("off")
    ax.set_title("Key takeaways (from the data)", fontsize=11, fontweight="bold", loc="left")
    mtm = fact[fact.contract == "Month-to-month"]
    txt = (f"- Overall churn is {kpi.churn_rate_pct:.1f}% ({int(kpi.churned_customers):,} of {int(kpi.total_customers):,} customers).\n"
           f"- Month-to-month customers churn at {mtm.churn_flag.mean()*100:.1f}% vs "
           f"{fact[fact.contract=='Two year'].churn_flag.mean()*100:.1f}% on two-year contracts.\n"
           f"- {fact[fact.tenure_group=='0-12 months'].churn_flag.mean()*100:.1f}% of first-year customers churn "
           f"vs {fact[fact.tenure_group=='49-72 months'].churn_flag.mean()*100:.1f}% after four years.\n"
           f"- Churned customers paid {USD}{fact[fact.churn_flag==1].monthly_charges.mean():.2f}/month on average vs "
           f"{USD}{fact[fact.churn_flag==0].monthly_charges.mean():.2f} for retained.\n"
           f"- Lost MRR so far: {USD}{kpi.lost_mrr/1000:,.0f}K/month.")
    ax.text(0, 0.88, txt, va="top", fontsize=10.5, linespacing=1.9)
    fig.savefig(OUT / "dashboard_preview_1_executive_overview.png", dpi=110, facecolor=BG)
    plt.close(fig)


def page2(fact: pd.DataFrame) -> None:
    fig = _frame("Customer360 | 2. Customer Behaviour", "Where churn concentrates (bars above the overall rate are red)")
    pos = [[0.04, 0.52, 0.27, 0.34], [0.375, 0.52, 0.27, 0.34], [0.78, 0.52, 0.20, 0.34],
           [0.04, 0.08, 0.27, 0.34], [0.50, 0.08, 0.15, 0.34], [0.78, 0.08, 0.20, 0.34]]
    _rate_bars(_panel(fig, pos[0], "Churn rate by contract type"), fact, "contract", config.CONTRACT_ORDER)
    _rate_bars(_panel(fig, pos[1], "Churn rate by tenure group"), fact, "tenure_group", config.TENURE_LABELS)
    order = fact.groupby("payment_method")["churn_flag"].mean().sort_values(ascending=False).index.tolist()
    _rate_bars(_panel(fig, pos[2], "Churn rate by payment method"), fact, "payment_method", order, horizontal=True)
    _rate_bars(_panel(fig, pos[3], "Churn rate by internet service"), fact, "internet_service", ["DSL", "Fiber optic", "No internet"])

    ax = _panel(fig, pos[4], "Customer segmentation: size and churn")
    seg = fact.groupby("customer_segment")["churn_flag"].agg(["size", "mean"]).reindex(
        pd.read_csv(D / "dim_segment.csv")["customer_segment"])
    y = np.arange(len(seg))
    ax.grid(axis="y", visible=False)
    ax.barh(y, seg["size"], color=COLOR_NEUTRAL)
    ax.set_yticks(y, seg.index, fontsize=8)
    ax.invert_yaxis()
    for i, (n, r) in enumerate(zip(seg["size"], seg["mean"])):
        ax.text(n + 20, i, f"{n:,} | churn {r*100:.0f}%", va="center", fontsize=8.5,
                color=COLOR_CHURNED if r > fact.churn_flag.mean() else "#333")
    ax.set_xlim(0, seg["size"].max() * 1.55)
    ax.set_xlabel("Customers")

    ax = _panel(fig, pos[5], "Monthly charges: churned vs retained")
    data = [fact[fact.churn_flag == 0].monthly_charges, fact[fact.churn_flag == 1].monthly_charges]
    bp = ax.boxplot(data, tick_labels=["Retained", "Churned"], patch_artist=True, widths=0.5,
                    medianprops={"color": "black"})
    for patch, c in zip(bp["boxes"], [COLOR_RETAINED, COLOR_CHURNED]):
        patch.set_facecolor(c), patch.set_alpha(0.7)
    ax.set_ylabel("Monthly charges ($)")
    fig.savefig(OUT / "dashboard_preview_2_customer_behaviour.png", dpi=110, facecolor=BG)
    plt.close(fig)


def page3(fact: pd.DataFrame, seg: pd.DataFrame, imp: pd.DataFrame, opp: pd.DataFrame, kpi: pd.Series) -> None:
    fig = _frame("Customer360 | 3. Retention Insights",
                 "High-risk segments, churn drivers, revenue at risk and suggested actions")
    _card(fig, [0.02, 0.77, 0.15, 0.11], "Active customers scored High risk", f"{int(kpi.high_risk_active_customers):,}", COLOR_CHURNED)
    _card(fig, [0.19, 0.77, 0.15, 0.11], "Model-scored MRR at risk", f"${kpi.model_scored_revenue_at_risk_mrr/1000:,.1f}K", COLOR_CHURNED)

    ax = _panel(fig, [0.14, 0.07, 0.22, 0.62], "Highest-churn segments (n >= 50)")
    top = seg.head(8).iloc[::-1]
    ax.grid(axis="y", visible=False)
    ax.barh(top["segment_label"].str.replace(" | ", "\n", regex=False), top["churn_rate_pct"], color=COLOR_CHURNED)
    for i, (v, n) in enumerate(zip(top["churn_rate_pct"], top["customers"])):
        ax.text(v + 0.8, i, f"{v:.0f}% (n={n:,})", va="center", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Churn rate (%)")

    ax = _panel(fig, [0.45, 0.42, 0.22, 0.27], "Main churn factors (permutation importance)")
    t = imp.head(7).iloc[::-1]
    ax.grid(axis="y", visible=False)
    ax.barh(t["feature"], t["importance_mean"], color=COLOR_RETAINED)
    ax.set_xlabel("Drop in ROC-AUC if shuffled")
    ax.tick_params(axis="y", labelsize=8)

    ax = _panel(fig, [0.45, 0.07, 0.22, 0.27], "Active customers & MRR by risk band")
    a = fact[fact.churn_flag == 0].groupby("risk_band")["monthly_charges"].agg(["size", "sum"]).reindex(["Low", "Medium", "High"])
    ax.bar(a.index, a["sum"] / 1000, color=[COLOR_RETAINED, "#E9A23B", COLOR_CHURNED], width=0.6)
    for i, (n, s) in enumerate(zip(a["size"], a["sum"])):
        ax.text(i, s / 1000 + 2, f"{n:,} cust.\n${s/1000:,.0f}K", ha="center", fontsize=8.5)
    ax.set_ylim(0, a["sum"].max() / 1000 * 1.25)
    ax.set_ylabel("MRR ($ thousand)")

    ax = fig.add_axes([0.69, 0.07, 0.30, 0.80], facecolor="white")
    ax.axis("off")
    ax.set_title("Retention opportunities (illustrative sizing, overlapping groups)", fontsize=11, fontweight="bold", loc="left")
    y = 0.97
    for _, r in opp.iterrows():
        ax.text(0.0, y, r["opportunity"], fontsize=9.5, fontweight="bold", va="top", wrap=True)
        ax.text(0.0, y - 0.065,
                f"{int(r['active_customers']):,} active customers | ${r['active_mrr']/1000:,.0f}K MRR\n"
                f"churn {r['target_churn_rate_pct']:.0f}% vs {r['comparison_churn_rate_pct']:.0f}% ({r['comparison_group']})",
                fontsize=8.5, va="top", color="#444", linespacing=1.5)
        y -= 0.19
    fig.savefig(OUT / "dashboard_preview_3_retention_insights.png", dpi=110, facecolor=BG)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fact = pd.read_csv(D / "customers.csv")
    kpi = pd.read_csv(D / "kpi_summary.csv").iloc[0]
    seg = pd.read_csv(D / "segment_summary.csv")
    imp = pd.read_csv(D / "feature_importance.csv")
    opp = pd.read_csv(D / "retention_opportunities.csv")
    page1(fact, kpi)
    page2(fact)
    page3(fact, seg, imp, opp, kpi)
    print(f"Wrote previews to {OUT}")


if __name__ == "__main__":
    main()
