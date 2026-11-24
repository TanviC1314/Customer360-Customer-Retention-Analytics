"""Exploratory data analysis: summary tables, statistical tests and charts.

Every plotting helper returns the Matplotlib figure and (optionally) saves a PNG into
``reports/figures``. ``run`` regenerates all figures and the summary tables used in the
notebooks and reports.

    python -m src.exploratory_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; notebooks override with inline backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config, feature_engineering

# --- consistent look ---------------------------------------------------------
COLOR_RETAINED = "#3A6EA5"  # blue
COLOR_CHURNED = "#D1495B"  # red
COLOR_NEUTRAL = "#8D99AE"
PALETTE = {"No": COLOR_RETAINED, "Yes": COLOR_CHURNED}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


set_style()


def load_features() -> pd.DataFrame:
    """Load the clean dataset with engineered features (building them if needed)."""
    if config.FEATURES_FILE.exists():
        return pd.read_csv(config.FEATURES_FILE)
    return feature_engineering.engineer_features(pd.read_csv(config.CLEAN_FILE))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def overall_churn(df: pd.DataFrame) -> dict:
    n, churned = len(df), int(df[config.TARGET].sum())
    return {
        "customers": n,
        "churned": churned,
        "retained": n - churned,
        "churn_rate_pct": round(churned / n * 100, 2),
    }


def churn_rate_by(df: pd.DataFrame, col: str, order: list[str] | None = None) -> pd.DataFrame:
    """Customers, churners and churn rate (%) per category of ``col``."""
    g = df.groupby(col, observed=True)[config.TARGET].agg(customers="count", churned="sum")
    g["churn_rate_pct"] = (g["churned"] / g["customers"] * 100).round(2)
    if order:
        g = g.reindex(order)
    else:
        g = g.sort_values("churn_rate_pct", ascending=False)
    return g.reset_index()


def numeric_by_churn(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean / median of numeric columns split by churn status."""
    out = df.groupby("churn")[cols].agg(["mean", "median"]).round(2)
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    return out.reset_index()


def cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Cramer's V effect size and chi-square p-value for two categorical series."""
    table = pd.crosstab(a, b)
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))), float(p)


def association_with_churn(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rank categorical columns by strength of association with churn (Cramer's V)."""
    rows = []
    for col in columns:
        v, p = cramers_v(df[col], df["churn"])
        rows.append({"feature": col, "cramers_v": round(v, 3), "p_value": p})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def churn_heatmap_table(df: pd.DataFrame, row: str, col: str, row_order=None, col_order=None) -> pd.DataFrame:
    """Churn rate (%) pivot for two attributes, with cell counts for context."""
    rate = df.pivot_table(index=row, columns=col, values=config.TARGET, aggfunc="mean", observed=True) * 100
    if row_order:
        rate = rate.reindex(row_order)
    if col_order:
        rate = rate[[c for c in col_order if c in rate.columns]]
    return rate.round(1)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, name: str | None, out_dir: Path) -> None:
    if name:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name, bbox_inches="tight", facecolor="white")


def plot_churn_overview(df: pd.DataFrame, name: str | None = None, out_dir: Path = config.FIGURES_DIR):
    stats_ = overall_churn(df)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.pie(
        [stats_["retained"], stats_["churned"]],
        labels=[f"Retained\n{stats_['retained']:,}", f"Churned\n{stats_['churned']:,}"],
        colors=[COLOR_RETAINED, COLOR_CHURNED],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    ax.text(0, 0, f"{stats_['churn_rate_pct']:.1f}%\nchurn", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_title("Overall customer churn (7,043 customers)")
    _save(fig, name, out_dir)
    return fig


def plot_churn_rate_bar(
    df: pd.DataFrame,
    col: str,
    title: str,
    order: list[str] | None = None,
    xlabel: str | None = None,
    horizontal: bool = False,
    name: str | None = None,
    out_dir: Path = config.FIGURES_DIR,
    ax: plt.Axes | None = None,
):
    """Churn-rate bar chart; bars above the overall rate are red, others blue."""
    table = churn_rate_by(df, col, order)
    overall = df[config.TARGET].mean() * 100
    colors = [COLOR_CHURNED if r > overall else COLOR_RETAINED for r in table["churn_rate_pct"]]
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 4.6 if not horizontal else 4))
    else:
        fig = ax.figure
    labels = [f"{c}\n(n={n:,})" for c, n in zip(table[col], table["customers"])]
    if horizontal:
        bars = ax.barh(labels, table["churn_rate_pct"], color=colors)
        ax.axvline(overall, color="#333", ls="--", lw=1)
        ax.text(overall + 0.4, len(table) - 0.45, f"average {overall:.1f}%", fontsize=9, color="#333", va="bottom")
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
        ax.set_xlabel("Churn rate (%)")
        ax.set_ylabel(xlabel or "")
        for b, v in zip(bars, table["churn_rate_pct"]):
            ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=10)
        ax.set_xlim(0, max(table["churn_rate_pct"]) * 1.18)
    else:
        bars = ax.bar(labels, table["churn_rate_pct"], color=colors, width=0.62)
        ax.axhline(overall, color="#333", ls="--", lw=1)
        ax.text(len(table) - 0.45, overall + 0.8, f"average {overall:.1f}%", ha="right", fontsize=9, color="#333")
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Churn rate (%)")
        ax.set_xlabel(xlabel or "")
        for b, v in zip(bars, table["churn_rate_pct"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10)
        ax.set_ylim(0, max(table["churn_rate_pct"]) * 1.18)
    ax.set_title(title)
    if created:
        fig.tight_layout()
        _save(fig, name, out_dir)
    return fig


def plot_distribution_by_churn(
    df: pd.DataFrame, col: str, title: str, xlabel: str, bins: int = 30, name: str | None = None,
    out_dir: Path = config.FIGURES_DIR,
):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.histplot(
        data=df, x=col, hue="churn", hue_order=["No", "Yes"], palette=PALETTE, bins=bins,
        stat="density", common_norm=False, element="step", alpha=0.35, ax=ax,
    )
    for status, color in PALETTE.items():
        ax.axvline(df.loc[df["churn"] == status, col].median(), color=color, ls="--", lw=1.2)
    ax.set_title(title + "  (dashed = median)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of customers (density)")
    leg = ax.get_legend()
    leg.set_title("Churned")
    fig.tight_layout()
    _save(fig, name, out_dir)
    return fig


def plot_single_distribution(df: pd.DataFrame, col: str, title: str, xlabel: str, bins: int = 36,
                             name: str | None = None, out_dir: Path = config.FIGURES_DIR):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.histplot(df[col], bins=bins, color=COLOR_NEUTRAL, ax=ax)
    ax.axvline(df[col].mean(), color="#333", ls="--", lw=1.2)
    ax.text(df[col].mean(), ax.get_ylim()[1] * 0.95, f"  mean {df[col].mean():.1f}", fontsize=9, va="top")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Customers")
    fig.tight_layout()
    _save(fig, name, out_dir)
    return fig


def plot_charges_vs_tenure(df: pd.DataFrame, name: str | None = None, out_dir: Path = config.FIGURES_DIR):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for status in ["No", "Yes"]:
        sub = df[df["churn"] == status]
        ax.scatter(sub["tenure_months"], sub["monthly_charges"], s=9, alpha=0.35 if status == "No" else 0.55,
                   color=PALETTE[status], label="Churned" if status == "Yes" else "Retained")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly charges ($)")
    ax.set_title("Monthly charges vs tenure, by churn status")
    ax.legend(markerscale=2.5)
    fig.tight_layout()
    _save(fig, name, out_dir)
    return fig


def plot_heatmap(table: pd.DataFrame, title: str, xlabel: str, ylabel: str, name: str | None = None,
                 out_dir: Path = config.FIGURES_DIR, fmt: str = ".1f", cbar_label: str = "Churn rate (%)"):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.heatmap(table, annot=True, fmt=fmt, cmap="Reds", vmin=0, cbar_kws={"label": cbar_label},
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    _save(fig, name, out_dir)
    return fig


def plot_correlation(df: pd.DataFrame, name: str | None = None, out_dir: Path = config.FIGURES_DIR):
    cols = ["tenure_months", "monthly_charges", "total_charges", "num_addon_services", config.TARGET]
    corr = df[cols].corr().round(2)
    corr.index = corr.columns = ["Tenure", "Monthly charges", "Total charges", "Add-on services", "Churn"]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation of numeric features with churn")
    fig.tight_layout()
    _save(fig, name, out_dir)
    return fig


def plot_association_ranking(assoc: pd.DataFrame, name: str | None = None, out_dir: Path = config.FIGURES_DIR):
    top = assoc.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(top["feature"], top["cramers_v"], color=COLOR_RETAINED)
    ax.set_xlabel("Cramer's V (association with churn)")
    ax.set_title("Strongest categorical associations with churn")
    for y, v in enumerate(top["cramers_v"]):
        ax.text(v + 0.003, y, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    _save(fig, name, out_dir)
    return fig


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def internet_only(df: pd.DataFrame) -> pd.DataFrame:
    """Customers with an internet service (support add-ons only apply to them)."""
    return df[df["internet_service"] != "No internet"]


def run(out_dir: Path = config.FIGURES_DIR) -> dict:
    """Regenerate every EDA figure and write ``reports/eda_summary.json``."""
    df = load_features()
    net = internet_only(df)
    summary: dict = {"overall": overall_churn(df)}

    plot_churn_overview(df, "eda_01_churn_overview.png", out_dir)
    plt.close("all")

    specs = [
        ("contract", "Churn rate by contract type", config.CONTRACT_ORDER, "Contract", False, "eda_02_churn_by_contract.png"),
        ("tenure_group", "Churn rate by tenure group", config.TENURE_LABELS, "Tenure group", False, "eda_03_churn_by_tenure.png"),
        ("internet_service", "Churn rate by internet service", ["DSL", "Fiber optic", "No internet"], "Internet service", False, "eda_04_churn_by_internet.png"),
        ("payment_method", "Churn rate by payment method", None, "", True, "eda_05_churn_by_payment.png"),
        ("monthly_charge_band", "Churn rate by monthly charge band", config.CHARGE_LABELS, "Monthly charges", False, "eda_06_churn_by_charges.png"),
        ("senior_citizen", "Churn rate by senior-citizen status", ["No", "Yes"], "Senior citizen", False, "eda_07_churn_by_senior.png"),
    ]
    summary["churn_by"] = {}
    for col, title, order, xlabel, horiz, fname in specs:
        plot_churn_rate_bar(df, col, title, order, xlabel, horiz, fname, out_dir)
        summary["churn_by"][col] = churn_rate_by(df, col, order).to_dict("records")
        plt.close("all")

    # support services, internet customers only
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    support = {
        "tech_support": "Tech support", "online_security": "Online security", "online_backup": "Online backup",
        "device_protection": "Device protection", "streaming_tv": "Streaming TV", "streaming_movies": "Streaming movies",
    }
    summary["churn_by_service_internet_customers"] = {}
    for ax, (col, label) in zip(axes.ravel(), support.items()):
        plot_churn_rate_bar(net, col, label, ["No", "Yes"], "", False, ax=ax)
        ax.set_xlabel("")
        summary["churn_by_service_internet_customers"][col] = churn_rate_by(net, col, ["No", "Yes"]).to_dict("records")
    fig.suptitle("Churn rate by add-on service (internet customers only)", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, "eda_08_churn_by_services.png", out_dir)
    plt.close("all")

    plot_churn_rate_bar(df, "num_support_services", "Churn rate by number of support services held",
                        None, "Support services (security, backup, protection, tech support)", False,
                        "eda_09_churn_by_support_count.png", out_dir)
    plt.close("all")

    plot_single_distribution(df, "monthly_charges", "Monthly charges distribution", "Monthly charges ($)",
                             name="eda_10_monthly_charges_dist.png", out_dir=out_dir)
    plot_single_distribution(df, "tenure_months", "Customer tenure distribution", "Tenure (months)", 36,
                             name="eda_11_tenure_dist.png", out_dir=out_dir)
    plot_distribution_by_churn(df, "monthly_charges", "Monthly charges by churn status", "Monthly charges ($)",
                               name="eda_12_charges_by_churn.png", out_dir=out_dir)
    plot_distribution_by_churn(df, "tenure_months", "Tenure by churn status", "Tenure (months)", 36,
                               name="eda_13_tenure_by_churn.png", out_dir=out_dir)
    plot_charges_vs_tenure(df, "eda_14_charges_vs_tenure.png", out_dir)
    plt.close("all")

    hm = churn_heatmap_table(df, "tenure_group", "monthly_charge_band", config.TENURE_LABELS, config.CHARGE_LABELS)
    plot_heatmap(hm, "Churn rate (%) by tenure group and monthly charge band", "Monthly charges", "Tenure group",
                 "eda_15_heatmap_tenure_charges.png", out_dir)
    hm2 = churn_heatmap_table(df, "contract", "internet_service", config.CONTRACT_ORDER, ["DSL", "Fiber optic", "No internet"])
    plot_heatmap(hm2, "Churn rate (%) by contract and internet service", "Internet service", "Contract",
                 "eda_16_heatmap_contract_internet.png", out_dir)
    plot_correlation(df, "eda_17_correlation.png", out_dir)
    cat_cols = [c for c in feature_engineering.CATEGORICAL_FEATURES + ["tenure_group", "monthly_charge_band", "is_autopay",
                                                                        "has_support_service", "has_household"]]
    assoc = association_with_churn(df, cat_cols)
    plot_association_ranking(assoc, "eda_18_association_ranking.png", out_dir)
    plt.close("all")

    summary["numeric_by_churn"] = numeric_by_churn(
        df, ["tenure_months", "monthly_charges", "total_charges", "num_addon_services"]).to_dict("records")
    summary["heatmap_tenure_x_charges"] = hm.reset_index().to_dict("records")
    summary["heatmap_contract_x_internet"] = hm2.reset_index().to_dict("records")
    summary["association_cramers_v"] = assoc.to_dict("records")
    summary["correlation_with_churn"] = (
        df[["tenure_months", "monthly_charges", "total_charges", "num_addon_services", config.TARGET]]
        .corr()[config.TARGET].drop(config.TARGET).round(3).to_dict()
    )
    mw = stats.mannwhitneyu(df.loc[df.churn == "Yes", "monthly_charges"], df.loc[df.churn == "No", "monthly_charges"])
    mt = stats.mannwhitneyu(df.loc[df.churn == "Yes", "tenure_months"], df.loc[df.churn == "No", "tenure_months"])
    summary["mann_whitney_p"] = {"monthly_charges": float(mw.pvalue), "tenure_months": float(mt.pvalue)}

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = run()
    print(json.dumps(s["overall"], indent=2))
    print(f"Figures written to {config.FIGURES_DIR}")
