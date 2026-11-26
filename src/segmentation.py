"""K-Means customer segmentation (Phase 2, enhancement 1).

Customers are clustered on *behaviour and value* attributes only. The churn outcome is
deliberately **not** an input, so a cluster's churn rate is an independent observation
about that group rather than something the algorithm was told to find.

    python -m src.segmentation
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config, model_training as mt
from .exploratory_analysis import set_style

set_style()

CLUSTER_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "num_addon_services",
    "contract_ordinal",
    "is_autopay_flag",
]
K_RANGE = range(2, 9)
MIN_K = 3  # two groups are too coarse to drive different retention actions
STABILITY_SEEDS = range(1, 11)


def build_cluster_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric clustering inputs (unscaled). Contains no churn information."""
    return pd.DataFrame(
        {
            "tenure_months": df["tenure_months"],
            "monthly_charges": df["monthly_charges"],
            "num_addon_services": df["num_addon_services"],
            "contract_ordinal": df["contract"].map({c: i for i, c in enumerate(config.CONTRACT_ORDER)}),
            "is_autopay_flag": (df["is_autopay"] == "Yes").astype(int),
        },
        index=df.index,
    )[CLUSTER_FEATURES].astype(float)


def scale(inputs: pd.DataFrame) -> np.ndarray:
    """Standardise so that no feature dominates the distance only because of its units."""
    return StandardScaler().fit_transform(inputs)


def fit_kmeans(Z: np.ndarray, k: int, seed: int = config.RANDOM_STATE) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)


def evaluate_k(Z: np.ndarray, k_range=K_RANGE, seeds=STABILITY_SEEDS) -> pd.DataFrame:
    """Inertia, silhouette and seed stability (mean adjusted Rand index) for each k.

    Stability fits the model on random 80 % subsamples and compares the resulting assignment
    of *all* customers with the full-data solution; 1.0 means the same groups every time.
    """
    rows = []
    for k in k_range:
        base = fit_kmeans(Z, k)
        aris = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(Z), size=int(0.8 * len(Z)), replace=False)
            other = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z[idx])
            aris.append(adjusted_rand_score(base.labels_, other.predict(Z)))
        rows.append(
            {
                "k": k,
                "inertia": float(base.inertia_),
                "silhouette": float(silhouette_score(Z, base.labels_)),
                "stability_ari_mean": float(np.mean(aris)),
                "stability_ari_min": float(np.min(aris)),
            }
        )
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame, min_k: int = MIN_K, min_stability: float = 0.9) -> int:
    """Highest silhouette among stable solutions with at least ``min_k`` clusters."""
    eligible = scores[(scores["k"] >= min_k) & (scores["stability_ari_mean"] >= min_stability)]
    if eligible.empty:
        eligible = scores[scores["k"] >= min_k]
    return int(eligible.sort_values("silhouette", ascending=False).iloc[0]["k"])


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Size, behaviour, value and (independently observed) churn of each cluster."""
    d = df.assign(cluster=labels)
    g = d.groupby("cluster")
    prof = pd.DataFrame(
        {
            "customers": g.size(),
            "churn_rate_pct": g[config.TARGET].mean() * 100,
            "avg_tenure_months": g["tenure_months"].mean(),
            "avg_monthly_charges": g["monthly_charges"].mean(),
            "avg_addon_services": g["num_addon_services"].mean(),
            "pct_month_to_month": g["contract"].apply(lambda s: (s == "Month-to-month").mean() * 100),
            "pct_two_year": g["contract"].apply(lambda s: (s == "Two year").mean() * 100),
            "pct_autopay": g["is_autopay"].apply(lambda s: (s == "Yes").mean() * 100),
            "pct_fiber": g["internet_service"].apply(lambda s: (s == "Fiber optic").mean() * 100),
        }
    )
    prof["share_of_customers_pct"] = prof["customers"] / len(d) * 100
    active = d[d[config.TARGET] == 0].groupby("cluster")["monthly_charges"].sum()
    prof["active_mrr"] = active.reindex(prof.index).fillna(0)
    prof["share_of_churners_pct"] = g[config.TARGET].sum() / d[config.TARGET].sum() * 100
    return prof.round(2).reset_index()


def name_cluster(row: pd.Series) -> str:
    """Rule-based, reproducible description built from the cluster's own averages."""
    tenure = "New" if row["avg_tenure_months"] < 20 else "Established" if row["avg_tenure_months"] < 45 else "Long-standing"
    charges = "low-spend" if row["avg_monthly_charges"] < 45 else "mid-spend" if row["avg_monthly_charges"] < 70 else "high-spend"
    if row["pct_month_to_month"] >= 60:
        contract = "month-to-month"
    elif row["pct_month_to_month"] <= 25:
        contract = "contracted"
    else:
        contract = "mixed-contract"
    pay = ", auto-pay" if row["pct_autopay"] >= 80 else ", manual pay" if row["pct_autopay"] <= 20 else ""
    return f"{tenure} {charges}, {contract}{pay}"


def add_names(profile: pd.DataFrame) -> pd.DataFrame:
    out = profile.copy()
    names = out.apply(name_cluster, axis=1)
    # keep names unique if two clusters share the same description
    dup = names.groupby(names).cumcount()
    out["cluster_name"] = np.where(dup > 0, names + " (" + (dup + 1).astype(str) + ")", names)
    return out


def segment(df: pd.DataFrame, k: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(assignments, profile, k_scores)``.

    Cluster ids are renumbered by descending churn rate (0 = highest churn) so they are stable
    and easy to read.
    """
    Z = scale(build_cluster_inputs(df))
    scores = evaluate_k(Z)
    k = k or choose_k(scores)
    labels = fit_kmeans(Z, k).labels_
    order = pd.Series(df[config.TARGET].to_numpy()).groupby(labels).mean().sort_values(ascending=False).index
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[v] for v in labels])
    profile = add_names(profile_clusters(df, labels))
    names = profile.set_index("cluster")["cluster_name"]
    assignments = pd.DataFrame(
        {config.ID_COL: df[config.ID_COL], "cluster": labels, "cluster_name": names.reindex(labels).to_numpy()}
    )
    return assignments, profile, scores


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def plot_k_selection(scores: pd.DataFrame, k: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(
        axes, ["inertia", "silhouette", "stability_ari_mean"],
        ["Inertia (elbow)", "Silhouette score", "Stability (adjusted Rand index)"],
    ):
        ax.plot(scores["k"], scores[col], marker="o", color="#3A6EA5")
        ax.axvline(k, ls="--", color="#D1495B")
        ax.set_title(title)
        ax.set_xlabel("Number of clusters (k)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "seg_01_k_selection.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_profiles(profile: pd.DataFrame, overall: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.8 * len(profile) + 2), sharey=True)
    labels = profile["cluster_name"]
    colors = ["#D1495B" if r > overall else "#3A6EA5" for r in profile["churn_rate_pct"]]
    for ax, col, title, fmt in [
        (axes[0], "churn_rate_pct", "Churn rate (%)", "{:.1f}%"),
        (axes[1], "customers", "Customers", "{:,.0f}"),
    ]:
        ax.barh(labels, profile[col], color=colors)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(profile[col]):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=10)
        ax.set_xlim(0, profile[col].max() * 1.2)
    axes[0].invert_yaxis()
    fig.suptitle("K-Means segments (churn was not used to build them)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "seg_02_cluster_profiles.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pca(df: pd.DataFrame, assignments: pd.DataFrame) -> float:
    Z = scale(build_cluster_inputs(df))
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    xy = pca.fit_transform(Z)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=assignments["cluster_name"], s=9, alpha=0.5, linewidth=0, ax=ax)
    explained = float(pca.explained_variance_ratio_.sum())
    ax.set_title(f"Segments projected to 2 principal components ({explained:.0%} of variance)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9, markerscale=2)
    fig.savefig(config.FIGURES_DIR / "seg_03_pca_projection.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return explained


def write_report(profile: pd.DataFrame, scores: pd.DataFrame, k: int, explained: float, overall: float) -> None:
    def table(d: pd.DataFrame) -> str:
        head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
        return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False))

    view = pd.DataFrame(
        {
            "Segment": profile["cluster_name"],
            "Customers": profile["customers"].map("{:,}".format),
            "Churn rate": profile["churn_rate_pct"].map("{:.1f}%".format),
            "Share of churners": profile["share_of_churners_pct"].map("{:.1f}%".format),
            "Avg tenure": profile["avg_tenure_months"].map("{:.1f}".format),
            "Avg monthly": profile["avg_monthly_charges"].map("${:.2f}".format),
            "Add-ons": profile["avg_addon_services"].map("{:.1f}".format),
            "Month-to-month": profile["pct_month_to_month"].map("{:.0f}%".format),
            "Auto-pay": profile["pct_autopay"].map("{:.0f}%".format),
            "Active MRR": profile["active_mrr"].map("${:,.0f}".format),
        }
    )
    sc = scores.assign(
        inertia=scores["inertia"].map("{:,.0f}".format),
        silhouette=scores["silhouette"].map("{:.3f}".format),
        stability_ari_mean=scores["stability_ari_mean"].map("{:.3f}".format),
        stability_ari_min=scores["stability_ari_min"].map("{:.3f}".format),
    )
    chosen = scores.set_index("k").loc[k]
    top, low = profile.iloc[0], profile.iloc[-1]
    strength = "weak-to-moderate" if chosen["silhouette"] < 0.5 else "reasonably clear"
    text = f"""# Customer segmentation (K-Means)

_Generated by `python -m src.segmentation`; every number below was computed from the data._

## Method

* Inputs (standardised): tenure, monthly charges, number of add-on services, contract length (0/1/2) and automatic payment.
* **Churn is not an input.** Churn rates per segment are observed afterwards.
* k was evaluated from {scores['k'].min()} to {scores['k'].max()}. Selection rule: highest silhouette among solutions with at least {MIN_K} clusters whose
  stability (mean adjusted Rand index across 10 refits on 80% subsamples) is at least 0.9.

{table(sc)}

**Chosen k = {k}** (silhouette {chosen['silhouette']:.3f}, stability {chosen['stability_ari_mean']:.3f}). A silhouette of this size indicates **{strength}**
separation: the segments are useful summaries, not sharply distinct natural groups.

![k selection](figures/seg_01_k_selection.png)

## Segments (ordered by churn rate; overall {overall:.1f}%)

{table(view)}

![Segment profiles](figures/seg_02_cluster_profiles.png)
![PCA projection](figures/seg_03_pca_projection.png)

The two-component projection retains {explained:.0%} of the variance, so overlap in the picture overstates or understates separation to some extent.

## What this suggests

* **{top['cluster_name']}** has the highest churn ({top['churn_rate_pct']:.1f}%) and holds {top['share_of_churners_pct']:.0f}% of all churners
  ({int(top['customers']):,} customers, ${top['active_mrr']:,.0f} active MRR): the first group to target.
* **{low['cluster_name']}** has the lowest churn ({low['churn_rate_pct']:.1f}%): protect it, and avoid spending retention discounts here.

## Limitations

* K-Means assumes roughly round, similar-sized clusters and treats the ordinal contract and binary auto-pay inputs as numbers.
* Results depend on the chosen inputs; other reasonable inputs would give different segments.
* Segments describe a single snapshot and show association only.
"""
    (config.REPORTS_DIR / "segmentation_results.md").write_text(text, encoding="utf-8")


def main() -> None:
    df = mt.load_modelling_data()
    assignments, profile, scores = segment(df)
    k = len(profile)
    overall = df[config.TARGET].mean() * 100

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_k_selection(scores, k)
    plot_profiles(profile, overall)
    explained = plot_pca(df, assignments)
    write_report(profile, scores, k, explained, overall)

    config.DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(config.DASHBOARD_DATA_DIR / "customer_clusters.csv", index=False)
    profile.to_csv(config.DASHBOARD_DATA_DIR / "cluster_profiles.csv", index=False)
    (config.REPORTS_DIR / "segmentation_summary.json").write_text(
        json.dumps({"k": k, "scores": scores.to_dict("records"), "profile": profile.to_dict("records")}, indent=2),
        encoding="utf-8",
    )
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(scores.round(3).to_string(index=False))
    print(f"\nChosen k = {k}")
    print(profile.to_string(index=False))


if __name__ == "__main__":
    main()
