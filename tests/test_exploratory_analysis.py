"""Tests for src/exploratory_analysis.py - table logic checked against the raw source file."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, exploratory_analysis as eda


def test_overall_churn_matches_raw(raw_df, features_df):
    res = eda.overall_churn(features_df)
    assert res["customers"] == len(raw_df)
    assert res["churned"] == (raw_df["Churn"] == "Yes").sum()
    assert res["churn_rate_pct"] == pytest.approx((raw_df["Churn"] == "Yes").mean() * 100, abs=0.01)


def test_churn_rate_by_contract_matches_raw(raw_df, features_df):
    table = eda.churn_rate_by(features_df, "contract", config.CONTRACT_ORDER).set_index("contract")
    raw_rate = raw_df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
    for contract in config.CONTRACT_ORDER:
        assert table.loc[contract, "churn_rate_pct"] == pytest.approx(raw_rate[contract], abs=0.01)
    assert table["customers"].sum() == 7043


def test_churn_rate_by_small_frame():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "b"], config.TARGET: [1, 0, 0, 0, 1]})
    table = eda.churn_rate_by(df, "g").set_index("g")
    assert table.loc["a", "churn_rate_pct"] == 50.0
    assert table.loc["b", "churn_rate_pct"] == pytest.approx(33.33)


def test_cramers_v_bounds():
    perfect = pd.Series(["x", "y"] * 50)
    assert eda.cramers_v(perfect, perfect)[0] == pytest.approx(1.0)
    independent = pd.Series(["x", "x", "y", "y"] * 25)
    other = pd.Series(["u", "v", "u", "v"] * 25)
    assert eda.cramers_v(independent, other)[0] == pytest.approx(0.0, abs=1e-9)


def test_contract_is_strongest_categorical_driver(features_df):
    assoc = eda.association_with_churn(features_df, ["contract", "gender", "partner", "payment_method"])
    assert assoc.iloc[0]["feature"] == "contract"
    gender_v = assoc.set_index("feature").loc["gender", "cramers_v"]
    assert gender_v < 0.02  # gender is essentially unrelated to churn


def test_internet_only_filter(features_df):
    net = eda.internet_only(features_df)
    assert (net["internet_service"] != "No internet").all()
    assert len(net) == 7043 - 1526


def test_plot_functions_return_figures(features_df, tmp_path):
    fig = eda.plot_churn_rate_bar(features_df, "contract", "t", config.CONTRACT_ORDER, name="x.png", out_dir=tmp_path)
    assert (tmp_path / "x.png").stat().st_size > 1000
    assert len(fig.axes) == 1
    import matplotlib.pyplot as plt
    plt.close("all")
