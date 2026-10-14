"""Forecast next quarter's house price growth for each NI council area.

Walk-forward backtest: for every quarter from 2015Q1 on, models are trained only on earlier
quarters, then predict that quarter. Compares simple baselines with two ML models.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nihpi

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
TEST_START = pd.Period("2015Q1", freq="Q")
# Train on a rolling 5-year window. With all history, the 2007-13 crash (Q1 avg -3.7%) was learned as
# seasonality and the model over-predicted Q2 spikes / Q1 falls. This window was picked with hindsight;
# compare_windows.py does the selection honestly and finds ridge does not beat simple baselines (see README).
WINDOW = 20
FEATURES = ["g_lag1", "g_lag2", "g_lag3", "g_lag4", "ni_lag1", "ni_lag2", "cpi_lag1", "q2", "q3", "q4"]


def build_dataset() -> pd.DataFrame:
    """One row per (council, quarter). target = log growth that quarter; features use earlier quarters only."""
    council, _ = nihpi.by_council()
    # one extra quarter at the end: its features are known, its target isn't -> the live forecast row
    quarters = pd.period_range(council.index[0], council.index[-1] + 1, freq="Q")
    g = np.log(council).diff().reindex(quarters)
    ni_g = np.log(nihpi.by_property_type()["Residential"]).diff().reindex(quarters)
    cpi_g = np.log(nihpi.cpi()).diff().reindex(quarters)

    frames = []
    for c in g.columns:
        d = pd.DataFrame({"council": c, "target": g[c]}, index=quarters)
        for k in range(1, 5):
            d[f"g_lag{k}"] = g[c].shift(k)
        d["ni_lag1"], d["ni_lag2"] = ni_g.shift(1), ni_g.shift(2)
        d["cpi_lag1"] = cpi_g.shift(1)
        for k in (2, 3, 4):  # quarter-of-year dummies for seasonality
            d[f"q{k}"] = (d.index.quarter == k).astype(int)
        frames.append(d)
    return pd.concat(frames).dropna(subset=FEATURES)


def models():
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        # deliberately small, untuned: tuning on the backtest would leak the test set
        "gbm": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
    }


def backtest(df: pd.DataFrame, features=FEATURES, window: int | None = WINDOW, only=None) -> pd.DataFrame:
    """window: train only on the last `window` quarters (None = all history). only: model names to fit (default all)."""
    known = df.dropna(subset=["target"])
    out = []
    for q in sorted(known.index[known.index >= TEST_START].unique()):
        train, test = known[known.index < q], known[known.index == q]
        if window:
            train = train[train.index >= q - window]
        assert train.index.max() < q  # no future data in training
        preds = pd.DataFrame({
            "council": test["council"],
            "actual": test["target"],
            "no_change": 0.0,                                                  # prices stay flat
            "repeat_last": test["g_lag1"],                                     # same growth as last quarter
            "avg_last_year": test[["g_lag1", "g_lag2", "g_lag3", "g_lag4"]].mean(axis=1),
        })
        for name, m in models().items():
            if only is None or name in only:
                preds[name] = m.fit(train[features], train["target"]).predict(test[features])
        out.append(preds)
    return pd.concat(out)


def live_forecast(df: pd.DataFrame, window: int = WINDOW) -> pd.Series:
    """Ridge forecast (log growth) per council for the next, unpublished quarter, trained on the last `window` quarters."""
    nxt = df[df["target"].isna()]
    known = df.dropna(subset=["target"])
    known = known[known.index >= nxt.index[0] - window]
    model = models()["ridge"].fit(known[FEATURES], known["target"])
    return pd.Series(model.predict(nxt[FEATURES]), index=nxt["council"], name=str(nxt.index[0]))


def abs_errors(bt: pd.DataFrame) -> pd.DataFrame:
    """Absolute forecast error per row and method, in percentage points of quarterly growth."""
    methods = bt.columns.drop(["council", "actual"])
    return np.exp(bt[methods]).sub(np.exp(bt["actual"]), axis=0).abs() * 100


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error in percentage points of quarterly growth, and skill vs the no-change baseline."""
    res = abs_errors(bt).mean().rename("MAE_pp").to_frame().sort_values("MAE_pp")
    res["vs_no_change_%"] = ((1 - res["MAE_pp"] / res.loc["no_change", "MAE_pp"]) * 100).round(1)
    return res.round(3)


def main() -> None:
    df = build_dataset()
    bt = backtest(df)
    bt.to_csv(OUT / "backtest_predictions.csv")

    print(f"Backtest: {bt.index.min()} to {bt.index.max()}, {bt.index.nunique()} quarters x 11 councils = {len(bt)} forecasts\n")
    print("== Overall (lower MAE is better)\n", score(bt), "\n")
    print("== Since 2020 (Covid, rate rises)\n", score(bt[bt.index >= pd.Period("2020Q1", freq="Q")]), "\n")
    per_council = abs_errors(bt)[["ridge", "no_change"]].groupby(bt["council"]).mean().round(2)
    print("== Per council, MAE pp\n", per_council.sort_values("ridge"), "\n")

    fc = live_forecast(df)
    print(f"== Forecast growth for {fc.name} (%), ridge\n", ((np.exp(fc) - 1) * 100).round(2).sort_values().to_string())

    # Chart: actual vs ridge vs repeat-last, NI average across councils
    avg = bt.groupby(level=0)[["actual", "ridge", "repeat_last"]].mean().apply(lambda s: (np.exp(s) - 1) * 100)
    ax = avg.to_timestamp().plot(figsize=(10, 5), lw=2, style=["-", "--", ":"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Quarterly house price growth, NI council average: actual vs forecast", ylabel="% growth", xlabel="")
    ax.grid(alpha=0.3)
    ax.legend(["Actual", "Forecast (ridge)", "Repeat last quarter"], frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "forecast_backtest.png", dpi=150)
    print(f"\nSaved {OUT / 'forecast_backtest.png'} and backtest_predictions.csv")


if __name__ == "__main__":
    main()
