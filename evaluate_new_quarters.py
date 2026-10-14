"""Final exam: score forecasts on quarters published after all modelling choices were made (2025Q3 onwards).

1. Real-time test: forecasts for 2025Q3 made from the data as published then (vintage snapshot of the
   OpenDataNI release ending 2025Q2), scored against the actual growth in the current release.
2. Walk-forward over every new quarter on current data, comparing the ridge variants with baselines.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import forecast as f
import nihpi

VINTAGE = Path(__file__).parent / "data" / "vintages" / "2025Q2_opendatani"
NEW_FROM = pd.Period("2025Q3", freq="Q")
WINDOWS = {"ridge_live_window": f.WINDOW, "ridge_3y": 12}  # forecast.py's window, and compare_windows.py's pick


def live_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Next-quarter forecast per council from each method."""
    nxt = df[df["target"].isna()].set_index("council")
    out = pd.DataFrame({"no_change": 0.0, "avg_last_year": nxt[["g_lag1", "g_lag2", "g_lag3", "g_lag4"]].mean(axis=1)})
    for name, window in WINDOWS.items():
        out[name] = f.live_forecast(df, window)
    return out


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    current = f.build_dataset()

    # 1. Real-time: what we would actually have forecast in 2025, with the data available then
    raw, nihpi.RAW = nihpi.RAW, VINTAGE
    try:
        then = f.build_dataset()
    finally:
        nihpi.RAW = raw
    q = then[then["target"].isna()].index[0]
    fc = live_forecasts(then)
    actual = current[current.index == q].set_index("council")["target"]
    pct = lambda s: (np.exp(s) - 1) * 100
    table = pct(fc).assign(actual=pct(actual)).round(2).sort_values("actual")
    print(f"== Real-time test: forecasts for {q} from data published up to {q - 1} (% growth)\n", table, "\n")
    mae = (table.drop(columns="actual").sub(table["actual"], axis=0)).abs().mean().sort_values().round(2)
    print(f"MAE (pp), {len(table)} forecasts:\n", mae.to_string(), "\n")

    # 2. Walk-forward over every new quarter (current data; each quarter trained only on earlier ones)
    res = {}
    for name, window in WINDOWS.items():
        bt = f.backtest(current, window=window, only=["ridge"])
        bt = bt[bt.index >= NEW_FROM]
        res[name] = f.score(bt).loc["ridge", "MAE_pp"]
    scores = f.score(bt).drop(index="ridge")["MAE_pp"]  # baselines don't depend on the window
    allres = pd.concat([pd.Series(res), scores]).sort_values()
    print(f"== Walk-forward on new quarters {NEW_FROM} to {bt.index.max()} ({len(bt)} forecasts), MAE pp\n",
          allres.round(3).to_string())


if __name__ == "__main__":
    main()
