"""Choose the training window honestly: select on 2015-2019 backtest quarters, report on untouched 2020+.

Each variant is a full walk-forward backtest (see forecast.backtest). Only the selection-period error
is used to pick the winner; the holdout column is reported after the choice is made.
"""
import pandas as pd

import forecast as f

SELECT_END = pd.Period("2019Q4", freq="Q")
NO_SEASON = [c for c in f.FEATURES if c not in ("q2", "q3", "q4")]
VARIANTS = {  # name -> (features, window in quarters; None = all history)
    "all history": (f.FEATURES, None),
    "all history, no season dummies": (NO_SEASON, None),
    "8-year window": (f.FEATURES, 32),
    "5-year window": (f.FEATURES, 20),
    "3-year window": (f.FEATURES, 12),
}


def main() -> None:
    df = f.build_dataset()
    rows, bts = {}, {}
    for name, (feats, window) in VARIANTS.items():
        bt = bts[name] = f.backtest(df, feats, window, only=["ridge"])
        sel, hold = bt[bt.index <= SELECT_END], bt[bt.index > SELECT_END]
        rows[name] = {"select_MAE": f.score(sel).loc["ridge", "MAE_pp"], "holdout_MAE": f.score(hold).loc["ridge", "MAE_pp"]}
    res = pd.DataFrame(rows).T
    chosen = res["select_MAE"].idxmin()  # decided on 2015-2019 only

    hold = bts[chosen][bts[chosen].index > SELECT_END]
    print(f"Ridge MAE (pp of quarterly growth). Selection: {f.TEST_START}-{SELECT_END}; holdout: after {SELECT_END}\n")
    print(res.sort_values("select_MAE").to_string(), "\n")
    print(f"Chosen on selection period: {chosen}\n")
    print(f"== Holdout ({hold.index.min()} to {hold.index.max()}, {len(hold)} forecasts), chosen variant vs baselines\n",
          f.score(hold))


if __name__ == "__main__":
    main()
