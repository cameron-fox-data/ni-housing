"""Guard against look-ahead leakage: every feature must come from quarters before the target quarter."""
import numpy as np
import pandas as pd

import nihpi
from forecast import FEATURES, build_dataset


def test_features_only_use_the_past():
    df = build_dataset()
    council, _ = nihpi.by_council()
    g = np.log(council).diff()
    ni_g = np.log(nihpi.by_property_type()["Residential"]).diff()
    cpi_g = np.log(nihpi.cpi()).diff()

    q = pd.Period("2020Q3", freq="Q")
    row = df[(df.index == q) & (df["council"] == "Belfast")].iloc[0]
    assert np.isclose(row["target"], g.loc[q, "Belfast"])
    for k in range(1, 5):
        assert np.isclose(row[f"g_lag{k}"], g.loc[q - k, "Belfast"])
    assert np.isclose(row["ni_lag1"], ni_g[q - 1]) and np.isclose(row["ni_lag2"], ni_g[q - 2])
    assert np.isclose(row["cpi_lag1"], cpi_g[q - 1])
    assert (row["q2"], row["q3"], row["q4"]) == (0, 1, 0)


def test_live_row_is_the_next_unpublished_quarter():
    df = build_dataset()
    council, _ = nihpi.by_council()
    live = df[df["target"].isna()]
    assert set(live.index) == {council.index[-1] + 1}
    assert len(live) == council.shape[1] and not live[FEATURES].isna().any().any()
