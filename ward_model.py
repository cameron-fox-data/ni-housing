"""What drives house prices across NI's wards, and which wards are priced above or below what they offer?

Target: log median sale price (2023-25, in 2025 money). Features: Census 2021 + location + mix of homes sold.
Evaluated two ways: random 10-fold CV, and leave-one-council-out (a council the model has never seen).
Out-of-fold predictions give each ward a "price vs model" gap; wards with no published price get an estimate.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import wards

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
# Shares that sum to ~100% leave one category out as the reference (tenure: owned outright; homes sold:
# semi-detached). Including all of them made ridge assign huge offsetting weights and inflated their importance.
FEATURES = [
    "km_to_belfast", "log_density", "pct_over_65", "pct_mortgage", "pct_social_rented", "pct_private_rented",
    "pct_detached_homes", "pct_flats", "pct_no_car", "pct_degree", "pct_no_qualifications", "pct_unemployed",
    "pct_social_grade_ab", "sold_detached", "sold_terrace",
]


def models():
    return {
        # imputer: one ward had no sales at all, so its sold_* mix is missing
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "gbm": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0),
    }


def cross_validate(df: pd.DataFrame, splitter, groups=None) -> pd.DataFrame:
    """Out-of-fold log-price predictions for every model plus a predict-the-mean baseline."""
    X, y = df[FEATURES], np.log(df["price"])
    oof = pd.DataFrame(index=df.index, columns=["mean", *models()], dtype=float)
    for train, test in splitter.split(X, y, groups):
        oof.iloc[test, 0] = y.iloc[train].mean()
        for name, m in models().items():
            oof.loc[oof.index[test], name] = m.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
    return oof


def pct_errors(oof: pd.DataFrame, y_log: pd.Series) -> pd.DataFrame:
    """Absolute error as % of the actual price."""
    return (np.exp(oof.sub(y_log, axis=0)) - 1).abs() * 100


def draw_map(ax, geo: list, values: pd.Series, cmap, norm, title: str) -> None:
    polys, colours = [], []
    for f in geo:
        g = f["geometry"]
        rings = [p[0] for p in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]])]
        v = values.get(f["properties"]["WardCode"], np.nan)
        for r in rings:
            polys.append(np.asarray(r))
            colours.append(v)
    pc = PolyCollection(polys, array=np.asarray(colours, dtype=float), cmap=cmap, norm=norm,
                        edgecolor="white", linewidth=0.1)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.radians(54.6)))
    ax.set_axis_off()
    ax.set_title(title)
    plt.colorbar(pc, ax=ax, shrink=0.6)


def main() -> None:
    pd.set_option("display.width", 160)
    df = pd.read_csv(wards.OUT, index_col=0)
    priced = df[df["price"].notna()]
    y_log = np.log(priced["price"])
    print(f"{len(priced)} wards with a price, {len(df) - len(priced)} without\n")

    random_oof = cross_validate(priced, KFold(10, shuffle=True, random_state=0))
    council_oof = cross_validate(priced, GroupKFold(n_splits=11), groups=priced["lgd"])
    res = pd.DataFrame({
        "random 10-fold": pct_errors(random_oof, y_log).median(),
        "unseen council": pct_errors(council_oof, y_log).median(),
    }).round(1)
    print("== Median absolute error, % of actual price (lower is better)\n", res, "\n")

    best = "ridge"  # ties gbm within 0.1pp in both CV schemes; simpler and its coefficients are explainable
    # Which features matter: permutation importance on held-out folds (drop in accuracy when a feature is shuffled)
    X = priced[FEATURES]
    imp = []
    for train, test in KFold(5, shuffle=True, random_state=0).split(X):
        m = models()[best].fit(X.iloc[train], y_log.iloc[train])
        r = permutation_importance(m, X.iloc[test], y_log.iloc[test], n_repeats=10, random_state=0,
                                   scoring="neg_mean_absolute_error")
        imp.append(r.importances_mean)
    importance = pd.Series(np.mean(imp, axis=0), index=FEATURES).sort_values(ascending=False)
    ridge = models()["ridge"].fit(X, y_log)
    direction = pd.Series(ridge[-1].coef_, index=FEATURES)  # sign: does price rise or fall with this feature?
    print(f"== What drives prices ({best} permutation importance; direction from ridge)")
    print(pd.DataFrame({"importance": importance.round(4),
                        "direction": np.sign(direction[importance.index]).map({1: "+", -1: "-", 0: "0"})}).head(10), "\n")

    # Price vs model: random-CV out-of-fold prediction = what comparable wards elsewhere suggest
    out = df[["lgd", "ward", "price", "price_years", "km_to_belfast"]].copy()
    out["predicted"] = np.exp(random_oof[best])
    unpriced = df[df["price"].isna()]
    out.loc[unpriced.index, "predicted"] = np.exp(models()[best].fit(X, y_log).predict(unpriced[FEATURES]))
    out["gap_%"] = (out["price"] / out["predicted"] - 1) * 100
    out.round(1).to_csv(OUT / "ward_predictions.csv")
    cols = ["lgd", "ward", "price", "predicted", "gap_%"]
    print("== Priced furthest ABOVE what their features predict\n", out.nlargest(8, "gap_%")[cols].round(0).to_string(), "\n")
    print("== Priced furthest BELOW what their features predict\n", out.nsmallest(8, "gap_%")[cols].round(0).to_string(), "\n")
    print(f"== Estimated prices for the {len(unpriced)} wards with too few sales to publish (first 5)\n",
          out.loc[unpriced.index, ["lgd", "ward", "predicted"]].round(-3).head().to_string())

    # Charts: map of price (published or estimated) and of the gap vs the model
    geo = json.loads((wards.GEO_DIR / "wards_2012.geojson").read_text(encoding="utf-8"))["features"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    price_or_est = out["price"].fillna(out["predicted"]) / 1000
    draw_map(axes[0], geo, price_or_est, "viridis", plt.Normalize(100, 350), "Median house price 2023-25, £k (estimated where unpublished)")
    draw_map(axes[1], geo, out["gap_%"], "RdBu_r", plt.Normalize(-30, 30), f"Price vs model ({best}), %: red = above, blue = below")
    plt.tight_layout()
    plt.savefig(OUT / "ward_maps.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(out["predicted"] / 1000, out["price"] / 1000, s=10, alpha=0.6)
    lim = [80, 520]
    ax.plot(lim, lim, color="black", lw=0.8)
    ax.set(xlim=lim, ylim=lim, xlabel="Predicted £k (out-of-fold)", ylabel="Actual median £k",
           title=f"Ward prices: actual vs predicted ({best})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "ward_actual_vs_predicted.png", dpi=150)
    print(f"\nSaved ward_predictions.csv, ward_maps.png, ward_actual_vs_predicted.png to {OUT}")


if __name__ == "__main__":
    main()
