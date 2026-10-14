"""First look at the NI House Price Index: the 2007 crash and the recovery."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import nihpi

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def crash_and_recovery(idx: pd.DataFrame) -> pd.DataFrame:
    """Per column: pre-2010 peak, post-peak trough, fall, rise since trough, and today vs old peak."""
    rows = {}
    for col, s in idx.items():
        peak_q = s[:"2009Q4"].idxmax()
        trough_q = s[peak_q:].idxmin()
        back = s[trough_q:][s[trough_q:] >= s[peak_q]]
        rows[col] = {
            "peak": str(peak_q),
            "trough": str(trough_q),
            "fall_%": round((s[trough_q] / s[peak_q] - 1) * 100, 1),
            "rise_since_trough_%": round((s.iloc[-1] / s[trough_q] - 1) * 100, 1),
            "latest_vs_peak_%": round((s.iloc[-1] / s[peak_q] - 1) * 100, 1),
            "back_above_peak": str(back.index[0]) if len(back) else "not yet",
        }
    return pd.DataFrame(rows).T


types = nihpi.by_property_type()
council_idx, council_price = nihpi.by_council()
ur_idx, _ = nihpi.by_urban_rural()

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)
print(f"Data: {types.index[0]} to {types.index[-1]} ({len(types)} quarters)\n")
print("== By property type\n", crash_and_recovery(types), "\n")
print("== Urban vs rural\n", crash_and_recovery(ur_idx), "\n")
council_table = crash_and_recovery(council_idx).sort_values("latest_vs_peak_%")
print("== By council area\n", council_table, "\n")
latest = council_price.iloc[-1].sort_values()
print(f"== Standardised price, {council_price.index[-1]}\n", latest.round(-3).astype(int).to_string(), "\n")

# Chart 1: index by property type
ax = types.drop(columns="Residential").to_timestamp().plot(figsize=(10, 5), lw=1.8)
types["Residential"].to_timestamp().plot(ax=ax, color="black", lw=2.5, ls="--", label="All NI")
ax.set(title="NI House Price Index by property type", ylabel="Index", xlabel="")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "index_by_property_type.png", dpi=150)

# Chart 2: where each council stands today relative to its pre-crash peak
fig, ax = plt.subplots(figsize=(9, 5))
vals = council_table["latest_vs_peak_%"].astype(float)
ax.barh(vals.index, vals, color=["#c0392b" if v < 0 else "#2e86c1" for v in vals])
ax.axvline(0, color="black", lw=0.8)
ax.set(title=f"House prices in {council_idx.index[-1]} vs pre-crash peak", xlabel="% above (+) / below (-) peak")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "councils_vs_peak.png", dpi=150)

# Real terms: same analysis after removing UK CPI inflation, in prices of the latest quarter
base = str(types.index[-1])
real_types = nihpi.to_real(types, base)
real_council = nihpi.to_real(council_idx, base)
print(f"== REAL TERMS (CPI-adjusted to {base} prices): by property type\n", crash_and_recovery(real_types), "\n")
real_council_table = crash_and_recovery(real_council).sort_values("latest_vs_peak_%")
print("== REAL TERMS: by council area\n", real_council_table, "\n")

# Chart 3: nominal vs real, all NI
fig, ax = plt.subplots(figsize=(10, 5))
types["Residential"].to_timestamp().plot(ax=ax, lw=2.2, label="Nominal (cash prices)")
real_types["Residential"].to_timestamp().plot(ax=ax, lw=2.2, label=f"Real ({base} prices, CPI-adjusted)")
ax.set(title="NI house prices: nominal vs real", ylabel="Index", xlabel="")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "nominal_vs_real.png", dpi=150)

print(f"Charts saved to {OUT}")
