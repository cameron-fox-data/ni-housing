"""Load the NI House Price Index CSVs as tidy quarterly DataFrames."""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"


def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / name)
    # "Q1 2005" -> Period 2005Q1
    q, y = df.pop("Quarter_Year").str.split(" ", expand=True).T.values
    df.index = pd.PeriodIndex([f"{yy}{qq}" for qq, yy in zip(q, y)], freq="Q", name="quarter")
    assert df.index.is_monotonic_increasing and df.index.is_unique, f"{name}: bad quarters"
    # a gap would make diff()/shift() silently treat a 6-month change as a 3-month one
    assert len(df) == (df.index[-1] - df.index[0]).n + 1, f"{name}: missing quarters"
    assert not df.isna().any().any(), f"{name}: missing values"
    return df


def by_property_type() -> pd.DataFrame:
    df = _load("hpi_by_property_type.csv")
    df.columns = [c.removeprefix("NI_").removesuffix("_Price_Index").removesuffix("_Property") for c in df.columns]
    return df  # Detached, SemiDetached, Terrace, Apartment, Residential (= all NI)


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split wide '<Area>_HPI' / '<Area>_Standardised_Price' columns into (index, price) frames."""
    hpi = df.filter(regex="_HPI$").rename(columns=lambda c: c.removesuffix("_HPI").replace("_", " "))
    price = df.filter(regex="_Standardised_Price$").rename(
        columns=lambda c: c.removesuffix("_Standardised_Price").replace("_", " "))
    return hpi, price


def by_council() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(index, standardised price) per council area. Prices only comparable within a calendar year."""
    return _split(_load("hpi_by_lgd.csv"))


def by_urban_rural() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _split(_load("hpi_by_urban_rural.csv"))


def cpi() -> pd.Series:
    """UK CPI, all items, 2015=100 (ONS D7BT), quarterly."""
    return _load("cpi.csv")["CPI"]


def to_real(df: pd.DataFrame, base: str) -> pd.DataFrame:
    """Deflate a quarterly frame by CPI, expressed in prices of the `base` quarter (e.g. '2025Q2')."""
    c = cpi()
    assert df.index.isin(c.index).all(), "CPI does not cover every quarter"
    return df.mul(c[base] / c[df.index], axis=0)
