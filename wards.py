"""Build one table of NI's 462 electoral wards: house prices plus Census 2021 and location features.

Sources (all UK Open Government Licence):
- Ward annual price statistics, Department of Finance / LPS (median sale price; only published where a ward
  had 30+ eligible sales in the year)
- Census 2021 main statistics for wards, NISRA
- Ward boundaries (2012, in use since 2014), OSNI via OpenDataNI

Religion and national identity are deliberately left out: in NI they map closely onto where people live, and
a price model built on them would effectively price neighbourhoods by community background.
"""
import functools
import html
import io
import json
import math
import re
import urllib.parse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import nihpi
from fetch_data import get

RAW = nihpi.RAW
WARD_DIR, CENSUS_DIR, GEO_DIR = RAW / "ward", RAW / "census", RAW / "geo"
OUT = Path(__file__).parent / "data" / "processed" / "wards.csv"

PRICE_PAGE = "https://www.finance-ni.gov.uk/publications/annual-ward-district-electoral-area-and-local-government-districts-statistics"
CENSUS_PAGE = "https://www.nisra.gov.uk/publications/census-2021-main-statistics-settlements-and-wards-northern-ireland"
BOUNDARIES = "https://admin.opendatani.gov.uk/api/3/action/package_show?id=osni-open-data-largescale-boundaries-wards-2012"
PRICE_YEARS = (2023, 2024, 2025)  # pooled to cover more wards; each year brought to max(PRICE_YEARS) prices
BELFAST_CITY_HALL = (54.5964, -5.9301)  # lat, lon


# ---------- download ----------

def _download(url: str, dest: Path, magic: bytes) -> None:
    # Files are kept once downloaded. For a new annual release, delete data/raw/ward/ (and update PRICE_YEARS).
    if dest.exists():
        return
    data = get(url)
    if not data.startswith(magic):  # don't save an error page served as 200
        raise RuntimeError(f"{url}: unexpected content, not saved: {data[:200]!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"saved {dest.relative_to(RAW)} ({len(data) / 1e6:.1f} MB)")


def _links(page: str, pattern: str) -> list[str]:
    found = re.findall(rf'href="([^"]*{pattern}[^"]*)"', get(page).decode("utf-8", "replace"))
    if not found:
        raise RuntimeError(f"no link matching {pattern!r} on {page}")
    return [urllib.parse.urljoin(page, u) for u in found]


def fetch() -> None:
    _download(_links(PRICE_PAGE, r'Electoral%20Ward[^"]*Property%20Types[^"]*\.xlsx')[0],
              WARD_DIR / "ward_property_types.xlsx", b"PK")
    for i, url in enumerate(_links(CENSUS_PAGE, r"settlements-and-wards-part-\d\.zip"), 1):
        _download(url, CENSUS_DIR / f"part{i}.zip", b"PK")
    resources = {r["format"]: r["url"] for r in json.loads(get(BOUNDARIES))["result"]["resources"]}
    _download(resources["GeoJSON"], GEO_DIR / "wards_2012.geojson", b"{")


# ---------- prices ----------

def _key(s: pd.Series) -> pd.Series:
    """Normalise names for joining: 'ARMAGH CITY, BANBRIDGE AND CRAIGAVON' -> 'armagh city banbridge and craigavon'."""
    s = s.astype(str).map(html.unescape, na_action="ignore")  # the price file has 'GIANT&apos;S CAUSEWAY'
    return s.str.lower().str.replace(",", "").str.replace(r"\s+", " ", regex=True).str.strip()


def annual_council_index() -> pd.DataFrame:
    """Council standardised price index, mean of each calendar year's quarters; columns keyed like ward data."""
    hpi, _ = nihpi.by_council()
    annual = hpi.groupby(hpi.index.year).mean()
    annual.columns = _key(pd.Series(annual.columns)).values
    return annual


def ward_prices() -> pd.DataFrame:
    """Per ward: median price in latest-year money (mean over PRICE_YEARS where published) and the mix of homes sold."""
    x = pd.ExcelFile(WARD_DIR / "ward_property_types.xlsx")
    rows = pd.concat([x.parse(s, header=4) for s in x.sheet_names if re.search(r"_(Apt|DET|SDT|TER|Total)$", s)])
    rows = rows.rename(columns={"Local Government District": "lgd", "Electoral Ward (2014)": "ward", "Sale Year": "year",
                                "Property Type": "type", "No. of Verified Sales": "sales", "Median Sale Price": "median"})
    rows = rows[rows["year"].isin(PRICE_YEARS)]
    rows["median"] = rows["median"].replace(0, np.nan)  # publisher uses 0 for "no sales"
    rows["lgd"], rows["ward"] = _key(rows["lgd"]), _key(rows["ward"])

    # bring each year's median to latest-year prices with the council's standardised index (annual mean of quarters)
    annual = annual_council_index()
    base = max(PRICE_YEARS)
    factor = rows.apply(lambda r: annual.loc[base, r["lgd"]] / annual.loc[r["year"], r["lgd"]], axis=1)
    rows["median_adj"] = rows["median"] * factor

    total = rows[rows["type"] == "All Sales"]
    out = total.groupby(["lgd", "ward"]).agg(price=("median_adj", "mean"), price_years=("median", "count"),
                                             sales=("sales", "sum"))
    # mix of homes sold: a ward selling mostly detached houses has a higher median for that reason alone
    by_type = rows[rows["type"] != "All Sales"].pivot_table(index=["lgd", "ward"], columns="type", values="sales",
                                                           aggfunc="sum", fill_value=0)
    for t, name in (("DET", "sold_detached"), ("SDT", "sold_semi"), ("TER", "sold_terrace")):
        out[name] = by_type[t] / out["sales"].where(out["sales"] > 0)
    return out.reset_index()


# ---------- census ----------

@functools.cache  # share() asks for the same table several times
def _census_table(code: str) -> pd.DataFrame:
    """Ward sheet of one Census main-statistics table (counts), indexed by ward code."""
    for zp in sorted(CENSUS_DIR.glob("part*.zip")):
        with zipfile.ZipFile(zp) as z:
            if (name := f"census-2021-ms-{code}.xlsx") in z.namelist():
                raw = pd.read_excel(io.BytesIO(z.read(name)), "Ward", header=None)
                break
    else:
        raise FileNotFoundError(code)
    head = raw.index[raw[0].astype(str).eq("Geography")][0]
    t = raw.iloc[head + 1:head + 463].set_axis(raw.iloc[head].astype(str).str.replace(r"\s+", " ", regex=True), axis=1)
    assert t["Geography code"].astype(str).str.match(r"N08\d{6}$").all(), f"{code}: unexpected layout"
    return t.set_index("Geography code")


def census() -> pd.DataFrame:
    def share(code: str, total: str, *parts: str) -> pd.Series:
        t = _census_table(code)
        cols = [c for c in t.columns if any(p in c for p in parts)]
        assert cols, f"{code}: none of {parts}"
        return t[cols].astype(float).sum(axis=1) / t[[c for c in t.columns if c.startswith(total)][0]].astype(float)

    age = _census_table("a02")
    names = age["Geography"].astype(str)
    return pd.DataFrame({
        # 'Abbey (Antrim and Newtownabbey)': names repeated across councils carry the council in brackets
        "ward": _key(names.str.replace(r"\s*\(.*\)$", "", regex=True)),
        "lgd_hint": _key(names.str.extract(r"\((.*?)(?: LGD)?\)$")[0]).where(names.str.contains(r"\)$")),
        "population": age["All usual residents"].astype(float),
        "pct_over_65": share("a02", "All usual residents", "65-69", "70-74", "75-79", "80-84", "85-89", "90+"),
        "pct_owned_outright": share("e15", "All households", "Owns outright"),
        "pct_mortgage": share("e15", "All households", "Owns with a mortgage"),
        "pct_social_rented": share("e15", "All households", "Social rented"),
        "pct_private_rented": share("e15", "All households", "Private rented"),
        "pct_detached_homes": share("e06", "All households", "Detached"),
        "pct_flats": share("e06", "All households", "Flat, maisonette"),
        "pct_no_car": share("e10", "All households", "No cars"),
        "pct_degree": share("g01", "All usual residents", "Level 4 qualifications"),
        "pct_no_qualifications": share("g01", "All usual residents", "No qualifications"),
        "pct_unemployed": share("h02", "All usual residents", "Usual residents aged 16 and over: Economically active: Unemployed"),
        "pct_social_grade_ab": share("h16", "All usual residents", "AB:"),
    })


# ---------- location ----------

def _centroid(geometry: dict) -> tuple[float, float]:
    """Area-weighted centroid (lat, lon) of the largest outer ring; fine at ward scale."""
    polys = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    best = (0.0, 0.0, 0.0)
    for poly in polys:
        ring = poly[0]
        a = cx = cy = 0.0
        for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
            c = x0 * y1 - x1 * y0
            a, cx, cy = a + c, cx + (x0 + x1) * c, cy + (y0 + y1) * c
        if abs(a) > abs(best[0]):
            best = (a, cx / (3 * a), cy / (3 * a))
    return best[2], best[1]


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def location() -> pd.DataFrame:
    rows = {}
    for feat in json.loads((GEO_DIR / "wards_2012.geojson").read_text(encoding="utf-8"))["features"]:
        p = feat["properties"]
        lat, lon = _centroid(feat["geometry"])
        rows[p["WardCode"]] = {"lat": lat, "lon": lon, "area_km2": p["SHAPE_Area"] / 1e6,
                               "km_to_belfast": _km(lat, lon, *BELFAST_CITY_HALL)}
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------- join ----------

def build() -> pd.DataFrame:
    c = census()
    loc = location()
    assert set(c.index) == set(loc.index) and len(c) == 462, "census and boundary ward codes differ"
    c = c.join(loc)
    c["log_density"] = np.log(c["population"] / c["area_km2"])

    p = ward_prices()
    m = p.merge(c.reset_index(names="ward_code"), on="ward", how="left")
    m = m[m["lgd_hint"].isna() | (m["lgd_hint"] == m["lgd"])]  # resolve names shared across councils
    assert m["ward_code"].notna().all() and m["ward_code"].is_unique and len(m) == 462, \
        f"ward join failed: {len(m)} rows, unmatched {sorted(m.loc[m['ward_code'].isna(), 'ward'])[:10]}"
    return m.drop(columns="lgd_hint").set_index("ward_code").sort_index()


if __name__ == "__main__":
    fetch()
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)
    print(f"saved {OUT.relative_to(Path(__file__).parent)}: {len(df)} wards, {df['price'].notna().sum()} with a price")
