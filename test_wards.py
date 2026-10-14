"""Checks for the ward dataset: geometry helpers and the price-to-Census join."""
import pandas as pd
import pytest

import nihpi
import wards


@pytest.mark.skipif(not (wards.WARD_DIR / "ward_property_types.xlsx").exists(), reason="run wards.py to download")
def test_ward_price_is_mean_of_medians_scaled_to_latest_year():
    sheet = pd.read_excel(wards.WARD_DIR / "ward_property_types.xlsx", "Belfast_Total", header=4)
    malone = sheet[sheet["Electoral Ward (2014)"].eq("MALONE")].set_index("Sale Year")["Median Sale Price"]
    hpi, _ = nihpi.by_council()
    belfast = hpi["Belfast"].groupby(hpi.index.year).mean()
    base = max(wards.PRICE_YEARS)
    expected = sum(malone[y] * belfast[base] / belfast[y] for y in wards.PRICE_YEARS) / len(wards.PRICE_YEARS)

    prices = wards.ward_prices().set_index(["lgd", "ward"])
    assert prices.loc[("belfast", "malone"), "price"] == pytest.approx(expected)
    assert prices.loc[("belfast", "malone"), "price_years"] == len(wards.PRICE_YEARS)


def test_centroid_of_square_and_largest_part_wins():
    square = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]  # lon, lat
    assert wards._centroid({"type": "Polygon", "coordinates": [square]}) == pytest.approx((1, 1))
    islet = [[10, 10], [10.1, 10], [10.1, 10.1], [10, 10.1], [10, 10]]
    multi = {"type": "MultiPolygon", "coordinates": [[islet], [square]]}
    assert wards._centroid(multi) == pytest.approx((1, 1))


def test_km_belfast_to_derry():
    assert wards._km(*wards.BELFAST_CITY_HALL, 54.9966, -7.3086) == pytest.approx(97, abs=3)


def test_ward_names_shared_across_councils_resolve_to_the_right_code():
    df = pd.read_csv(wards.OUT, index_col=0)
    assert len(df) == 462 and df.index.is_unique
    abbey = df[df["ward"] == "abbey"].set_index("lgd").index
    assert set(abbey) == {"antrim and newtownabbey", "newry mourne and down"}
    assert df.loc["N08000101", "lgd"] == "antrim and newtownabbey"  # 'Abbey (Antrim and Newtownabbey)' in the Census
