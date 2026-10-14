"""Assistant tools and eval grading, tested without calling the API."""
import json

import numpy as np
import pytest

import assistant as a
import nihpi
from eval_assistant import cases, has_number, numbers_in


def test_search_wards_applies_every_filter():
    res = json.loads(a.search_wards(max_price=200000, max_km_to_belfast=15, council="belfast", limit=25))
    assert res["wards"]
    for w in res["wards"]:
        assert (w["price"] or w["estimate"]) <= 200000 and w["km_to_belfast"] <= 15 and w["council"] == "Belfast"
    prices = [w["price"] or w["estimate"] for w in res["wards"]]
    assert prices == sorted(prices)


def test_search_wards_without_estimates_returns_only_published_prices():
    res = json.loads(a.search_wards(include_estimates=False, limit=25))
    assert all(w["estimate"] is None and w["price"] for w in res["wards"])
    assert res["matching_wards"] == a.ward_table()["price"].notna().sum()


def test_ward_details_handles_misspelling_and_shared_names():
    assert json.loads(a.ward_details("cullra"))["ward"] == "Cultra"
    shared = json.loads(a.ward_details("abbey"))
    assert set(shared["councils"]) == {"Antrim and Newtownabbey", "Newry Mourne and Down"}
    assert json.loads(a.ward_details("abbey", "newry"))["council"] == "Newry Mourne and Down"
    assert "error" in json.loads(a.ward_details("zzzz"))


def test_council_trends_matches_index():
    idx, _ = nihpi.by_council()
    s = idx["Belfast"]
    row = json.loads(a.council_trends("belfast"))["councils"][0]
    assert row["vs_2007_peak_%"] == pytest.approx((s.iloc[-1] / s[:"2009Q4"].max() - 1) * 100, abs=0.06)
    assert len(json.loads(a.council_trends())["councils"]) == 11
    with pytest.raises(ValueError):
        a.council_trends("atlantis")


@pytest.mark.parametrize("name", ["antrim", "down", "mid"])
def test_council_name_matching_two_councils_is_an_error_not_a_guess(name):
    with pytest.raises(ValueError, match="matches 2 councils"):
        a.search_wards(council=name)
    assert json.loads(a.council_trends("mid ulster"))["councils"][0]["council"] == "Mid Ulster"


def test_numbers_in_reads_pound_shorthand():
    assert numbers_in("about £360k, or £0.5m, or £1,250,000; up 9.6%") == [360000, 500000, 1250000, 9.6]
    assert has_number("prices are 37.1% below the peak", -37.1, 0.6)
    assert not has_number("prices rose 3%", 37.1, 0.6)


def test_eval_cases_build_and_grade_a_good_answer():
    cs = {c["id"]: c for c in cases()}
    assert len(cs) == 13
    assert cs["no_rent_data"]["check"]("Sorry, I don't have rent data, only sale prices.")
    assert not cs["no_rent_data"]["check"]("Rent in Belfast averages £900 a month.")
    malone = json.loads(a.ward_details("malone", "belfast"))["price"]
    assert cs["ward_price"]["check"](f"The median in Malone is about £{malone / 1000:.0f}k.")
