"""Chat assistant for NI house prices. Claude answers by calling tools over this project's own data.

Run:  .venv\\Scripts\\python assistant.py      (needs ANTHROPIC_API_KEY)
"""
import difflib
import functools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import forecast
import nihpi
import wards

MODEL = "claude-opus-5-5"
PREDICTIONS = Path(__file__).parent / "outputs" / "ward_predictions.csv"

SYSTEM = """You answer questions about house prices in Northern Ireland using the tools provided, which read \
official open data analysed in this project. Use the tools for every number you state; never answer price \
questions from memory. If the tools can't answer something, say so plainly.

What the data is, so you can explain its limits:
- Ward prices: median sale price per electoral ward (462 wards), 2023-2025 pooled and adjusted to 2025 prices \
with the council's official price index. Published only where a ward had 30+ sales in a year. Wards without \
a published price have a model estimate instead, always label those as estimates.
- "Predicted" and "gap": a ridge regression predicts each ward's price from Census 2021 statistics (education, \
social grade, tenure, housing type), density, distance to Belfast and the mix of homes sold. Typical error is \
about 7.5%. A positive gap means the ward sells above what comparable wards suggest; it reflects things the data \
can't see (views, schools, reputation, condition). Never call a ward "overpriced" or a "bargain" on this alone.
- Distance is straight-line km from Belfast City Hall, not travel time. No other distances are available.
- Council trends: the official NI House Price Index by council (2005 to latest quarter). The next-quarter \
forecast is from a simple model that, on unseen data, does not reliably beat a 4-quarter average; present it \
as a rough indication.
- There is no data on individual properties, streets, rents, or listings.

Answer concisely in British English with £ figures rounded sensibly. You are not a financial adviser: give \
data, not advice on whether to buy."""


# ---------- data (loaded once) ----------

@functools.cache
def ward_table() -> pd.DataFrame:
    df = pd.read_csv(wards.OUT, index_col=0)
    pred = pd.read_csv(PREDICTIONS, index_col=0)[["predicted", "gap_%"]]
    df = df.join(pred)
    df["estimated"] = df["price"].isna()
    df["price_or_estimate"] = df["price"].fillna(df["predicted"])
    return df


@functools.cache
def council_table() -> pd.DataFrame:
    idx, price = nihpi.by_council()
    real = nihpi.to_real(idx, str(idx.index[-1]))
    fc = forecast.live_forecast(forecast.build_dataset())
    rows = {}
    for c in idx.columns:
        s, r = idx[c], real[c]
        peak = s[:"2009Q4"].idxmax()
        rows[wards._key(pd.Series([c]))[0]] = {
            "council": c,
            "latest_quarter": str(s.index[-1]),
            "standardised_price": round(price[c].iloc[-1], -3),
            "change_1y_%": round((s.iloc[-1] / s.iloc[-5] - 1) * 100, 1),
            "change_5y_%": round((s.iloc[-1] / s.iloc[-21] - 1) * 100, 1),
            "vs_2007_peak_%": round((s.iloc[-1] / s[peak] - 1) * 100, 1),
            "vs_2007_peak_after_inflation_%": round((r.iloc[-1] / r[peak] - 1) * 100, 1),
            "forecast_quarter": fc.name,
            "forecast_next_quarter_%": round((np.exp(fc[c]) - 1) * 100, 1),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _council_key(council: str | None) -> str | None:
    """Match a council name loosely ('Derry', 'newry mourne & down'); None if no council given."""
    if not council:
        return None
    key = wards._key(pd.Series([council.replace("&", "and")]))[0]
    names = list(council_table().index)
    hits = [n for n in names if key in n] or difflib.get_close_matches(key, names, n=1, cutoff=0.5)
    if not hits:
        raise ValueError(f"unknown council {council!r}; councils are: {', '.join(council_table()['council'])}")
    if len(hits) > 1:  # 'antrim', 'down', 'mid' each match two councils
        raise ValueError(f"{council!r} matches {len(hits)} councils: {', '.join(council_table().loc[hits, 'council'])}")
    return hits[0]


def _name(key: str) -> str:
    """'giant's causeway' -> "Giant's Causeway", 'curran and inver' -> 'Curran and Inver'."""
    return " ".join(w if w in ("and", "of") else w[:1].upper() + w[1:] for w in key.split())


def _ward_row(r: pd.Series) -> dict:
    return {
        "ward": _name(r["ward"]), "council": council_table().loc[r["lgd"], "council"],
        "price": None if r["estimated"] else round(r["price"], -3),
        "estimate": round(r["predicted"], -3) if r["estimated"] else None,
        "km_to_belfast": round(r["km_to_belfast"], 1),
        "gap_vs_model_%": None if pd.isna(r["gap_%"]) else round(r["gap_%"]),
    }


# ---------- tools ----------

def search_wards(max_price: float | None = None, min_price: float | None = None,
                 max_km_to_belfast: float | None = None, council: str | None = None,
                 include_estimates: bool = True, sort_by: str = "price_asc", limit: int = 10) -> str:
    """Find electoral wards matching price, location and council filters.

    Args:
        max_price: Highest median price in £ (published price, or estimate where unpublished).
        min_price: Lowest median price in £.
        max_km_to_belfast: Maximum straight-line distance from Belfast City Hall in km.
        council: Council area name, e.g. "Belfast", "Derry City and Strabane". Partial names work.
        include_estimates: Include wards whose price is a model estimate (too few sales to publish).
        sort_by: One of "price_asc", "price_desc", "distance", "gap_desc" (most above model first), "gap_asc".
        limit: Number of wards to return, 1 to 25.
    """
    df = ward_table()
    if not include_estimates:
        df = df[~df["estimated"]]
    p = df["price_or_estimate"]
    if max_price is not None:
        df = df[p <= max_price]
    if min_price is not None:
        df = df[df["price_or_estimate"] >= min_price]
    if max_km_to_belfast is not None:
        df = df[df["km_to_belfast"] <= max_km_to_belfast]
    if (key := _council_key(council)) is not None:
        df = df[df["lgd"] == key]
    order = {"price_asc": ("price_or_estimate", True), "price_desc": ("price_or_estimate", False),
             "distance": ("km_to_belfast", True), "gap_desc": ("gap_%", False), "gap_asc": ("gap_%", True)}
    if sort_by not in order:
        raise ValueError(f"sort_by must be one of {list(order)}")
    col, asc = order[sort_by]
    df = df.sort_values(col, ascending=asc, na_position="last")
    limit = max(1, min(int(limit), 25))
    return json.dumps({"matching_wards": len(df), "wards": [_ward_row(r) for _, r in df.head(limit).iterrows()]})


def ward_details(ward: str, council: str | None = None) -> str:
    """Everything known about one electoral ward: price, model prediction, Census profile and its council's trend.

    Args:
        ward: Ward name, e.g. "Malone", "Giant's Causeway". Spelling need not be exact.
        council: Council area, needed only when the ward name exists in more than one council.
    """
    df = ward_table()
    key = wards._key(pd.Series([ward]))[0]
    hits = df[df["ward"] == key]
    if hits.empty:
        close = difflib.get_close_matches(key, df["ward"].unique(), n=5, cutoff=0.6)
        hits = df[df["ward"].isin(close[:1])]
        if hits.empty:
            return json.dumps({"error": f"no ward called {ward!r}", "similar_names": [_name(c) for c in close]})
    if (ckey := _council_key(council)) is not None:
        hits = hits[hits["lgd"] == ckey]
    if len(hits) > 1:
        return json.dumps({"error": "ward name exists in several councils, pass council",
                           "councils": sorted(council_table().loc[hits["lgd"], "council"])})
    if hits.empty:
        return json.dumps({"error": f"no ward {ward!r} in council {council!r}"})
    r = hits.iloc[0]
    pct = lambda c: round(r[c] * 100, 1)
    return json.dumps({
        **_ward_row(r),
        "price_years_published": int(r["price_years"]),
        "sales_2023_2025": int(r["sales"]),
        "model_predicted_price": round(r["predicted"], -3),
        "census_2021": {
            "population": int(r["population"]), "people_per_km2": round(np.exp(r["log_density"])),
            "degree_%": pct("pct_degree"), "no_qualifications_%": pct("pct_no_qualifications"),
            "professional_managerial_households_%": pct("pct_social_grade_ab"),
            "owned_outright_%": pct("pct_owned_outright"), "mortgage_%": pct("pct_mortgage"),
            "social_rented_%": pct("pct_social_rented"), "private_rented_%": pct("pct_private_rented"),
            "detached_homes_%": pct("pct_detached_homes"), "flats_%": pct("pct_flats"),
            "over_65_%": pct("pct_over_65"), "no_car_households_%": pct("pct_no_car"),
        },
        "homes_sold_mix_%": {"detached": pct("sold_detached"), "semi": pct("sold_semi"), "terrace": pct("sold_terrace")},
        "council_trend": council_table().loc[r["lgd"]].to_dict(),
    })


def council_trends(council: str | None = None) -> str:
    """Official house price index trends for one council area, or all 11 if no council is given.

    Returns the latest standardised price, change over 1 and 5 years, position vs the 2007 peak (in cash and
    after inflation), and a rough next-quarter forecast.

    Args:
        council: Council area name, partial names work. Omit to get all councils.
    """
    t = council_table()
    key = _council_key(council)
    rows = t if key is None else t.loc[[key]]
    return json.dumps({"councils": rows.to_dict(orient="records")})


# ---------- chat ----------

def ask(client, history: list, question: str, usage: dict | None = None) -> tuple[str, list[str]]:
    """Run one user turn through the tool loop. Appends to history; returns (answer text, tool names called).

    usage: optional dict, token counts are added to it (input, output, cache_read, cache_write).
    """
    from anthropic import beta_tool

    history.append({"role": "user", "content": question})
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        tools=[beta_tool(f) for f in (search_wards, ward_details, council_trends)],
        messages=list(history),
        output_config={"effort": "medium"},
        cache_control={"type": "ephemeral"},  # caches system + tools + history across turns
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if a safety classifier declines, retry on Anthropic's recommended model
    )
    last, called = None, []
    for message in runner:
        last = message
        if usage is not None:
            u = message.usage
            for k, v in (("input", u.input_tokens), ("output", u.output_tokens),
                         ("cache_read", u.cache_read_input_tokens), ("cache_write", u.cache_creation_input_tokens)):
                usage[k] = usage.get(k, 0) + (v or 0)
        history.append({"role": "assistant", "content": message.content})
        called += [b.name for b in message.content if b.type == "tool_use"]
        if (tool_response := runner.generate_tool_call_response()) is not None:
            history.append(tool_response)
    if last is None or last.stop_reason == "refusal":
        return "Sorry, I can't help with that request.", called
    return "".join(b.text for b in last.content if b.type == "text"), called


def main() -> None:
    import anthropic

    # tool errors already go back to Claude; without a handler Python also prints their tracebacks in the chat
    logging.getLogger("anthropic").addHandler(logging.NullHandler())
    client = anthropic.Anthropic()
    history: list = []
    print("NI house prices assistant. Ask a question, or press Enter on an empty line to quit.\n")
    while question := input("> ").strip():
        n = len(history)
        try:
            answer, _ = ask(client, history, question)
        except anthropic.AuthenticationError:
            raise SystemExit("No valid API key. Set ANTHROPIC_API_KEY (see README).")
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as e:
            del history[n:]  # drop the half-finished turn so the conversation stays valid
            print(f"{type(e).__name__}, please try again.\n")
            continue
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
