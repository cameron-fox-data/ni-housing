"""Eval for the chat assistant: questions with answers computed from the data at run time, graded by code.

Each case checks facts (the right ward/council/number appears) or behaviour (caveats, refusing what the data
can't answer, labelling estimates). Calls the real API, so it costs money: run with --yes.

    .venv\\Scripts\\python eval_assistant.py --yes
"""
import json
import re
import sys
from pathlib import Path

import assistant as a

OUT = Path(__file__).parent / "outputs" / "eval_results.json"
PRICE_IN, PRICE_OUT = 4.00, 20.00  # $ per million tokens, claude-opus-5-5
CACHE_READ, CACHE_WRITE = 0.20, 5.00  # cache read 0.1x is listed at $0.20; 5-minute cache writes cost 1.25x input


def numbers_in(text: str) -> list[float]:
    """Every number in the text, with £ shorthand expanded: '£360k' -> 360000, '£0.5m' -> 500000, '1,250' -> 1250."""
    out = []
    for m in re.finditer(r"(-?\d[\d,]*(?:\.\d+)?)\s*(k|m|million|thousand)?\b", text, flags=re.I):
        v = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        out.append(v * {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get(unit, 1))
    return out


def has_number(text: str, target: float, tol: float) -> bool:
    """A number within tol of target (absolute tolerance), sign-insensitive so '-9.6%' and '9.6% below' both count."""
    return any(abs(abs(n) - abs(target)) <= tol for n in numbers_in(text))


def has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


def cases() -> list[dict]:
    """(question, check) pairs; expected values come from the same data the tools read."""
    w = lambda s: json.loads(s)
    malone = w(a.ward_details("malone", "belfast"))["price"]
    cheapest10 = w(a.search_wards(max_km_to_belfast=10, include_estimates=False, sort_by="price_asc", limit=1))["wards"][0]
    top = w(a.council_trends())["councils"]
    priciest = max(top, key=lambda c: c["standardised_price"])["council"]
    derry = w(a.council_trends("derry"))["councils"][0]
    belfast = w(a.council_trends("belfast"))["councils"][0]
    ccg = [r["ward"] for r in w(a.search_wards(council="causeway", sort_by="gap_desc", include_estimates=False, limit=3))["wards"]]
    under150 = [r["ward"] for r in w(a.search_wards(max_price=150000, max_km_to_belfast=20, limit=25))["wards"]]
    ballymurphy = w(a.ward_details("ballymurphy"))
    cultra_gap = w(a.ward_details("cultra"))["gap_vs_model_%"]

    return [
        {"id": "ward_price", "q": "What's the typical house price in Malone, Belfast?",
         "check": lambda t: has_number(t, malone, malone * 0.03), "expect": f"~£{malone:,.0f}"},
        {"id": "cheapest_near_belfast", "q": "Which ward within 10 km of Belfast City Hall has the lowest published median price?",
         "check": lambda t: has(t, re.escape(cheapest10["ward"])), "expect": cheapest10["ward"]},
        {"id": "priciest_council", "q": "Which council area has the highest standardised house price right now?",
         "check": lambda t: has(t, re.escape(priciest)), "expect": priciest},
        {"id": "peak_nominal_real", "q": "How do Derry City and Strabane house prices compare with their 2007 peak, before and after inflation?",
         "check": lambda t: has_number(t, derry["vs_2007_peak_%"], 0.6) and has_number(t, derry["vs_2007_peak_after_inflation_%"], 0.6),
         "expect": f"{derry['vs_2007_peak_%']}% and {derry['vs_2007_peak_after_inflation_%']}%"},
        {"id": "above_model", "q": "Which three wards in Causeway Coast and Glens sell furthest above what the model predicts?",
         "check": lambda t: sum(has(t, re.escape(n)) for n in ccg) >= 2, "expect": ", ".join(ccg)},
        {"id": "budget_search", "q": "Find me some wards under £150,000 within 20 km of Belfast.",
         "check": lambda t: any(has(t, re.escape(n)) for n in under150) and not has(t, r"no wards"),
         "expect": f"any of {len(under150)} wards"},
        {"id": "estimate_labelled", "q": "What do houses cost in Ballymurphy?",
         "check": lambda t: has(t, r"estimat") and has_number(t, ballymurphy["estimate"], ballymurphy["estimate"] * 0.05),
         "expect": f"estimate ~£{ballymurphy['estimate']:,.0f}, labelled as an estimate"},
        {"id": "forecast_caveat", "q": "What will Belfast house prices do next quarter?",
         "check": lambda t: has_number(t, belfast["forecast_next_quarter_%"], 0.3)
                            and has(t, r"rough|indicat|uncertain|not reliabl|caution|simple"),
         "expect": f"{belfast['forecast_next_quarter_%']}% with a caveat"},
        {"id": "gap_not_overpriced", "q": "Is Cultra overpriced?",
         "check": lambda t: has_number(t, cultra_gap, 1.5)
                            and has(t, r"view|school|reputation|prestige|can't see|cannot see|not captured|doesn't capture|not in the data")
                            and not has(t, r"\byes,? (it|cultra) is overpriced"),
         "expect": f"gap ~{cultra_gap}% explained as unmeasured factors, not 'overpriced'"},
        {"id": "ambiguous_ward", "q": "Tell me about Abbey ward.",
         "check": lambda t: has(t, r"Antrim") and has(t, r"Newry"), "expect": "notes both Abbey wards / asks which"},
        {"id": "no_address_data", "q": "How much is 14 Malone Road worth?",
         "check": lambda t: has(t, r"individual|specific (propert|home|house|address)|particular (propert|house)|street|address"),
         "expect": "says it has no property-level data"},
        {"id": "no_rent_data", "q": "How much does it cost to rent a flat in Belfast?",
         "check": lambda t: has(t, r"(no|don't have|do not have|doesn't (include|cover|have)|not (include|cover|available)).{0,40}rent|rent.{0,60}(not|no) (available|data|cover)"),
         "expect": "says rents aren't in the data"},
        {"id": "no_advice", "q": "Should I buy a house in Ballymena now?",
         "check": lambda t: has(t, r"financial advi|can't (tell you|advise)|cannot advise|not (able|in a position) to advise|not advice|your (own )?circumstances"),
         "expect": "gives data, declines to advise"},
    ]


def main() -> None:
    import anthropic

    cs = cases()
    if "--yes" not in sys.argv:
        raise SystemExit(f"{len(cs)} questions, each a fresh conversation on {a.MODEL}; roughly $1-3 per run. "
                         "Re-run with --yes to call the API.")
    client = anthropic.Anthropic()
    usage, results = {}, []
    for c in cs:
        answer, tools = a.ask(client, [], c["q"], usage)
        ok = bool(c["check"](answer))
        results.append({"id": c["id"], "pass": ok, "expect": c["expect"], "tools": tools, "question": c["q"], "answer": answer})
        print(f"{'PASS' if ok else 'FAIL'}  {c['id']:<22} tools={tools}")
    cost = (usage.get("input", 0) * PRICE_IN + usage.get("output", 0) * PRICE_OUT
            + usage.get("cache_read", 0) * CACHE_READ + usage.get("cache_write", 0) * CACHE_WRITE) / 1e6
    passed = sum(r["pass"] for r in results)
    print(f"\n{passed}/{len(results)} passed · tokens {usage} · cost ≈ ${cost:.2f}")
    OUT.write_text(json.dumps({"model": a.MODEL, "passed": passed, "total": len(results), "usage": usage,
                               "cost_usd": round(cost, 3), "results": results}, indent=1), encoding="utf-8")
    print(f"Answers saved to {OUT}")


if __name__ == "__main__":
    main()
