"""Download raw data: NI House Price Index (Department of Finance / LPS) and UK CPI (ONS series D7BT).

The HPI comes from the quarterly "Detailed Statistics" spreadsheet on finance-ni.gov.uk, which is about a
year ahead of the OpenDataNI copy. It is converted to the same CSV layout nihpi.py already reads.
"""
import csv
import io
import json
import re
import urllib.parse
import urllib.request

import pandas as pd

from nihpi import RAW

# both sites return 403 to non-browser user agents
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36"}
HPI_PAGE = "https://www.finance-ni.gov.uk/publications/ni-house-price-index-statistical-reports"
CPI = "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/mm23/data"


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def latest_xlsx_url() -> str:
    links = re.findall(r'href="([^"]*Detailed[^"]*Statistics[^"]*\.xlsx)"', get(HPI_PAGE).decode("utf-8", "replace"))
    if not links:
        raise RuntimeError(f"no 'Detailed Statistics' spreadsheet link found on {HPI_PAGE}")
    return urllib.parse.urljoin(HPI_PAGE, links[0])


def table(xlsx: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """Quarterly rows of one sheet, indexed 'Q1 2005' (same format as the CSVs), keeping the sheet's headers."""
    raw = xlsx.parse(sheet, header=None)
    head = raw.index[raw[0].astype(str).str.contains("Year", na=False)][0]  # 'Year' or 'Sale Year'
    t = raw.iloc[head + 1:].set_axis(raw.iloc[head].astype(str).str.strip(), axis=1)
    t = t[t.iloc[:, 1].astype(str).str.fullmatch(r"Q[1-4]")]  # drop notes and blank rows
    year = t.iloc[:, 0].ffill().astype(int).astype(str)  # year is only filled on each Q1 row
    t.index = pd.Index(t.iloc[:, 1] + " " + year, name="Quarter_Year")
    return t.iloc[:, 2:].astype(float)


def csv_name(header: str) -> str:
    """'Armagh City, Banbridge and Craigavon Standardised Price' -> 'Armagh_City_Banbridge_and_Craigavon_Standardised_Price'."""
    return header.replace(",", "").replace("Standardised HPI", "HPI").replace(" ", "_")


def convert(xlsx: pd.ExcelFile) -> dict[str, pd.DataFrame]:
    types = pd.DataFrame({
        "NI_Detached_Property_Price_Index": table(xlsx, "Table 2a").iloc[:, 0],
        "NI_SemiDetached_Property_Price_Index": table(xlsx, "Table 2b").iloc[:, 0],
        "NI_Terrace_Property_Price_Index": table(xlsx, "Table 2c").iloc[:, 0],
        "NI_Apartment_Price_Index": table(xlsx, "Table 2d").iloc[:, 0],
        "NI_Residential_Property_Price_Index": table(xlsx, "Table 1").iloc[:, 0],
    })
    lgd = table(xlsx, "Table 5").filter(regex="(HPI|Price)$").rename(columns=csv_name)
    urban_rural = table(xlsx, "Table 6").filter(regex="(HPI|Price)$").rename(columns=csv_name)
    assert lgd.shape[1] == 22 and urban_rural.shape[1] == 4, "spreadsheet layout changed"
    return {"hpi_by_property_type.csv": types, "hpi_by_lgd.csv": lgd, "hpi_by_urban_rural.csv": urban_rural}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    url = latest_xlsx_url()
    data = get(url)
    if not data.startswith(b"PK"):  # xlsx is a zip; don't save an error/block page served as 200
        raise RuntimeError(f"{url}: not a spreadsheet, nothing saved: {data[:200]!r}")
    frames = convert(pd.ExcelFile(io.BytesIO(data)))  # convert everything before writing anything
    for file, df in frames.items():
        df.to_csv(RAW / file)
    print(f"saved {', '.join(frames)} ({df.index[0]} to {df.index[-1]}) from {url}")

    quarters = json.loads(get(CPI))["quarters"]
    with open(RAW / "cpi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Quarter_Year", "CPI"])  # same quarter format as the HPI files: "Q1 2005"
        w.writerows([q["quarter"] + " " + q["year"], q["value"]] for q in quarters)
    print(f"saved cpi.csv ({quarters[0]['date']} to {quarters[-1]['date']})")


if __name__ == "__main__":
    main()
