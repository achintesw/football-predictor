"""Phase 2 — ingest and cache raw data. No analysis, no 2025/26 loading downstream.

Sources:
  - football-data.co.uk SP1 (La Liga) season CSVs -> data/raw/football_data/{season}.csv
  - Understat La_liga match-level xG (scraped JSON embedded in page) -> data/raw/understat/{season}.json

Run: python ingest.py
"""
import json
import re
import time
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parent / "data" / "raw"
FD_DIR = RAW_DIR / "football_data"
US_DIR = RAW_DIR / "understat"
FD_DIR.mkdir(parents=True, exist_ok=True)
US_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-fetch"}

# Seasons 2014/15 ... 2025/26, labeled by start year.
SEASON_START_YEARS = list(range(2014, 2026))


def season_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1 - 2000}"  # e.g. "2014-15"


def fd_season_code(start_year: int) -> str:
    # football-data.co.uk codes seasons as YYZZ, e.g. 2014/15 -> "1415"
    yy1 = start_year % 100
    yy2 = (start_year + 1) % 100
    return f"{yy1:02d}{yy2:02d}"


def fetch_football_data():
    print("=== football-data.co.uk SP1 ===")
    results = []
    for start_year in SEASON_START_YEARS:
        label = season_label(start_year)
        code = fd_season_code(start_year)
        out_path = FD_DIR / f"{label}.csv"
        if out_path.exists():
            print(f"  {label}: cached ({out_path.stat().st_size} bytes)")
            results.append((label, out_path, "cached"))
            continue
        url = f"https://www.football-data.co.uk/mmz4281/{code}/SP1.csv"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  {label}: FAILED ({e})")
            results.append((label, None, f"error: {e}"))
            continue
        out_path.write_bytes(resp.content)
        print(f"  {label}: downloaded ({len(resp.content)} bytes) from {url}")
        results.append((label, out_path, "downloaded"))
        time.sleep(1)
    return results


def understat_season_code(start_year: int) -> str:
    # Understat seasons are keyed by the start year, e.g. 2014/15 -> "2014"
    return str(start_year)


def fetch_understat():
    print("=== Understat La_liga match xG ===")
    results = []
    for start_year in SEASON_START_YEARS:
        label = season_label(start_year)
        code = understat_season_code(start_year)
        out_path = US_DIR / f"{label}.json"
        if out_path.exists():
            print(f"  {label}: cached ({out_path.stat().st_size} bytes)")
            results.append((label, out_path, "cached"))
            continue
        url = f"https://understat.com/league/La_liga/{code}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  {label}: FAILED to fetch page ({e})")
            results.append((label, None, f"error: {e}"))
            continue

        html = resp.text
        match = re.search(
            r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", html, re.DOTALL
        )
        if not match:
            print(f"  {label}: FAILED (datesData not found in page)")
            results.append((label, None, "error: datesData not found"))
            continue

        raw = match.group(1)
        # Understat escapes the JSON string for JS single-quoted literal use
        # (\xHH byte escapes + standard JSON escapes); decode both layers.
        decoded = raw.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
        try:
            data = json.loads(decoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  {label}: FAILED to parse datesData ({e})")
            results.append((label, None, f"error: parse failed {e}"))
            continue

        out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"  {label}: downloaded ({len(data)} matches) from {url}")
        results.append((label, out_path, "downloaded"))
        time.sleep(1.5)
    return results


if __name__ == "__main__":
    fd_results = fetch_football_data()
    us_results = fetch_understat()
    print("\n=== Done. Cached under data/raw/ (git-ignored). ===")
