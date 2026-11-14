"""Phase 3a — unified match table from cached football-data SP1 files.

Builds one row per La Liga match (2014/15-2024/25) with goals, Pinnacle
opening/closing odds, and de-vigged 1X2 probabilities (proportional method,
Shin as cross-check). 2025/26 stays cached raw and is never loaded here.

Run: python features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "data" / "raw" / "football_data"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]
# 2025/26 is deliberately excluded — sealed holdout, cached raw only.

# dd/mm/yy through 2017/18, dd/mm/yyyy from 2018/19 (per Phase 2 QA finding).
SHORT_YEAR_SEASONS = {"2014-15", "2015-16", "2016-17", "2017-18"}

REQUIRED_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
    "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
]


def _devig_proportional(ph: float, pd_: float, pa: float) -> tuple[float, float, float]:
    raw = np.array([1.0 / ph, 1.0 / pd_, 1.0 / pa])
    norm = raw / raw.sum()
    return tuple(norm)


def _shin_probs(ph: float, pd_: float, pa: float) -> tuple[float, float, float]:
    """Shin (1992) de-vig: solves for insider-trading proportion z via bisection."""
    q = np.array([1.0 / ph, 1.0 / pd_, 1.0 / pa])

    def total(z: float) -> float:
        return float(np.sum((np.sqrt(z**2 + 4 * (1 - z) * q**2) - z) / (2 * (1 - z))))

    lo, hi = 0.0, 0.49
    f_lo, f_hi = total(lo) - 1.0, total(hi) - 1.0
    if f_lo <= 0:
        # Already balanced (no overround) -> no insider adjustment.
        z = 0.0
    elif f_hi > 0:
        # Degenerate/extreme odds; fall back to z at hi bound.
        z = hi
    else:
        for _ in range(100):
            mid = (lo + hi) / 2
            f_mid = total(mid) - 1.0
            if f_mid > 0:
                lo = mid
            else:
                hi = mid
        z = (lo + hi) / 2

    p = (np.sqrt(z**2 + 4 * (1 - z) * q**2) - z) / (2 * (1 - z))
    p = p / p.sum()  # numerical cleanup so it sums exactly to 1
    return tuple(p)


def build_match_table() -> pd.DataFrame:
    rows = []
    for season in TRAIN_SEASONS:
        path = RAW_DIR / f"{season}.csv"
        df = pd.read_csv(path, encoding="latin1")

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        assert not missing, f"{season}: missing required columns {missing}"

        date_fmt = "%d/%m/%y" if season in SHORT_YEAR_SEASONS else "%d/%m/%Y"
        dates = pd.to_datetime(df["Date"], format=date_fmt)

        out = pd.DataFrame({
            "season": season,
            "date": dates,
            "home_team": df["HomeTeam"],
            "away_team": df["AwayTeam"],
            "fthg": df["FTHG"],
            "ftag": df["FTAG"],
            "psh": df["PSH"],
            "psd": df["PSD"],
            "psa": df["PSA"],
            "psch": df["PSCH"],
            "pscd": df["PSCD"],
            "psca": df["PSCA"],
        })
        out["missing_closing_odds"] = out[["psch", "pscd", "psca"]].isna().any(axis=1)
        out["missing_opening_odds"] = out[["psh", "psd", "psa"]].isna().any(axis=1)
        rows.append(out)

    matches = pd.concat(rows, ignore_index=True)

    def devig(r, fn, odds_cols, missing_col):
        if r[missing_col]:
            return (np.nan, np.nan, np.nan)
        return fn(*(r[c] for c in odds_cols))

    prop_open = matches.apply(lambda r: devig(r, _devig_proportional, ["psh", "psd", "psa"], "missing_opening_odds"), axis=1)
    matches["prop_open_h"], matches["prop_open_d"], matches["prop_open_a"] = zip(*prop_open)
    shin_open = matches.apply(lambda r: devig(r, _shin_probs, ["psh", "psd", "psa"], "missing_opening_odds"), axis=1)
    matches["shin_open_h"], matches["shin_open_d"], matches["shin_open_a"] = zip(*shin_open)

    prop_close = matches.apply(lambda r: devig(r, _devig_proportional, ["psch", "pscd", "psca"], "missing_closing_odds"), axis=1)
    matches["prop_close_h"], matches["prop_close_d"], matches["prop_close_a"] = zip(*prop_close)
    shin_close = matches.apply(lambda r: devig(r, _shin_probs, ["psch", "pscd", "psca"], "missing_closing_odds"), axis=1)
    matches["shin_close_h"], matches["shin_close_d"], matches["shin_close_a"] = zip(*shin_close)

    return matches


def qa_report(matches: pd.DataFrame) -> None:
    print("=== QA report ===")
    for season in TRAIN_SEASONS:
        sub = matches[matches.season == season]
        n = len(sub)
        dupes = sub.duplicated(subset=["date", "home_team", "away_team"]).sum()
        goal_range_ok = sub["fthg"].between(0, 15).all() and sub["ftag"].between(0, 15).all()
        flagged_close = int(sub["missing_closing_odds"].sum())
        flagged_open = int(sub["missing_opening_odds"].sum())
        print(
            f"  {season}: rows={n} dupes={dupes} goals_plausible={goal_range_ok} "
            f"flagged_missing_closing={flagged_close} flagged_missing_opening={flagged_open}"
        )

    total_dupes = matches.duplicated(subset=["season", "date", "home_team", "away_team"]).sum()
    print(f"\n  total rows: {len(matches)} (expect {380 * len(TRAIN_SEASONS)})")
    print(f"  total duplicate fixtures: {total_dupes}")
    print(f"  total flagged missing-closing-odds matches: {int(matches['missing_closing_odds'].sum())}")
    print(f"  total flagged missing-opening-odds matches: {int(matches['missing_opening_odds'].sum())}")

    for label, cols, missing_col in [
        ("prop_open", ["prop_open_h", "prop_open_d", "prop_open_a"], "missing_opening_odds"),
        ("shin_open", ["shin_open_h", "shin_open_d", "shin_open_a"], "missing_opening_odds"),
        ("prop_close", ["prop_close_h", "prop_close_d", "prop_close_a"], "missing_closing_odds"),
        ("shin_close", ["shin_close_h", "shin_close_d", "shin_close_a"], "missing_closing_odds"),
    ]:
        non_missing = matches[~matches[missing_col]]
        sums = non_missing[cols].sum(axis=1)
        max_dev = (sums - 1.0).abs().max()
        print(f"  {label} probs sum-to-1 max deviation (non-missing rows, n={len(non_missing)}): {max_dev:.2e}")

    print("\n  flagged rows (missing closing odds):")
    print(matches[matches["missing_closing_odds"]][["season", "date", "home_team", "away_team"]].to_string(index=False))

    print("\n  flagged rows (missing opening odds) -- NOT requested in task, found during QA:")
    print(matches[matches["missing_opening_odds"]][["season", "date", "home_team", "away_team"]].to_string(index=False))


if __name__ == "__main__":
    matches = build_match_table()
    out_path = PROCESSED_DIR / "matches_2014_2024.parquet"
    matches.to_parquet(out_path, index=False)
    print(f"Wrote {len(matches)} rows to {out_path}\n")
    qa_report(matches)
