"""Phase 4 — naive and market benchmarks, walk-forward by season.

Reads data/processed/matches_2014_2024.parquet (built by features.py).
Never loads 2025/26. Produces counts/metrics only; no tuning, no PREREG edits.

Run: python benchmarks.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MATCH_TABLE = ROOT / "data" / "processed" / "matches_2014_2024.parquet"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]
TEST_SEASONS = [
    "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25",
]
OUTCOMES = ["H", "D", "A"]  # fixed order for RPS cumulative sums


def outcome_label(fthg: int, ftag: int) -> str:
    if fthg > ftag:
        return "H"
    if fthg < ftag:
        return "A"
    return "D"


def outcome_one_hot(labels: pd.Series) -> np.ndarray:
    return np.stack([
        (labels == "H").astype(float),
        (labels == "D").astype(float),
        (labels == "A").astype(float),
    ], axis=1)


def rps(probs: np.ndarray, one_hot: np.ndarray) -> np.ndarray:
    """Per-match RPS, outcome order H, D, A.

    RPS = 0.5 * sum_{i=1}^{2} (cumP_i - cumE_i)^2
    """
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    cum_e = np.cumsum(one_hot, axis=1)[:, :2]
    return 0.5 * np.sum((cum_p - cum_e) ** 2, axis=1)


def log_loss(probs: np.ndarray, one_hot: np.ndarray) -> np.ndarray:
    p_true = np.sum(probs * one_hot, axis=1)
    return -np.log(p_true)


def brier(probs: np.ndarray, one_hot: np.ndarray) -> np.ndarray:
    return np.sum((probs - one_hot) ** 2, axis=1)


def main() -> None:
    matches = pd.read_parquet(MATCH_TABLE)
    matches["outcome"] = [
        outcome_label(h, a) for h, a in zip(matches["fthg"], matches["ftag"])
    ]

    sanity_failures = []
    stop_conditions = []
    per_season_rows = []

    for t in TEST_SEASONS:
        train_seasons = [s for s in ALL_SEASONS if s < t]
        test_df_full = matches[matches.season == t]

        # Same match set dropped from scoring for both benchmarks.
        scored = test_df_full[~test_df_full["missing_closing_odds"]].copy()
        n_scored = len(scored)
        n_dropped = len(test_df_full) - n_scored

        # --- sanity: n per season ---
        expected_n = 379 if t == "2017-18" else 380
        if len(test_df_full) != 380:
            sanity_failures.append(f"{t}: season row count = {len(test_df_full)}, expected 380")
        if n_scored != expected_n:
            sanity_failures.append(f"{t}: scored n = {n_scored}, expected {expected_n}")

        one_hot = outcome_one_hot(scored["outcome"])

        # --- 1. NAIVE: expanding-window H/D/A base rates from train_seasons ---
        train_df = matches[matches.season.isin(train_seasons)]
        rates = train_df["outcome"].value_counts(normalize=True)
        naive_probs_row = np.array([rates.get(o, 0.0) for o in OUTCOMES])
        naive_probs = np.tile(naive_probs_row, (n_scored, 1))

        # --- 2. MARKET: Pinnacle closing, proportional de-vig (+ Shin cross-check) ---
        market_prop = scored[["prop_close_h", "prop_close_d", "prop_close_a"]].to_numpy()
        market_shin = scored[["shin_close_h", "shin_close_d", "shin_close_a"]].to_numpy()

        # --- sanity: probs sum to 1, no NaN ---
        for label, arr in [("naive", naive_probs), ("market_prop", market_prop), ("market_shin", market_shin)]:
            if np.isnan(arr).any():
                sanity_failures.append(f"{t}: NaN found in {label} probabilities among scored rows")
            sums = arr.sum(axis=1)
            max_dev = np.abs(sums - 1.0).max() if len(sums) else 0.0
            if max_dev > 1e-6:
                sanity_failures.append(f"{t}: {label} probs sum-to-1 max deviation {max_dev:.2e} exceeds tolerance")

        # --- 4. metrics ---
        def metrics(probs):
            return {
                "rps": float(np.mean(rps(probs, one_hot))),
                "log_loss": float(np.mean(log_loss(probs, one_hot))),
                "brier": float(np.mean(brier(probs, one_hot))),
            }

        m_naive = metrics(naive_probs)
        m_market_prop = metrics(market_prop)
        m_market_shin = metrics(market_shin)

        # --- 8. stop condition: market RPS >= naive RPS ---
        if m_market_prop["rps"] >= m_naive["rps"]:
            stop_conditions.append(
                f"{t}: market_prop RPS ({m_market_prop['rps']:.4f}) >= naive RPS ({m_naive['rps']:.4f})"
            )

        outcome_freq = scored["outcome"].value_counts(normalize=True)
        per_season_rows.append({
            "season": t,
            "n_test_total": len(test_df_full),
            "n_scored": n_scored,
            "n_dropped_missing_closing": n_dropped,
            "train_seasons_n": len(train_seasons),
            "naive_H": naive_probs_row[0], "naive_D": naive_probs_row[1], "naive_A": naive_probs_row[2],
            "outcome_freq_H": outcome_freq.get("H", 0.0),
            "outcome_freq_D": outcome_freq.get("D", 0.0),
            "outcome_freq_A": outcome_freq.get("A", 0.0),
            "rps_naive": m_naive["rps"],
            "rps_market_prop": m_market_prop["rps"],
            "rps_market_shin": m_market_shin["rps"],
            "rps_gap_naive_minus_market_prop": m_naive["rps"] - m_market_prop["rps"],
            "rps_prop_minus_shin": m_market_prop["rps"] - m_market_shin["rps"],
            "log_loss_naive": m_naive["log_loss"],
            "log_loss_market_prop": m_market_prop["log_loss"],
            "log_loss_market_shin": m_market_shin["log_loss"],
            "brier_naive": m_naive["brier"],
            "brier_market_prop": m_market_prop["brier"],
            "brier_market_shin": m_market_shin["brier"],
        })

    results = pd.DataFrame(per_season_rows)

    # --- pooled (all test seasons) ---
    pooled_scored_mask = (~matches["missing_closing_odds"]) & (matches["season"].isin(TEST_SEASONS))
    pooled = matches[pooled_scored_mask].copy()
    pooled_one_hot = outcome_one_hot(pooled["outcome"])

    # Pooled naive: for each row use that row's season-specific expanding-window naive prob.
    pooled_naive_probs = results.set_index("season").loc[pooled["season"], ["naive_H", "naive_D", "naive_A"]].to_numpy()
    pooled_market_prop = pooled[["prop_close_h", "prop_close_d", "prop_close_a"]].to_numpy()
    pooled_market_shin = pooled[["shin_close_h", "shin_close_d", "shin_close_a"]].to_numpy()

    pooled_rps_naive = float(np.mean(rps(pooled_naive_probs, pooled_one_hot)))
    pooled_rps_market_prop = float(np.mean(rps(pooled_market_prop, pooled_one_hot)))
    pooled_rps_market_shin = float(np.mean(rps(pooled_market_shin, pooled_one_hot)))
    pooled_log_loss_naive = float(np.mean(log_loss(pooled_naive_probs, pooled_one_hot)))
    pooled_log_loss_market_prop = float(np.mean(log_loss(pooled_market_prop, pooled_one_hot)))
    pooled_log_loss_market_shin = float(np.mean(log_loss(pooled_market_shin, pooled_one_hot)))
    pooled_brier_naive = float(np.mean(brier(pooled_naive_probs, pooled_one_hot)))
    pooled_brier_market_prop = float(np.mean(brier(pooled_market_prop, pooled_one_hot)))
    pooled_brier_market_shin = float(np.mean(brier(pooled_market_shin, pooled_one_hot)))

    print("=== Phase 4 - per-season benchmark table ===")
    cols_to_print = [
        "season", "n_test_total", "n_scored", "n_dropped_missing_closing",
        "outcome_freq_H", "outcome_freq_D", "outcome_freq_A",
        "rps_naive", "rps_market_prop", "rps_market_shin",
        "rps_gap_naive_minus_market_prop", "rps_prop_minus_shin",
        "log_loss_naive", "log_loss_market_prop",
        "brier_naive", "brier_market_prop",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(results[cols_to_print].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Pooled (all 8 test seasons, scored rows only) ===")
    print(f"  n pooled scored: {len(pooled)}")
    print(f"  rps_naive={pooled_rps_naive:.4f} rps_market_prop={pooled_rps_market_prop:.4f} "
          f"rps_market_shin={pooled_rps_market_shin:.4f}")
    print(f"  rps_gap (naive - market_prop) = {pooled_rps_naive - pooled_rps_market_prop:.4f}")
    print(f"  rps_prop_minus_shin = {pooled_rps_market_prop - pooled_rps_market_shin:.4f}")
    print(f"  log_loss_naive={pooled_log_loss_naive:.4f} log_loss_market_prop={pooled_log_loss_market_prop:.4f} "
          f"log_loss_market_shin={pooled_log_loss_market_shin:.4f}")
    print(f"  brier_naive={pooled_brier_naive:.4f} brier_market_prop={pooled_brier_market_prop:.4f} "
          f"brier_market_shin={pooled_brier_market_shin:.4f}")

    print("\n=== Sanity checks ===")
    if sanity_failures:
        print("  FAILURES:")
        for f in sanity_failures:
            print(f"    - {f}")
    else:
        print("  all passed: n per season correct, probabilities sum to 1, no NaN in scored rows")

    print("\n=== Stop conditions (market RPS >= naive RPS) ===")
    if stop_conditions:
        print("  TRIGGERED:")
        for c in stop_conditions:
            print(f"    - {c}")
    else:
        print("  none triggered: market_prop RPS < naive RPS in every test season")

    # Append pooled row to the saved CSV for completeness.
    pooled_row = {
        "season": "POOLED",
        "n_test_total": np.nan,
        "n_scored": len(pooled),
        "n_dropped_missing_closing": np.nan,
        "train_seasons_n": np.nan,
        "naive_H": np.nan, "naive_D": np.nan, "naive_A": np.nan,
        "outcome_freq_H": np.nan, "outcome_freq_D": np.nan, "outcome_freq_A": np.nan,
        "rps_naive": pooled_rps_naive,
        "rps_market_prop": pooled_rps_market_prop,
        "rps_market_shin": pooled_rps_market_shin,
        "rps_gap_naive_minus_market_prop": pooled_rps_naive - pooled_rps_market_prop,
        "rps_prop_minus_shin": pooled_rps_market_prop - pooled_rps_market_shin,
        "log_loss_naive": pooled_log_loss_naive,
        "log_loss_market_prop": pooled_log_loss_market_prop,
        "log_loss_market_shin": pooled_log_loss_market_shin,
        "brier_naive": pooled_brier_naive,
        "brier_market_prop": pooled_brier_market_prop,
        "brier_market_shin": pooled_brier_market_shin,
    }
    out = pd.concat([results, pd.DataFrame([pooled_row])], ignore_index=True)
    out_path = REPORTS_DIR / "phase4_benchmarks.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
