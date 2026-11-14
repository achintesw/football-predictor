# PREREG — La Liga Match Predictor + Season Simulator

**Status:** LOCKED 2026-11-14 by Ach. No edits except an appended, dated amendment log (amendments never apply to results already seen).

---

## 1. Scope
- League: La Liga (football-data `SP1`, Understat `La_liga`)
- Seasons: 2014/15–2025/26
- Outcome: 1X2 probabilities (home / draw / away) + scoreline distribution
- Simulator: league table only

## 2. Data splits
- Walk-forward by season: train ≤ t−1, test t, for t = 2017/18 … 2024/25 (first 3 seasons burn-in for ratings)
- **Sealed holdout:** 2025/26 — not loaded into any evaluation until Phase 8
- Hyperparameters (Elo K, DC decay ξ, XGBoost params) tuned only inside training windows

## 3. Metrics
- Primary: Ranked Probability Score (RPS), mean per match, lower = better
- Secondary: log loss, Brier (multi-class), calibration curves (reliability by decile)
- Simulator: Brier across all team × outcome pairs (title, top-4, relegation) from mid-season snapshots

## 4. Benchmarks
- Floor: naive base rates (H/D/A frequencies from training seasons)
- Ceiling: de-vigged closing odds (proportional method; Shin as cross-check)
- If closing odds missing for a season → **use opening odds** for that season (decision: Ach). Seasons benchmarked on opening odds are labelled as such in every results table.

## 5. Verdict bars
Bars are expressed as **share of the naive→market RPS gap closed**:
`gap_closed = (RPS_naive − RPS_model) / (RPS_naive − RPS_market)`

| Rung | Bar |
|---|---|
| Elo GO | gap_closed ≥ 0.50 |
| Dixon-Coles GO (vs Elo) | improvement ≥ +0.05 |
| xG-DC GO (vs DC) | improvement ≥ +0.03 |
| XGBoost GO (vs best prior) | improvement ≥ +0.03 |
| "Competitive with market" | gap_closed ≥ 0.85 |
| **Leakage alarm** | gap_closed > 1.00 → halt, audit before any verdict |
| Consistency | rung must win in ≥ 5 of 8 test seasons |

Calibration bar: max decile deviation ≤ 5 percentage points.

## 6. Leakage controls (mandatory, not bars)
- Every feature timestamp < kickoff of the predicted match
- Shuffled-target run must score within noise of naive
- No full-season aggregates (e.g., season xG) used for in-season predictions

## 7. Simulator
- Tiebreakers: LaLiga head-to-head rules, verified against current regulations before implementation
- Runs: 10,000 per snapshot
- Backtest snapshots: MW10, MW19, MW29
- Simulator GO bar: must beat a points-per-game-extrapolation baseline on Brier across all team × outcome pairs (decision: Ach)

## 8. Holdout
- Opened once, after the Phase 6 and Phase 7 verdicts
- Final verdict: does the chosen model hold its bars on 2025/26? Result reported as-is.

## 9. Amendment log
_(empty)_
