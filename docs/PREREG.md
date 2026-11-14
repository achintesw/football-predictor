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
### Amendment 1 — 2026-11-14
Made after Phase 2 ingest. Basis: row counts and null counts only; no model results seen.

1. Market benchmark: Pinnacle closing odds (PSCH/PSCD/PSCA), de-vigged (proportional; Shin as cross-check), for all seasons.
2. A match missing Pinnacle closing odds is dropped from scoring for all models and benchmarks in that season. Phase 2 found one such match (2017/18); it is flagged in the match table, not deleted.
3. 2025/26 holdout: Pinnacle closing odds are absent for 192 of 380 matches in the cached file. Per Section 4, the season is benchmarked on opening odds (Pinnacle PSH/PSD/PSA) and labelled as such in every results table. Opening odds are less sharp than closing odds, so gap_closed for that season may be flattered. Bars are unchanged.

### Amendment 2 — 2026-11-14 (made before any 2025/26 results were scored)
1. Finding: in the cached 2025/26 file, Pinnacle opening and closing odds are
   null for 191/380 matches (one contiguous block, 2026-01-16 to season end).
   Opening odds exist for 189 matches, closing for 188. Amendment 1 item 3's
   opening-odds fallback therefore covers only 1 of the 192 closing gaps.
2. Rule (refines Amendment 1 item 3): the 2025/26 market benchmark is computed
   only on the 189 matches with Pinnacle opening odds (PSH/PSD/PSA), de-vigged
   as in Amendment 1. The naive baseline and all models are scored on the same
   189 matches for gap_closed. No substitute bookmaker is used.
3. Bars are unchanged and applied to the 189-match result. Model RPS on all 380
   matches is reported alongside, labelled, and not used for bar decisions.
   Every 2025/26 results table is labelled "Pinnacle opening odds, matches
   before 2026-01-16 only."
4. A bootstrap interval for gap_closed on the 189-match subset is reported; it
   does not alter any bar.
5. For the record: 6 matches in 2014/15-2021/22 lack Pinnacle opening odds
   (closing present). No pre-registered quantity uses them.
