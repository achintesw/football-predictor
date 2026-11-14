# Match Predictor + Season Simulator — Plan

**Scope v1:** La Liga only · league tables only (no cups/Europe) · seasons 2014/15–2025/26 (Understat xG window)
**Governance:** Pre-registered bars (Phase 0) · Claude Code produces counts/evidence only · **Verdicts by Ach** at every checkpoint · stop-and-report at access walls, format divergences, architectural questions.

---

## Phase 0 — Pre-registration (no data touched)
- Fill and lock `PREREG.md` (verdict bars, metrics, splits, holdout).
- Done when: every `[ACH]` field is filled and the file is committed.

## Phase 1 — Repo & environment
- Repo `football-predictor`; Python, pandas, DuckDB, statsmodels, scikit-learn, XGBoost, Streamlit.
- Files: `ingest.py`, `features.py`, `models.py`, `simulate.py`, `app.py`.
- `data/raw/` git-ignored, cached, never redistributed.

## Phase 2 — Ingest
- football-data.co.uk `SP1` CSVs: results, shots, opening + closing odds.
- Understat La Liga match-level xG (`soccerdata` or minimal scraper), cached per season.
- **CHECKPOINT (stop & report):** rows per season (expect 380), missing matches, which seasons carry closing-odds columns (assumption to verify: early seasons may lack them).

## Phase 3 — Unified match table
- Team-name mapping table between sources; no fuzzy matching without manual review; unmapped team = build failure.
- One row per match: date, teams, goals, xG, de-vigged odds (proportional; Shin as cross-check).
- QA report: goals agree across sources, xG plausibility, no duplicate fixtures.
- **CHECKPOINT (stop & report).**

## Phase 4 — Benchmarks
- Naive base rates (floor) and de-vigged closing odds (ceiling), scored per test season.
- Benchmark levels define the scale for the relative bars in PREREG.

## Phase 5 — Model ladder (each rung must clear its bar over the one below)
1. Elo — home advantage; draws via ordered logit on rating diff; K tuned on train seasons only.
2. Dixon-Coles — Poisson + low-score correction + time decay; full scoreline matrix.
3. xG-augmented DC — fitted on xG or goals/xG blend.
4. XGBoost — pre-match features only (Elo, DC ratings, rolling xG for/against, rest days, promoted flag).
- Leakage guards: every feature timestamp < kickoff; shuffled-target test must land at naive level.

## Phase 6 — Evaluation
- Comparison table across walk-forward test seasons; calibration plots.
- **VERDICT (Ach):** which model feeds the simulator. Clean negatives are written up as findings.

## Phase 7 — Season simulator (La Liga table only)
- Simulate remaining fixtures 10,000× from any date.
- La Liga tiebreakers: head-to-head first (verify current RFEF/LaLiga rules before implementing).
- Outputs: title, top-4/European places, relegation (bottom 3), expected-points distribution. (Verify the number of UCL places per season — it can vary.)
- Backtest from past mid-season snapshots; Brier across all team × outcome pairs.
- **VERDICT (Ach).**

## Phase 8 — Sealed holdout
- 2025/26 opened once. Final verdict against PREREG bars. No re-tuning after.

## Phase 9 — Front end & write-up
- Streamlit: match card (W/D/L + scoreline heatmap), live table odds, model vs market.
- README/methodology: data, validation design, results incl. negatives, limitations.

---

## Timeline (part-time)
- Week 1: Phases 0–3
- Week 2: Phases 4–6
- Week 3: Phases 7–9

## Expansion (after v1 verdict)
- Remaining Big 5 leagues reuse the same pipeline; each league gets its own tiebreaker module and PREREG addendum.
