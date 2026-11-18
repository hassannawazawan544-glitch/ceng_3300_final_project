# DTMD frozen final confirmation test — CONFIRMED (both backbones)

Run once on 2026-11-11, following protocol amendment A1 (`docs/v1/V1_PROTOCOL.md`, SHA-256 214d3f68…a1a0).

## Freeze

The freeze manifest `results/v1_dtmd/freeze_manifest.json` (SHA-256 f2506122…6350) hashes every frozen model and every script, including the final-test script itself, before any reserved outcome was read. Frozen components:
- MLP: 3 seeds, 5 epochs (median best epoch from cross-fitting).
- XGBoost: 559 trees (median best iteration + 1).
- Residual-model hyperparameters refitted on all 657 development groups. Kernel MLP: α_u 0.93, α_v 0.81, ℓ 1.12.
- Development-only density fits and solvent-descriptor statistics.

Nothing was refitted or tuned on the test cohort.

## Cohort

Source: BigSolDB v2.2 calibration + final_candidate partitions (solute-hash assignment, PUB-EXT-v1).

| Step | Count |
|---|---|
| Rows (before → after valid target & neutral single-fragment filter) | 8,038 → 6,482 |
| Solutes removed for overlap with development (InChIKey connectivity) | 4 |
| Curves | 529 |
| Groups (≥ 3 solvents, in-range curve endpoints) | 48 groups / 47 solutes / 48 sources |
| **Groups with complete hot density (primary U objective)** | **37 groups / 36 solutes** |

Curves SHA-256: a5c24193…a11a1.

## Primary endpoint

ISO-1 regime (h = 1, σ_meas = 0.04). Solute-macro AUC of the near-best (≤ 5%) success curve, KG_kernel minus the best heuristic. The best heuristic is re-chosen in every bootstrap resample; 95% CI from 4,000 solute-cluster resamples.

| | KG_kernel | KG_null | CF_bayes | CF_classic | PAIR | RANDOM | FULL (order) | Best heuristic | **ΔAUC [95% CI]** | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| MLP | **0.816** | 0.779 | 0.709 | 0.522 | 0.696 | 0.715 | 0.679 | RANDOM | **+0.102 [0.029, 0.134]** | **Confirmed** |
| XGBoost | **0.829** | 0.769 | 0.734 | 0.509 | 0.730 | 0.708 | 0.713 | CF_bayes | **+0.095 [0.031, 0.128]** | **Confirmed** |

## Secondary results

**f90** (budget fraction for near-best in 90% of solutes):

| h | MLP: KG_kernel / best heuristic | XGB: KG_kernel / best heuristic |
|---|---|---|
| 0.5 | 0.25 / 0.75 (PAIR) | 0.25 / 0.80 |
| 1 | 0.46 / 0.70 (cold-first) | 0.31 / 0.70 |
| 2 | 0.54 / 0.60 | 0.54 / 0.60 |
| 5 | 0.47 / 0.50 | 0.50 / 0.50 |
| 10 | 0.45 / 0.46 | 0.48 / 0.46 |

The same cost-dependent pattern as in development: a large saving at h ≤ 1 and about parity at h ≥ 5.

**Stopping rule** (stop when P(within 5% of best) ≥ 0.90), σ = 0.04, shown as success @ cost fraction:

| Regime | KG_kernel | KG_null | Cold-first top-2 | PAIR_half | FULL |
|---|---|---|---|---|---|
| ISO-1 (MLP) | **0.93 @ 0.48** | 0.88 @ 0.60 | 0.83 @ 0.61 | 0.82 @ 0.59 | 0.94 @ 1.00 |
| ISO-1 (XGB) | **0.93 @ 0.52** | 0.89 @ 0.67 | 0.83 @ 0.61 | 0.79 @ 0.59 | 0.94 @ 1.00 |
| ISO-2 (XGB) | 0.92 @ 0.57 | 0.89 @ 0.66 | 0.86 @ 0.48 | 0.82 @ 0.59 | 0.92 @ 1.00 |
| POLY (XGB) | 0.82 @ 0.70 | 0.86 @ 0.83 | — | 0.79 @ 0.52 | 0.83 @ 1.00 |

- In the realistic isothermal setting, KG_kernel with the stopping rule reaches **full-screen success at about half the cost**, and does better than cold-first top-2 at lower cost.
- In the polythermal regime the advantage is small and mixed, consistent with development.

**Not run:** the recovery-fraction objective for the 11 groups without hot density (listed as secondary in A1). It is reported as not done.

## Interpretation and limits

- This **confirms on independent, frozen data** the development finding: decision-targeted, model-guided measurement beats the best simple strategy over the whole budget range when hot and cold measurements cost about the same.
- Kernel versus null shows the same direction as in development: +0.04 to +0.06 AUC, and lower stopping cost.
- The cohort is small (36 solutes), so the intervals are wide.
- The original pre-registered G2 (≥ 20% cost reduction for h = 1–5) **still failed** and must be reported. The confirmation concerns the A1 primary endpoint (anytime AUC at h = 1), declared before this test.
- Everything is retrospective, using literature data and simulated measurement. Source independence and solid-phase consistency of the v2.2 records are not fully audited.
- Interesting side note: the best heuristic differed between backbones (RANDOM with Bayesian decision for MLP, CF_bayes for XGB). Simple orders are close to each other; the Bayesian decision rule does much of their work.

Artifacts: lab `results/v1_dtmd/final_test/` (curves, predictions, per-group scenario and policy outputs, `final_results.json`, `final_cohort_summary.json`); local `publication/v1/results/final_test/`.
