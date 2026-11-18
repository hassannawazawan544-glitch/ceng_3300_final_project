# DTMD gate G1: solvent-kernel residual structure — PASS (both backbones)

Run 2026-11-11 on the lab machine.
- Protocol: `docs/v1/V1_PROTOCOL.md`, SHA-256 72174fba…e611, written before any v1 analysis.
- Scripts: `scripts/v1/v1_build_cohort.py`, `v1_crossfit_mlp.py`, `v1_crossfit_xgb.py`, `v1_g1_residual.py`.
- The G1 script was switched from a sequential loop to 10 parallel (backbone × fold) jobs plus an aggregation step. The computation and the pass rule were unchanged. Each fold uses its own fixed random seed for subset sampling.

## Cohort

- BigSolDB curated cohort, curve-fitted endpoints at 298.15/323.15 K, read only inside the measured range.
- **657 decision groups, 573 solutes, 599 sources, 121 solvents, 8.0 solvents per group.**
- 576 groups have complete hot-density coverage (needed for recovered mass).
- 5 solute-hash folds of 124–146 groups each.

## Backbones (out-of-fold)

| Backbone | Out-of-fold endpoint MAE, log10 x |
|---|---:|
| RDKit-descriptor MLP (van't Hoff head, 3 seeds) | 0.617 |
| XGBoost (RDKit descriptors + 1000/T) | 0.600 |

## Error-mode shares (pooled squared ln r residuals)

| | Common level | Solvent level | Common contrast | Solvent contrast |
|---|---:|---:|---:|---:|
| MLP | 74.1% | 24.6% | 0.9% | 0.4% |
| XGBoost | 71.1% | 27.4% | 0.9% | 0.6% |

This reproduces PUB-009 (69–78% common level) at about 20× the sample. The common level does not affect the ranking; nearly all decision-relevant error is solvent-level.

## G1 tests (held-out folds; solute-cluster bootstrap, 4,000 resamples)

| | MLP | XGBoost |
|---|---|---|
| Held-out log-likelihood per endpoint, null → kernel | −1.002 → −0.864 | −1.049 → −0.897 |
| **Δ log-likelihood [95% CI]** | **+0.139 [0.111, 0.165]** | **+0.152 [0.123, 0.181]** |
| Conditional RMSE, prior only (no measurement) | 1.666 | 1.631 |
| Conditional RMSE, null after 2 cold measurements | 1.104 | 1.117 |
| Conditional RMSE, kernel after 2 cold measurements | 1.047 | 1.060 |
| **Δ RMSE kernel − null [95% CI]** | **−0.057 [−0.069, −0.045]** | **−0.058 [−0.072, −0.044]** |
| G1 | **PASS** | **PASS** |

## Learned structure (stable across folds and backbones)

- Kernel weight on solvent-level error: α_u ≈ 0.92–0.94. Kernel weight on solvent-contrast error: α_v ≈ 0.79–0.85.
- Length scale ℓ ≈ 1.0–1.3, in standardized solvent-descriptor units (logP, TPSA, H-bond donors/acceptors, MolMR, Fsp3, aromatic rings).
- Interpretation: **about 93% of a model's solvent-specific error co-varies smoothly with solvent chemistry.** A model that is wrong about methanol is wrong in the same direction about ethanol. This is a statistical description, not a proven chemical mechanism.

## Honest reading of the effect size

- Two cold measurements cut residual RMSE from about 1.65 to about 1.10 in ln r (−33%). **Most of that comes from learning the shared level**, which the null model already captures.
- The kernel adds a further, statistically clear but **modest** improvement (about −5% RMSE) on top.
- Since the shared level cannot change rankings, what matters for decisions is how much the kernel improves *ranking-relevant* (solvent-level) predictions. G2 will measure that directly through decision regret against measurement cost. G1 shows the mechanism exists; it does not guarantee a 20% cost reduction.

## Decision aleatoric limit (preliminary)

- Only 37 source pairs (32 solutes) share at least 3 solvents.
- Agreement on the best recovered-mass solvent is 100%, and on the best recovery-fraction solvent 82% (chance 25%).
- The median cross-source difference in ln x is only 0.056. That suggests many pairs re-report the same data, so they are **not established as independent labs**. This needs source-independence checks (authors, institutions) before it can be used as a claim.

## Next: G2 (policy)

- Implement Bayesian updating with this residual model plus cost-aware knowledge-gradient acquisition. Run virtual campaigns on all 657 groups (hot density present for 576).
- Compare with cold-first top-k, model-led pair screen, static anchoring, uncertainty sampling, random, and the independent-residual ablation, at hot/cold cost ratios 0.5–10.
- Pass: ≥ 20% cost reduction at equal success against the best heuristic, CI excluding 0, both backbones.

## Verification (added 2026-11-11, after a re-check request)

Script: `scripts/v1/v1_g1_verify.py` (10 parallel fold jobs). Same fits and subsets as G1, plus two new tests. The G1 rule itself is unchanged.

1. **Reproduction.** Recomputing from the per-group files gives identical ΔLL and ΔRMSE. No solute appears in more than one fold, and curve folds match training-row folds. The gain holds in all 5 folds for both backbones: the kernel has better LL in 75–82% of held-out groups and better conditional RMSE in 57–64%. The LL gain is smaller for groups with 3–4 solvents (+0.07) than for larger groups (+0.14 to +0.18).

2. **Negative control: shuffled solvent descriptors.** Descriptor rows were permuted among solvents, so similarity no longer reflects chemistry, and the kernel model was refitted.

   | | MLP | XGBoost |
   |---|---|---|
   | ΔLL shuffled − null | +0.0007 [−0.0002, 0.0015] | +0.0009 [−0.0001, 0.0019] |
   | ΔLL real − shuffled | +0.138 [0.111, 0.165] | +0.152 [0.121, 0.179] |

   **The entire gain depends on real solvent chemistry.** It is not extra model flexibility.

3. **Ranking-relevant conditional test.** Errors are centred within each temperature across the unobserved solvents, which removes the shared shift that cannot change rankings.

   | | MLP | XGBoost |
   |---|---|---|
   | Ranking RMSE: null / kernel / shuffled | 0.781 / **0.752** / 0.780 | 0.789 / **0.762** / 0.789 |
   | Δ kernel − null [95% CI] | **−0.029 [−0.039, −0.018]** | **−0.026 [−0.038, −0.015]** |
   | Spearman of predicted vs true solvent corrections (cold, ≥ 3 target solvents): kernel / shuffled | 0.21 / 0.01 | 0.20 / −0.00 |

   Under the null model, two measurements shift all unobserved solvents by the same amount, so they carry *no* ranking information (Spearman undefined). Under the kernel model they do, but only weakly (ρ ≈ 0.2).

### Correction to the interpretation above

The sentence "about 93% of a model's solvent-specific error co-varies smoothly with solvent chemistry" over-reads the fitted prior weight α_u. With ℓ ≈ 1.1 and the chemically similar solvents typical within a group, part of the kernel behaves like an extra shared component.

The **predictive** ranking benefit is what matters, and it is small: ranking-relevant error falls only about 3–4% after two measurements, and the correction correlates about 0.2 with the truth. The mechanism is real and chemically grounded, but weak.

### Consequence for G2

G1 stands as passed: the pre-registered criteria are met and confirmed by the negative control. However, the size of the ranking gain makes a ≥ 20% cost reduction over the best heuristic **uncertain**. The policy's value may come mostly from measuring adaptively rather than from solvent transfer. G2 must include the independent-residual ablation to show how much the kernel contributes, and the paper must report it honestly whichever way it comes out.
