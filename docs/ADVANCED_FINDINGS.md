# ADV-001 — Findings and interpretation

Completed and reviewed: 2026-11-06 (Asia/Hong_Kong). All numerical claims below derive from saved lab outputs.

## What ran

14 new model/representation families, plus a validation-weighted combination, compared with 3 CLASS-001 references. This required 33 advanced-model training fits: 12 tabular candidate/fixed fits and 21 neural fits across seven three-seed families. Fitted candidates used for selection are not all separate deployed models. A completion marker verifies all six planned Chemprop seed runs reached their full 35-epoch training end; validation chose the checkpoint.

Representations: 217 registered RDKit 2D descriptors per molecule; chiral Morgan count fingerprints (radius 2 or 3, 1,024 bins per molecule); combined descriptors/fingerprints; and 1,613 Mordred 2D descriptors. Separate solute and solvent inputs, plus temperature or inverse temperature. Molecular features derive from structure alone. Descriptor imputation/variance/scaling fits only training rows. Neutral, single-fragment chemistry and source limitations remain as documented in CLASS-001.

Frozen data: 81,197 rows, 1,130 solutes. Train 790 solutes / 57,976 rows; validation 170 / 11,608; test 170 / 11,613. No solute overlap between subsets. This is the same internal development split used for CLASS-001, not a new independent publication test.

## Main comparison

| Model | Test MAE log10(x) | Test RMSE | R² | Recovery MAE, pp | Top mass choice / 53 |
| --- | ---: | ---: | ---: | ---: | ---: |
| class_xgboost | 0.6503 | 0.8379 | 0.6017 | 17.5874 | 25 |
| xgb_mordred | 0.6290 | 0.8195 | 0.6191 | 14.8603 | 27 |
| chemprop_dmpnn | 0.6472 | 0.8418 | 0.5980 | 12.5867 | 33 |
| chemprop_dmpnn_rdkit | 0.6246 | 0.8327 | 0.6067 | 12.2175 | 30 |
| mordred_mlp | 0.6142 | 0.8228 | 0.6159 | 13.0075 | 35 |
| mordred_sobolev_mlp | 0.6217 | 0.8307 | 0.6086 | 12.8355 | 35 |
| validation_convex_ensemble | 0.6047 | 0.8070 | 0.6306 | 12.1812 | 31 |

Recovery columns use 1,309 endpoint groups valid for every method, from 1,316 total pair/source groups. Decision columns use all 53 eligible solute/source groups (51 solutes), each with at least three solvents at directly measured 25/50°C endpoints and in-range hot-solvent density. These are retrospective equilibrium estimates derived from solubility, not measured crystallization yields.

## Which model is selected, and why

Validation MAE identifies the three candidate families GNN+RDKit, Mordred+gradient supervision and Mordred value-only MLP. Searching nonnegative weights in 0.1 steps gives 0.5 / 0.5 / 0.0, respectively. This averages two three-seed model ensembles and reaches validation MAE 0.5651. Its test MAE is 0.6047 and R² 0.6306. This is the current class-project solubility candidate, selected on validation rather than test.

Against CLASS-001 XGBoost on identical test rows, row MAE improves by about 7.0%. On the common recovery subset, MAE falls from 17.5874 to 12.1812 percentage points (about 30.7%). The ensemble chooses the reference top recovered-mass solvent in 31/53 groups versus 25/53. Mean choice regret falls from 47.3017 to 35.6427 g per litre of initial hot pure solvent. Regret is the reference oracle mass minus reference mass for the ML-selected solvent, not an experimental loss measurement.

## Findings that qualify the improvement

- The Mordred value-only MLP has lower single-family test MAE (0.6142) than the GNN+RDKit (0.6246), although the GNN has better validation MAE. This does not establish a universal best representation.
- Mordred gradient supervision improves validation MAE and common recovery MAE modestly (13.0075 to 12.8355 pp), while test solubility MAE becomes slightly worse (0.6142 to 0.6217). Both Mordred variants choose the top mass option in 35/53 groups. Those test decision results are exploratory and were not used to select ensemble weights.
- Fingerprints alone underperform here: XGBoost radius-2 Morgan MAE 0.7615, radius-3 0.7465. Adding 1,024-bin counts to RDKit does not automatically help. This is a representation-specific observation under this cohort/tuning budget, not a rejection of molecular fingerprints generally.
- ExtraTrees has a very low training error but common recovery MAE 24.2332 pp. Solubility fit and process-temperature response are different tasks.
- The combined Morgan/RDKit MLP has common recovery MAE 11.0474 pp but solubility MAE 0.6835. A better recovery ratio can coexist with worse absolute concentration/capacity prediction.
- Mean recovered-mass absolute error does NOT improve for the selected ensemble: 516.774 versus 490.110 g/L for CLASS-001 XGBoost over the same 957 density-covered native endpoint groups. Median error improves from 36.149 to 33.107 g/L. Large concentration errors create a heavy tail. Do not present the ensemble as uniformly better or process-ready.

## Physical validity and uncertainty

GNN+RDKit predicts x >= 1 on 6 test rows and has 3 invalid endpoint groups. Mordred+gradient has 2 such rows and 2 invalid endpoint groups; the inverse-T model also produces invalid values. The selected ensemble has zero invalid test rows or endpoint groups, but does produce invalid predictions on some training/validation rows. This empirical observation is not a physical guarantee or a constraint built into the architecture. Invalid endpoint groups are excluded and counted, never clipped into plausible recovery.

A 2,000-draw paired solute-bootstrap gives the ensemble-minus-baseline macro-solute MAE difference -0.04537, with a 95% interval [-0.09538, +0.00528]. It crosses zero. Accordingly, we do not claim a conclusive statistically significant improvement. These intervals are conditional sampling variation on one explored split; they are not prediction intervals and are not corrected for multiple comparisons/model selection. Model-seed variation and source/polymorph uncertainty require more work.

## Chemistry and process assumptions

Density is used to convert fixed solvent mass to initial pure-solvent volume. For r=x/(1-x), q=r Msolute/Msolvent, recovered mass per litre of initial hot solvent is 1000 rho_hot max(0,q_hot-q_cold). Density cancels from recovery fraction on a fixed solvent-amount basis. Pure-solvent density does not establish true solution molarity.

Do not silently equate neutral SMILES with non-ionizing solutes or identical solid forms. Source-specific curves still lack a complete phase/pH/protocol audit. Endpoints at 25/50°C are a supported comparison example rather than optimized or universally safe operating temperatures. Kinetics, supersaturation trajectory, isolation losses, impurities, solvent recovery, heat duty and safety have not been validated.

## Reproduction, integrity and literature comparison

All-row prediction files match source-row IDs, targets and split membership. Seven neural ensembles each match the mean of their three seed files within float/CSV tolerance. The independent dimensional mass-balance check passes. Predictions are finite; physical-domain failures are separately counted. Scientific plots were inspected and overlapping labels repaired.

Primary code: scripts/prepare_advanced.py, advanced_common.py, train_advanced_tabular.py, train_advanced_mlp.py, train_advanced_gnn.py, prepare_mordred.py, train_mordred_models.py, train_mordred_tree.py, evaluate_advanced.py, plot_advanced.py. Full models/preprocessing/logs remain in the lab project. The local lab_baseline/results/advanced_v1 copy contains portable metrics, held-out predictions, process calculations and figures rather than every large model checkpoint.

[Published-method comparison](PUBLISHED_COMPARISON.md) separates common-split Chemprop retraining from adaptations and paper-reported external scores. No claim of outperforming FASTSOLV, DMPNN-MoE or HASolGNN is justified by the current comparisons.

## Next work for the class project

1. Audit the largest recovered-mass errors and the source conditions of a few contrasting solutes.
2. Examine full temperature curves and domain failures for the current candidate and representative failures.
3. Validate one practical, density-supported solvent comparison against original source tables.
