# Plan v1 study protocol: decision-targeted measurement design (DTMD)

Written 2026-11-11, before any v1 analysis was run. The design is in [DIGITAL_DISCOVERY_PLAN.md](../DIGITAL_DISCOVERY_PLAN.md). Changes after this point go in a dated amendments section; old text is never edited.

## Cohorts

- **Development cohort:** the BigSolDB 2.0 curated cohort (`results/advanced_v1/cohort.csv`, 81,197 rows, 1,130 solutes).
  - Each (solute, source, solvent) curve is fitted as ln x against 1/T: linear, or quadratic when there are ≥ 5 distinct temperatures and BIC prefers it.
  - Endpoints at 298.15 K and 323.15 K are read only inside the measured range (± 0.5 K).
  - A curve is dropped if its fit RMSE in ln x exceeds 0.15.
  - A decision group is a (solute, source) with ≥ 3 solvents.
- **Folds:** 5 solute-disjoint folds, assigned deterministically from a hash of the canonical solute SMILES.
- **Frozen final test (scored once, after the policy freeze):** reserved BigSolDB v2.2 calibration/final cohorts plus the NIST-verified subset. These are not touched during development.

## Backbones (cross-fitted)

1. MLP on RDKit 2D descriptors of solute and solvent, with a van't Hoff output head (as in SOLTEXT-G2 arm A0).
2. XGBoost on the same descriptors plus 1000/T.

For each fold, train on the other four folds' rows (one inner fold held out for early stopping), then predict ln x at both endpoints for every held-out group curve.

## Residual model (used for G1 and later as the policy prior)

- e = ln r̂ − ln r, with r = x/(1−x).
- Per group, the stacked cold/hot residual vector has mean μ_b·1 + μ_d·τ, where τ = −1 for cold and +1 for hot.
- Covariance:

  Σ = σ_b²·11ᵀ + σ_d²·ττᵀ + σ_u²[(1−α_u)I + α_u K] ⊗ J₊ + σ_v²[(1−α_v)I + α_v K] ⊗ J₋ + σ_η² I

  - J₊ = [[1,1],[1,1]] and J₋ = [[1,−1],[−1,1]];
  - K_ss′ = exp(−‖φ_s − φ_s′‖² / 2ℓ²), where φ = standardized solvent descriptors (logP, TPSA, H-bond donors, H-bond acceptors, MolMR, fraction sp3, aromatic ring count);
  - hyperparameters are fitted by maximum marginal likelihood on training-fold groups only.
- Null (independent) model: α_u = α_v = 0.

## Gate G1 (pass required before any policy code)

On held-out folds, for **both** backbones:

- (a) mean per-group log-likelihood gain of the kernel model over the null, with a solute-cluster bootstrap 95% CI above 0; **and**
- (b) conditional prediction: given the measured cold residuals of a random 2 solvents in the group, predict the remaining solvents' cold and hot residuals. The kernel model's RMSE must be lower than the null's, with the CI excluding 0. This test isolates the solvent-transfer mechanism the policy relies on.

## Pillar A analyses (reported regardless of G1)

- Variance shares of the four error modes across all groups and both backbones.
- Decision aleatoric limit: solutes with ≥ 2 independent sources sharing ≥ 3 solvents with valid endpoints. How often do sources agree on (i) the best recovery-fraction solvent and (ii) the best recovered-mass solvent where hot density is available?

## Later gates (unchanged from the plan)

- **G2:** the policy beats the best heuristic by ≥ 20% in cost at equal success, CI excluding 0, for cost ratios 1–5 and both backbones.
- **G3:** calibration of the posterior P(best).
- **G4:** frozen final test.

---

## Amendment A1 — 2026-11-11 (written after G2, before any new analysis)

Reason: strengthen the study for *Digital Chemical Engineering* with realistic measurement regimes, a practical stopping rule, a noise sensitivity analysis and a single frozen confirmation test. The G1 and G2 results and their verdicts (G1 pass, G2 fail as pre-registered) stand unchanged. Nothing below re-scores them or replaces the original G2 rule.

### A1.1 Measurement regimes

Grounded in published method descriptions: isothermal shake-flask needs separate equilibration per temperature and has hot-sampling errors; polythermal clear-point instruments give a temperature curve per vial run; aqueous inter-lab RSD is about 20%.

- **ISO-1:** isothermal, one endpoint per action, hot/cold cost h = 1.
- **ISO-2:** isothermal, h = 2, hot measurement noise twice the cold noise.
- **POLY:** polythermal, one action = one solvent with both endpoints, cost 1 per solvent.
- **Noise levels:** observations are truth plus Gaussian noise in ln r, with σ_meas ∈ {0.04, 0.10, 0.20}. The posterior uses the same σ (σ_meas² added to the model noise).

### A1.2 Policies and stopping

- KG_kernel and KG_null as in G2. In POLY, actions are per-solvent pairs: the KG of the pair is evaluated by sequential conditioning on its two endpoints with the same Gauss–Hermite nodes.
- **Stopping rule:** stop when the posterior probability that the current choice is within 5% of the best U is ≥ 0.90. Estimated from 400 posterior samples, checked after every action.
- **Heuristics with natural stopping:**
  - CF_top2: all cold endpoints, then hot endpoints for the top 2 by measured cold capacity (the PUB-011 rule); in POLY, the top 2 solvents by predicted U plus all remaining cold endpoints are not available, so CF in POLY = measure the top-k solvents by predicted U with k = ceil(n/2);
  - PAIR_half: predicted-U order, floor((n+2)/2) complete solvents;
  - FULL: everything.

### A1.3 Development analysis (development folds only; G1 fold hyperparameters)

For each regime and noise level, report for every policy: realized success (relative regret ≤ 5%), mean cost fraction, and mean relative regret, solute-macro with cluster bootstrap. Report anytime AUC as well. No pass/fail gate; these are descriptive. **These analyses may not change the policy, the residual model or the primary endpoint below.**

### A1.4 Frozen final confirmation test (scored once)

- **Cohort:** BigSolDB v2.2 calibration + final_candidate partitions (registry `results/publication_v1/external_candidate_registry.csv`, solute-hash assignment from PUB-EXT-v1).
  - Same eligibility as the external development cohort: neutral single-fragment solute and solvent, valid target, eligible structure.
  - Solutes whose InChIKey connectivity block is in the development cohort are excluded.
  - Curves are built exactly as in the DTMD cohort (in-range endpoints, RMSE ≤ 0.15, ≥ 3 solvents per group). Hot density comes from the development density fits only.
- **Frozen models:**
  - MLP (3 seeds) and XGBoost retrained on all development rows with the fixed settings;
  - residual-model hyperparameters (null and kernel) refitted on all 657 development groups' out-of-fold residuals;
  - all scripts and model files hashed before the final outcomes are read.
- **Primary confirmation endpoint (ISO-1, σ_meas = 0.04, U objective):** difference in solute-macro AUC of the success curve, KG_kernel minus the best heuristic (re-selected per bootstrap resample from CF_classic, CF_bayes, PAIR, RANDOM), with a 95% solute-cluster bootstrap CI.
  - **Confirmed** if the CI lies above 0 for both backbones.
- **Secondary:** f90 reduction at h ∈ {0.5, 1, 2, 5, 10}; POLY regime; stopping-rule success and cost; the recovery-fraction objective R for groups without hot density.
- Given the expected size (about 30–50 decisions), results are reported as confirmation estimates with intervals. Failure to confirm is reported as such.
