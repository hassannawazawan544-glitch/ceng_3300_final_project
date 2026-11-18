# DTMD gate G2: measurement-policy simulation — FAIL as pre-registered (cost-dependent win)

Run 2026-11-11 on the lab machine.
- Protocol: header of `scripts/v1/v1_g2_policy.py`, SHA-256 c5333358…e048, hashed before any policy was scored.
- Smoke test after hashing on one group: all policies reach zero regret once everything is measured. The script was not changed.
- Scale: 576 groups with complete hot density, 5 folds × 2 backbones × 5 hot/cold cost ratios. 10 parallel jobs, 83 s.

## Primary endpoint: f90

f90 is the budget fraction, relative to a full two-temperature screen, needed to find a near-best solvent (within 5% of the best recovered mass) for 90% of solutes. Reduction = 1 − f90(KG_kernel) / f90(best heuristic). 95% CI from a solute-cluster bootstrap.

| h = hot/cold cost | MLP: KG_kernel f90 | MLP: best heuristic (f90) | MLP reduction [CI] | XGB: KG_kernel f90 | XGB: best heuristic (f90) | XGB reduction [CI] |
|---|---:|---|---|---:|---|---|
| 0.5 | 0.34 | PAIR (0.67) | **49% [43, 63]** | 0.34 | PAIR (0.67) | **49% [37, 53]** |
| 1 | 0.40 | Cold-first (0.67) | **40% [30, 43]** | 0.45 | Cold-first (0.67) | **33% [18, 39]** |
| 2 | 0.50 | Cold-first (0.56) | 11% [4, 28] | 0.50 | Cold-first (0.56) | 11% [0, 20] |
| 5 | 0.44 | Cold-first (0.45) | 2% [−11, 17] | 0.45 | Cold-first (0.45) | 0% [−12, 13] |
| 10 | 0.41 | Cold-first (0.40) | −3% [−18, 14] | 0.44 | Cold-first (0.40) | −10% [−19, 8] |

**G2 pass rule:** at least 20% reduction with CI above 0 for every h in {1, 2, 5} and both backbones. Not met: h = 2 gives only 11%, and h = 5 gives about 0%. **G2 FAILS.**

## Secondary results (anytime performance)

Area under the solute-macro success curve (higher is better):

| h | KG_kernel | KG_null | Cold-first + Bayes | Cold-first classic | PAIR | Uncertainty sampling | Random |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 (MLP) | **0.860** | 0.836 | 0.781 | 0.598 | 0.789 | 0.757 | 0.723 |
| 5 (MLP) | **0.888** | 0.872 | 0.875 | 0.766 | 0.779 | 0.817 | 0.718 |
| 1 (XGB) | **0.854** | 0.808 | 0.778 | 0.604 | 0.795 | 0.753 | 0.726 |
| 5 (XGB) | **0.883** | 0.851 | 0.868 | 0.771 | 0.783 | 0.813 | 0.717 |

- **KG_kernel has the highest anytime success at every cost ratio for both backbones.** At high hot cost its margin over cold-first + Bayes is small (+0.01 to +0.02).
- Success at 25% of a full-screen budget, h = 1: KG_kernel 0.78 (MLP) / 0.77 (XGB), against the best heuristic 0.67 / 0.70 and classic cold-first 0.30 / 0.29.
- **Kernel contribution (KG_null minus KG_kernel f90):**
  - XGB: +0.11 to +0.12 budget fraction at h ≤ 2 (CIs exclude 0), +0.08 at h = 5;
  - MLP: +0.10 at h = 1 (CI [0.04, 0.12]); 0–0.04 elsewhere (CIs include 0);
  - the AUC gain is consistently +0.015 to +0.047.

  The solvent-chemistry transfer helps, modestly, consistent with G1.
- Using the Bayesian posterior to make the final choice (CF_bayes against CF_classic) improves classic cold-first a lot at low budgets, even with the same measurement order.

## Interpretation

1. **When hot and cold measurements cost about the same (h ≤ 1), model-guided adaptive measurement cuts the budget needed to find a near-best recovery solvent by roughly a third to a half** compared with the best simple strategy, for two different backbones.
2. **When hot measurements are expensive (h ≥ 2–5), measuring every solvent cold first is about as good**, and the advantage disappears. This repeats the cost dependence seen in PUB-011, now on 576 groups (17× larger) with a principled policy.
3. The solvent-similarity kernel adds a real but secondary benefit. Most of the gain comes from adaptive, value-of-information selection plus Bayesian decisions.
4. The pre-registered claim (≥ 20% across h = 1–5) is **not supported**, and that must be stated. Redefining the claim to "h ≤ 1 only" after seeing these results would be post-hoc. Any regime-specific claim needs to be framed as a finding and confirmed on the frozen final test, not substituted for the original gate.

## Limitations

- Retrospective simulation: measurements are curve-fit values, the noise model is optimistic (σ_η ≈ 0.04 in ln r), and costs are assumed.
- Myopic one-step knowledge gradient. No stopping rule was evaluated, only fixed budgets.
- f90 is a coarse step statistic; AUC is reported alongside it.
- The source independence of groups is not verified.
- Development folds only. The reserved final cohorts are untouched.

## Where this leaves the publication

Candidate paper title: *"When does machine learning reduce solubility screening for crystallization? Error structure, solvent transfer and measurement costs across 576 literature systems."*
- The positive regime (h ≤ 1: 33–49% saving), the null regime (expensive hot measurements), the error-mode analysis and the negative controls form a coherent, decision-relevant evaluation.
- Whether that clears Digital Discovery's novelty bar is uncertain. *Digital Chemical Engineering* is a strong fit. A prospective test, or a demonstration on the frozen final cohort, would strengthen either venue.
