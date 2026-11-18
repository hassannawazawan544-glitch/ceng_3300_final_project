# Digital Discovery publication plan

Drafted 2026-11-11. This is a research proposal, not a result. All claims below are hypotheses to test. The literature search was targeted (2024–2026 web/arXiv/ChemRxiv), not exhaustive; a full prior-art read of the closest papers is step 1 of the plan.

## 1. Bottom line

The current project is **not yet a Digital Discovery paper**. Digital Discovery asks for a "significant development" in originality and contribution. What we have today is a careful evaluation of existing models on 34–53 decisions, plus a finding that a cheap cold-first heuristic matches ML. That is useful, but small and mostly negative.

Recommended paper, building directly on what we already found:

> **Decision-targeted measurement design for crystallization solvent selection.** Learn the *structure* of solubility-model errors (which errors are shared, which are solvent-specific, and how solvents co-vary) from hundreds of literature systems. Then use that structure in a Bayesian, cost-aware policy that decides *which solubility measurement to make next* (which solvent, hot or cold) to find the best temperature-swing recovery solvent for a new solute with the fewest measurements.

Working title: *Measure less, choose better: learned error structure enables decision-targeted experimental design for crystallization solvent selection.*

Why this can be novel: no paper we found closes the loop between (a) ML solubility predictions, (b) the nonlinear two-temperature recovery objective, and (c) adaptive, cost-aware choice of real measurements. Neighbouring work only covers static anchoring, few-shot accuracy, or Bayesian optimization of *predicted* solubility (Section 3).

Why it is feasible now: fitting each curve inside its measured range turns BigSolDB 2.0 into **833 source-specific decision groups for 730 solutes** (25→50 °C, ≥3 solvents per group, average 8.1 solvents). That is about 25× the 34 decisions behind the current results, and it fixes the sample-size weakness.

## 2. Honest assessment of the current work against the Digital Discovery bar

| What we have | Why it is not enough alone |
|---|---|
| 15 model families, ensemble test MAE 0.605 | Accuracy is near the reported aleatoric limit of about 0.5–1 logS [2]. More architectures give small, non-novel gains (e.g., Solvaformer MAE 0.643 [6]). |
| "Lower MAE ≠ better solvent choice" | True but broad; already argued in ranking work [7, 9]. |
| Error decomposition: shared level is 69–78% of squared error but rank-invariant | Elementary algebra. Valuable as *insight*, not as a contribution by itself. |
| Cold-first screen: 33/34 best with 293 vs 450 measurements | Retrospective, 29 solutes; two-stage screening is not new; the model-led advantage reverses with hot-measurement cost. |
| Decision-KL fine-tunes, utility-gap conformal sets | Negative or not better than simple baselines. Keep in the SI as honest negatives. |

What survives and becomes the foundation: the error decomposition, the equal-budget evaluation framework, the cost model, the NIST source-verification pipeline, and the trained backbones.

## 3. Literature landscape (what is already taken)

| Work | What it establishes | What it rules out for us |
|---|---|---|
| Attia et al., *Nat. Commun.* 2025 (FASTSOLV) [2] | Temperature-dependent organic solubility on BigSolDB; accuracy near the aleatoric limit | "A better solubility model" as the headline |
| Al Ibrahim, …, Green, *JACS* 2025 [3] | Thermodynamic fusion cycle (ML activity coefficient, ΔH_fus, T_m); **reference ensembling** uses measured solubilities of the solute in other solvents | Static anchoring / using a few measurements to correct predictions |
| Krasnov et al. 2025, BigSolDB 2.0 [1]; SC3 preprint 2026 [8] | Curated multi-solvent data, leakage-aware benchmarks, uncertainty | A generic curation/benchmark paper |
| EnSol preprint 2026 [5] | Probabilistic GNN (mixture density) with solvent ranking, Spearman about 0.72 | Probabilistic GNN plus ranking as the novelty |
| Solvmate, *Digital Discovery* 2024 [7] | Relative/pairwise solvent ranking | Ranking instead of regression as the novelty |
| SolECOs, *Green Chem.* 2025 [9] | Temperature-dependent API solubility, point-adjusted networks, Monte Carlo uncertainty, crystallization solvent recommendation with sustainability | Uncertainty-aware crystallization solvent recommendation as the novelty |
| *Digital Discovery* 2023 cross-solvent framework [10]; LAMeL meta-learning 2025 [11] | One measured value in another solvent improves prediction; few-shot adaptation cuts MAE | Few-shot accuracy gains as the novelty |
| Buzzi, …, Vermeire, LAPSE 2026 [12] | Bayesian optimization over solvent pairs, composition and T to maximize *predicted* API solubility (in silico surrogate) | BO "for solvents" in general; theirs optimizes predictions, not real measurements |
| Muhieddine et al., *OPRD* 2026 [13]; Ma et al., *I&EC Res.* 2024 [14] | Model-based solvent/process design for crystallization; ML yield/PSD for cooling crystallization | Process-design framing without data-driven measurement design |
| Negoescu, Frazier & Powell 2011 [15]; iKG 2023 [16] | Knowledge-gradient / best-arm identification with correlated priors (drug discovery) | New acquisition theory. Our acquisition must be a domain adaptation, not claimed theory |
| autoHSP 2025 [17]; electrolyte-solvent active learning, *Nat. Commun.* 2025 [18] | Active learning for solubility-type experiments in automated labs | "Active learning for solvents" in general |

**The gap**, which still needs confirming by reading [3], [9], [10], [11], [12] in full: nobody has

1. learned how solubility-model *errors* co-vary across solvents and temperatures at scale and shown which parts matter for decisions;
2. used that structure as a prior in a cost-aware sequential design over **real measurement actions** (solvent × hot/cold);
3. targeted the **nonlinear two-temperature recovery objective** (best-arm identification, not error reduction);
4. benchmarked such policies against strong heuristics across hundreds of leakage-safe virtual campaigns, with calibrated decision probabilities.

## 4. Proposed contribution: three pillars plus an optional fourth

### Pillar A: decision-relevant error structure, at scale (insight)

- Cross-fitting: retrain 3–4 backbones in solute-grouped 5-fold CV over BigSolDB 2.0 (XGBoost-Mordred, Mordred NN, Chemprop D-MPNN, and FASTSOLV retrained from the pinned author code), so every solute has out-of-fold predictions.
- Decompose residuals in ln r (r = x/(1−x)) for all 833 groups: shared level b, solvent level u_s, shared hot/cold contrast d, solvent contrast v_s. This extends PUB-009 from 34 to about 800 decisions.
- **New analysis: solvent error covariance.** Estimate how u_s co-varies between solvents across solutes (for example, do alcohols err together?). Test whether a learned solvent kernel predicts held-out residuals better than independent errors (held-out log-likelihood). Make it chemically interpretable: cluster solvents by error co-movement and compare with H-bond donor/acceptor and polarity classes.
- **New analysis: a decision-level aleatoric limit.** For solutes measured by two or more independent labs with overlapping solvents, how often do the labs *agree on the best recovery solvent*? This bounds achievable decision accuracy. It is the decision analogue of the aleatoric limit in logS reported in [2], and we found no such analysis. Our NIST/author-SI checks already show that reference rankings can be fragile (4 of 6 corroborated winners not robust to compiled uncertainties).

### Pillar B: Bayesian, cost-aware measurement policy (method)

For a new solute with candidate solvents s and endpoints t ∈ {cold, hot}:

- Mean: ln r̂_{s,t} from any backbone.
- Residual prior: ε_{s,t} = b + u_s + τ_t (d + v_s) + η, with τ_cold = −1 and τ_hot = +1.
  - b ~ N(0, σ_b²) and d ~ N(0, σ_d²), shared across solvents;
  - u ~ N(0, σ_u² K), v ~ N(0, σ_v² K′), with K and K′ solvent kernels learned by empirical Bayes from training-fold residuals only;
  - η ~ N(0, σ_η²): measurement and inter-lab noise.
- The posterior is closed-form Gaussian in ln r. Recovered mass U_s = 1000 ρ_hot (M_solute/M_solvent)(r_hot − r_cold) is nonlinear, so the decision posterior uses Monte Carlo samples.
- Action: measure (s, t) at cost c_t. A hot measurement may cost more; we sweep hot/cold cost 0.5–10 as in PUB-011.
- Acquisition: cost-normalized knowledge gradient on U, i.e., the expected gain in the value of the final choice per unit cost. Thompson sampling and top-two sampling are compared.
- Stopping: stop when the posterior expected relative regret of the current choice is ≤ ε (e.g., 2%), or when the posterior P(best) ≥ 0.9, or when the budget runs out.
- Why the error structure matters: a measurement mostly informs the *shared* level b, which cannot change the ranking. The useful information is the solvent-specific part and how it spreads to chemically similar solvents through K. A structure-aware policy should therefore pick measurements that a naive uncertainty-sampling or generic GP-BO policy would not. This is the testable mechanism linking Pillar A to Pillar B.

### Pillar C: open decision-level benchmark (resource)

- A virtual-campaign simulator over the 833 groups (plus 25→40 °C and 20→50 °C windows), leakage-safe solute/source splits, cost models, and the metrics below.
- Released with all baselines, cross-fitted predictions, curve fits, source-verification flags, code, and a Zenodo DOI. Digital Discovery explicitly values reusable data and code.

### Optional Pillar D: prospective demonstration (strongest upgrade)

If any wet-lab access exists: run the policy live on 2–3 new solutes × about 8 solvents (shake-flask, gravimetric or HPLC at 25/50 °C, triplicates, PXRD of residual solids). It is cheap and standard, and it changes the paper from retrospective to prospective. Without it, use the reserved BigSolDB 2.2 / NIST-verified cohort as a frozen pseudo-prospective test.

## 5. Evaluation design (pre-registered before final testing)

**Ground truth.** Fit ln x versus 1/T for each source/solute/solvent curve (van't Hoff, plus a quadratic in 1/T or Apelblat if justified). Read endpoints only inside the measured range, and drop poor fits (RMSE threshold fixed in advance). Sensitivity analysis: exact-measured-endpoint subset only, and the existing 34/53-decision cohorts.

**Splits.** Solute-grouped 5-fold cross-fitting; residual kernels learned only from training folds; campaigns run on held-out-fold solutes. Report results by solute, not by row. Hold source-level clusters together.

**Baselines (all equal-cost):**
1. measure everything;
2. random order;
3. cold-first top-k (our current heuristic);
4. model-led pair screen (zero-shot ranking);
5. static anchoring / reference ensembling in the style of [3]: measure k solvents, shift, choose;
6. uncertainty sampling;
7. generic GP-BO on solvent descriptors without an ML backbone;
8. ablation: our policy with *independent* residuals (no K, no shared level), which isolates the value of the learned structure;
9. backbone swap: XGBoost / Mordred NN / Chemprop / FASTSOLV.

**Primary endpoint (fixed in advance).** Mean measurement cost needed to identify a near-optimal solvent (relative regret ≤ 5%) with ≥ 90% success, compared against the *best* heuristic baseline, with a solute-cluster bootstrap CI.

**Secondary endpoints.** Regret-versus-cost curves (area under the curve); calibration of the posterior P(best) (reliability diagram, i.e., is "90% sure" right 90% of the time?); results across cost ratios, candidate-set sizes and temperature windows; failure analysis (phase changes, near-ties, ionizable solutes).

**Final test.** Freeze code, hyperparameters and the protocol (SHA-256), then evaluate the reserved calibration/final cohorts and the BigSolDB 2.2 new-source cohort once.

## 6. Claims we may and may not make

May claim, if the results hold:
- "Learned solubility-error structure transfers across solutes and can be used for decision-targeted measurement design."
- "Reduces measurements needed to find a near-best recovery solvent by X% versus strong heuristics, at calibrated confidence."
- "First decision-level aleatoric-limit analysis for solvent selection," worded as "to our knowledge," after the full prior-art read.

Must not claim: a new solubility model; a new acquisition theory; ML solvent selection, anchoring, ranking, or uncertainty as novel; measured crystallization yield, safety, or sustainability benefits.

## 7. Go/no-go gates and fallback

| Gate | Test | Pass condition | If it fails |
|---|---|---|---|
| G1 | The learned solvent kernel predicts held-out residuals | Held-out log-likelihood clearly better than independent residuals for ≥ 2 backbones | Pillar B loses its mechanism; pivot to the Pillar A + C benchmark/insight paper |
| G2 | Policy versus best heuristic, development folds | ≥ 20% lower cost at equal success, CI excluding zero, cost ratios 1–5, ≥ 2 backbones | Report honestly as a benchmark; consider *Digital Chemical Engineering* |
| G3 | Calibration | Posterior P(best) reliability within about ±10% | Add conformal recalibration of P(best), checked on calibration folds |
| G4 | Frozen final test | Primary endpoint holds | Publish whatever the result is; the main text must follow the final result |

The fallback is still publishable: Pillar A (error structure plus the decision aleatoric limit) and Pillar C (benchmark) are new and useful on their own, but they make a weaker Digital Discovery case.

## 8. Work plan (rough effort, sequential gates)

1. **Prior-art read and protocol draft (about 1 week).** Read [3], [9]–[12] in full, including SI; write the protocol and the frozen primary endpoint.
2. **Data (about 1–2 weeks).** Curve fitting, the 833-group cohort, source flags (SC3 overlap, NIST), exclusion ledger.
3. **Cross-fitted backbones (about 1–2 weeks, RTX 3090).** 4 backbones × 5 folds; FASTSOLV from pinned author code.
4. **Pillar A analyses (about 1 week).** Error decomposition at scale, solvent kernel, inter-lab decision agreement → gate G1.
5. **Pillar B policy and virtual campaigns (about 2–3 weeks).** Baselines, ablations, cost sweeps → gates G2 and G3.
6. **Frozen final test (about 0.5 week)** → gate G4.
7. **Manuscript, ESI, code/data release (about 2–3 weeks).**

Total is roughly 10–12 weeks of focused work. If Pillar D experiments happen, run them in parallel with step 5.

## 9. Planned main figures

1. Concept: error modes, the decision, and the measurement loop.
2. Error structure at scale: variance shares, rank invariance, solvent error-covariance map with chemical clusters.
3. Decision aleatoric limit: inter-lab agreement on the best solvent versus model accuracy.
4. Regret-versus-cost curves: policy versus all baselines; cost-ratio panel.
5. Calibration of P(best) and stopping behaviour.
6. Case studies (for example, paracetamol), showing which measurements the policy chose and why.
7. Frozen final/prospective test.

## 10. Immediate next steps

1. Full read of the five closest papers (gap confirmation). Record it in `docs/PRIOR_ART_AUDIT.md`.
2. Build the curve-fit ground truth and freeze the 833-group cohort with its exclusion ledger.
3. Start solute-grouped cross-fitting of XGBoost-Mordred and the Mordred NN (fast), then Chemprop and FASTSOLV.
4. Run gate G1 (solvent-kernel test) before writing any policy code.

## References

1. Krasnov et al., BigSolDB 2.0, *Scientific Data* (2025). doi:10.1038/s41597-025-05559-8
2. Attia et al., Data-driven organic solubility prediction at the limit of aleatoric uncertainty, *Nat. Commun.* (2025). https://www.nature.com/articles/s41467-025-62717-7
3. Al Ibrahim, Morgan, Müller, Motati, Green, Accurately predicting solubility curves via a thermodynamic cycle, machine learning, and solvent ensembles, *JACS* 147, 45057 (2025). doi:10.1021/jacs.5c13746; preprint https://www.cambridge.org/engage/chemrxiv/article-details/688e4366fc5f0acb527327da
4. Vermeire, Chung, Green, Predicting solubility limits of organic solutes…, *JACS* (2022). doi:10.1021/jacs.2c01768
5. EnSol: an environment-aware GNN for molecular solubility prediction, arXiv:2609.21151. https://arxiv.org/html/2609.21151
6. Solvaformer, arXiv:2511.09774. https://arxiv.org/abs/2511.09774
7. Solvmate, *Digital Discovery* (2024). doi:10.1039/D4DD00138A
8. SC3: The Multi-Solvent Solubility Challenge and Benchmark, arXiv:2606.07656. https://arxiv.org/pdf/2606.07656
9. Ma, Gao, Mehta, Fu, Li, Benyahia, SolECOs, *Green Chem.* 27, 12621 (2025). doi:10.1039/D5GC04176G
10. A unified ML framework for solubility prediction across organic solvents, *Digital Discovery* (2023). https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00024e
11. Meta-Learning Linear Models for Molecular Property Prediction (LAMeL), arXiv:2509.13527. https://arxiv.org/pdf/2509.13527
12. Buzzi, Caprio, Bongartz, Vermeire, Bayesian optimization of mixed-solvent API solubility, LAPSE:2026.0413. https://psecommunity.org/LAPSE:2026.0413
13. Muhieddine et al., Computer-aided solvent selection for integrated synthesis, crystallization and isolation, *OPRD* (2026). https://strathprints.strath.ac.uk/97391/
14. Ma et al., Digital design of cooling crystallization processes using a machine learning-based strategy, *Ind. Eng. Chem. Res.* (2024). https://pubs.acs.org/doi/10.1021/acs.iecr.4c01651
15. Negoescu, Frazier, Powell, The knowledge-gradient algorithm for sequencing experiments in drug discovery, *INFORMS J. Comput.* (2011).
16. Improving the Knowledge Gradient Algorithm (iKG), arXiv:2310.17901. https://arxiv.org/pdf/2310.17901
17. Autonomous Determination of Hansen Solubility Parameters via Active Learning, ChemRxiv (2025). https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-c7g4r
18. Active learning accelerates electrolyte solvent screening for anode-free lithium metal batteries, *Nat. Commun.* (2025). https://www.nature.com/articles/s41467-025-63303-7
