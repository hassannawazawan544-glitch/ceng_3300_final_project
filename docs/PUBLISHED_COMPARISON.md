# Comparison with published solubility methods

Updated: 2026-11-06. This distinguishes an actual same-split computation from numbers reported in papers. Do not combine the two into a state-of-the-art ranking.

## What we computed

ADV-001 reuses the CLASS-001 neutral molecular cohort and frozen solute split: 81,197 measurements, 790 training solutes, 170 validation solutes and 170 test solutes. Primary target is log10 mole-fraction solubility. Epoch/configuration selection uses validation, not test. The split is an internal development benchmark whose outcomes have already been explored.

We retrained the official Chemprop 1.6.1 paired D-MPNN from scratch, with and without expanded RDKit descriptors. We also trained three-seed paired descriptor neural models using 1,613 Mordred descriptors, with and without finite-difference temperature-gradient supervision. Exact scores and process metrics are in `results/advanced_v1/comparison.csv` on the lab machine and in the local summary copy.

These are literature-backed method-family comparisons on identical data. They are not exact reproductions of FASTSOLV or DMPNN-MoE. No pretrained paper weights were used, so these runs do not inherit hidden pretraining overlap. Small differences in scores on a single split do not establish superiority.

## Published results for context

| Published model / evaluation | Paper-reported result | Target / validation | Relation to our computation |
| --- | --- | --- | --- |
| FASTSOLV / FASTPROP, Leeds external test | RMSE 0.95 | log10 mol/L, external unseen solutes | Different concentration definition and test distribution |
| CHEMPROP variant, Leeds external test | RMSE 0.99 | log10 mol/L, external unseen solutes | Same broad graph family, different architecture/training/version |
| FASTPROP and CHEMPROP, SolProp external test | RMSE 0.83 for each | log10 mol/L, overlap-filtered external evaluation | Different dataset/target, not comparable to our test RMSE |
| DMPNN-MoE, reported 10-fold CV | MAE 0.256 ± 0.010; R² 0.863 ± 0.016 | Author repository describes log(S, mol/mol) | Split details/log base must be verified before direct comparison |
| DMPNN-MoE, 12 unseen solutes / 1,221 rows | MAE 0.413 | Different external chemical cohort | Cannot infer a model ranking against our 170 test solutes |
| HASolGNN, BigSolDB | MAE approximately 0.75 | Natural-log solubility; grouped pairs | Different log base and generalization task |

Sources:

- Attia et al., [Data-driven organic solubility prediction at the limit of aleatoric uncertainty](https://pmc.ncbi.nlm.nih.gov/articles/PMC12365040/), Nature Communications (2025), DOI 10.1038/s41467-025-62717-7. External-test figures/results reviewed in the earlier full-text search. The latest direct fetch encountered a browser challenge; it did not invalidate the earlier retrieved text.
- Guo et al., [JCIM article](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c00781), DOI 10.1021/acs.jcim.5c00781; [author DMPNN-MoE code and reported results](https://github.com/GLX1999/DMPNN-MoE). Paper abstract and author repository reviewed. Full fold construction and target transformation still require audit.
- [HASolGNN article](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00407a), Digital Discovery, DOI 10.1039/D5DD00407A. Indexed methods/results specify natural-log errors and grouped pair splitting; direct publisher access failed.

## Differences that matter chemically and statistically

1. Concentration: log mole fraction cannot be converted into log mol/L error by multiplying by a fixed constant. Molecular mass, solvent density and concentration matter. A log-base change alone is a constant factor, but it does not fix a different physical concentration definition.
2. Generalization: a pair-held-out split may still contain the same solute in other solvents during training. Our split holds out solutes; an external unseen-solute dataset is a further test. None of these is interchangeable with random-row CV.
3. Solid state: neutral molecular SMILES do not identify polymorph, hydrate, solvate, amorphous content, equilibrium status or pH. Different curation can change task difficulty substantially.
4. Pretraining: published weights trained on BigSolDB may already know our test molecules. Direct inference without an overlap audit would provide a potentially contaminated comparison.
5. Process decisions: a lower solubility MAE does not guarantee smaller hot/cold difference error, recovery error or top-choice regret. Those require a separate common-condition evaluation, which ADV-001 supplies internally.

## Reproduction ledger

| Method | Status | Main differences / remaining work |
| --- | --- | --- |
| Chemprop paired D-MPNN | Computed from official package | v1.6.1, our features/hyperparameters, own split and mole-fraction target |
| FASTSOLV-style descriptor/Sobolev model | Adapted and computed | Smaller widths, 3 rather than 4 seeds, different target, gradient filter/loss weight and split |
| Exact published FASTSOLV | Not reproduced | Need original training/cohort/unit pipeline and version-matched implementation |
| Exact DMPNN-MoE | Not reproduced | DGL/CUDA compatibility, authors' charge/explicit-H atom features, MoE and full split/training audit |
| Exact HASolGNN | Not reproduced | Architecture, feature/fold audit and compatible implementation |

Claim: we have compared strong molecular representations and literature-backed neural families under a common internal protocol. We have not demonstrated that we outperform the published papers.
