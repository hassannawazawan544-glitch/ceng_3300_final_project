# ADV-001 — Advanced model comparison

Started 2026-11-06 (Asia/Hong_Kong). This is a class-project development benchmark; publication validation is a later stage.

## Fixed evaluation basis

Reuse CLASS-001 cohort and split exactly: 81,197 rows; 790 train, 170 validation and 170 test solutes. Membership fingerprint in `results/advanced_v1/feature_manifest.json`. Existing test outcomes have been inspected, so do not describe this as an untouched publication test. Hyperparameter/epoch selection uses validation MAE, not test results.

## Completed comparisons

- Expanded RDKit 2D descriptors with XGBoost, ExtraTrees and histogram gradient boosting.
- Chiral Morgan count fingerprints: 1,024 hashed bins, radii 2 and 3, separate solute/solvent roles; log1p counts.
- RDKit + Morgan combination.
- Three-seed paired descriptor/fingerprint MLP ensembles and a pair-dependent apparent van’t Hoff head.
- Official Chemprop paired D-MPNN, with temperature/inverse temperature; descriptor-augmented variant; three seeds each.
- 1,613 Mordred 2D descriptors, paired neural models with/without temperature-gradient supervision.

Temperature and inverse T accompany tabular/fingerprint representations. Training-only imputation, constant-feature filtering and scaling. Raw solubility/density labels never enter molecular predictors. Density remains a process-capacity input and has a separate CLASS-001 ablation.

## Literature and reproduction status

- [FASTSOLV author repository](https://github.com/JacksonBurns/fastsolv) and [paper Chemprop training code](https://github.com/JacksonBurns/fastsolv/blob/main/paper/models/chemprop/train.py) support dual-molecule representations and temperature-gradient supervision. Our smaller descriptor models change the descriptors/widths/target/split/loss details; call them adaptations, not exact reproductions.
- [Official Chemprop 1.6.1](https://github.com/chemprop/chemprop/tree/v1.6.1) is retrained from scratch on our frozen split. The lab's reusable GPU environment is Python 3.10, whereas modern Chemprop requires Python 3.11. Version differences from the solubility paper are explicit.
- [DMPNN-MoE author repository](https://github.com/GLX1999/DMPNN-MoE) reviewed. Its graph features include explicit hydrogens, Gasteiger charges and atomwise Crippen terms and it uses DGL. It has not been reproduced in this run; our ordinary Chemprop GNN must not be labelled DMPNN-MoE.
- [RDKit documentation](https://www.rdkit.org/docs/GettingStartedInPython.html) supports descriptor/fingerprint generation.

Paper-reported scores are context only unless data, target units/log base, splits and training overlap match. Predictions from published-model families retrained here will be compared directly on the same rows. No pretrained weights have been used, avoiding hidden pretraining overlap for these fits.

## Runtime and startup issues

- Shared lab environments were not modified. `.venv-graph` is project-local and inherits the existing `esm3` GPU packages.
- Modern Chemprop installation in Python 3.10 failed due to its Python requirement. A separate Python 3.11/PyTorch download was started, then stopped once the existing GPU environment was reusable; it was not used for results.
- Chemprop 1.6.1 CLI initially failed with typed-argument-parser 1.12.0. Pinning 1.8.1 fixed parsing. A zero-epoch startup check then exposed PyTorch 2.6 checkpoint-loading defaults. The training runner enables legacy loading only for its own newly trained local checkpoints. Failed startup logs are preserved and not counted as fitted models.

## Artifacts

Lab: `results/advanced_v1/`; scripts under `scripts/`. Each completed family saves metrics, all-row predictions, fitted preprocessing and weights/checkpoints. Exact package snapshots are recorded. See [ADVANCED_FINDINGS.md](ADVANCED_FINDINGS.md) for verified results and [PUBLISHED_COMPARISON.md](PUBLISHED_COMPARISON.md) for literature comparability.

Additional compatibility issue: Chemprop’s optional RMSE evaluation called an API removed in newer scikit-learn. The training runner omits that optional metric and checkpoints by validation MAE; the independent evaluation calculates RMSE afterward. A completion marker prevents treating a failed run’s initial checkpoint as a finished model. A missing optional Markdown formatter was resolved by dependency-free report formatting. These failures did not contribute fitted results.
