# Machine-Learning-Assisted Solvent Selection for Drug Crystallization

**CENG 3300 class project.** Department of Chemical and Biological Engineering, HKUST.

This repository contains the data, code, trained models and results of our project. We predict how well a drug dissolves in different solvents at different temperatures, use those predictions to choose the best **cooling-crystallization solvent** for a drug the model has never seen, and test whether a smart measurement policy can find that solvent with fewer lab experiments.

---

## Contents

1. [Background and question](#1-background-and-question)
2. [Method overview](#2-method-overview)
3. [Results](#3-results)
4. [Repository layout](#4-repository-layout)
5. [Trained models (weights)](#5-trained-models-weights)
6. [Setup](#6-setup)
7. [How to reproduce](#7-how-to-reproduce)
8. [What is not included](#8-what-is-not-included)
9. [Data sources and references](#9-data-sources-and-references)

---

## 1. Background and question

Drugs must be purified before use. For solid drugs, the main purification step is **cooling crystallization**: dissolve the drug in a hot solvent, cool it so pure crystals form, and leave the impurities in the liquid. The amount recovered per litre is

```
U  =  dissolved at 50 °C  −  dissolved at 25 °C        (g per L of hot solvent)
```

and it depends strongly on the solvent. For paracetamol, methanol gives about 200 g/L, while 1-butanol gives about 45 g/L. For a **new** drug there is no solubility data. With over a hundred candidate solvents, two temperatures each, and hours per measurement, screening everything is slow and expensive.

**Our question:** can a model trained on published solubility data rank solvents for an unseen drug, and which measurements should a chemist actually run?

## 2. Method overview

### Stage 1: predict solubility

| Step | What we did | Script(s) | Output |
|---|---|---|---|
| Data | BigSolDB 2.0: 103,944 measurements. Kept valid, neutral, single-fragment molecules and removed duplicates, leaving **81,197 rows, 1,130 solutes**. Target = log₁₀(mole-fraction solubility). | `train_linear.py`, `train_class_v1.py` | `results/linear_v1`, `results/class_v1` |
| Density check | Audited the dataset's solvent densities (used to convert mole fraction into g/L). | `audit_density.py` | `results/density_audit_v1` |
| Split | Split **by molecule**, 70/15/15: 790 / 170 / 170 solutes. Test drugs are never seen in training. | `train_class_v1.py` | `results/class_v1/split.csv` |
| Features | 217 RDKit descriptors, 1,613 Mordred descriptors, and Morgan count fingerprints (radius 2 and 3), computed separately for solute and solvent, plus T and 1/T. | `prepare_advanced.py`, `prepare_mordred.py` | `results/advanced_v1/*features*` |
| Models | Ridge regression; XGBoost, extra trees and gradient boosting; neural networks (two-tower MLP; Mordred and RDKit inputs; van 't Hoff and gradient-supervised variants; 3 seeds each); Chemprop D-MPNN graph neural networks (3 seeds); a validation-weighted ensemble. Model choices were made on validation only. | `train_advanced_*.py`, `train_mordred_*.py` | `results/advanced_v1/<model>/` |
| Evaluation | Accuracy metrics, plus **decision metrics**: predicted recovered mass U → does the model pick the truly best solvent? How many g/L are lost (regret)? | `evaluate_advanced.py`, `plot_advanced.py` | `results/advanced_v1/comparison.csv` |

### Stage 2: measure smart (decision-targeted measurement design)

| Step | What we did | Script | Output |
|---|---|---|---|
| Decision cohort | Fitted ln x vs 1/T for each solute–solvent pair from one source, and read it at 25 and 50 °C (inside the measured range only). That gives **657 decision cases** (one solute, one source, ≥ 3 solvents). | `v1/v1_build_cohort.py` | `curves.csv`, `groups.csv` |
| Cross-fitting | 5-fold, solute-grouped cross-fitting of an MLP and XGBoost, so every prediction is for a held-out molecule. | `v1/v1_crossfit_mlp.py`, `v1/v1_crossfit_xgb.py` | `oof_mlp.csv`, `oof_xgb.csv` |
| Gate G1 | Split the prediction error into a shared part (shifts all solvents) and a solvent-specific part, and modelled the solvent-specific part with a solvent-descriptor kernel. Shuffled-descriptor negative control. | `v1/v1_g1_residual.py`, `v1/v1_g1_verify.py` | `g1_results.json`, `g1_verification.json` |
| Gate G2 | Simulated measurement policies: a cost-aware **knowledge-gradient** policy against cold-first, model-ranked, random and uncertainty-sampling strategies, at hot/cold cost ratios from 0.5 to 10. | `v1/v1_g2_policy.py`, `v1/v1_g2b_scenarios.py` | `g2_results.json`, `g2_part_*.csv.gz` |
| Freeze | Locked models and settings, with SHA-256 hashes of every frozen file and script. | `v1/v1_freeze.py` | `frozen/`, `freeze_manifest.json` |
| Final test | A one-time test on **36 new solutes** from BigSolDB 2.2, with no tuning. | `v1/v1_final_test.py` | `final_test/` |

The study protocol was written and hashed **before** the analysis: [`docs/v1/V1_PROTOCOL.md`](docs/v1/V1_PROTOCOL.md).

## 3. Results

### Stage 1: test set of 170 unseen solutes

| Model | MAE (log₁₀ x) ↓ | R² ↑ | Recovery error (pp) ↓ | Best solvent found (of 53) ↑ | Regret (g/L) ↓ |
|---|---:|---:|---:|---:|---:|
| Mean (no model) | 1.057 | −0.00 | 72.9 | 4 | 220 |
| Ridge regression | 0.817 | 0.36 | 18.5 | 18 | 112 |
| XGBoost + Morgan fingerprint | 0.762 | 0.47 | 21.3 | 19 | 142 |
| XGBoost (class model) | 0.650 | 0.60 | 17.6 | 25 | 47 |
| XGBoost + Mordred | 0.629 | 0.62 | 14.9 | 27 | 48 |
| Chemprop D-MPNN | 0.647 | 0.60 | 12.6 | 33 | 43 |
| Chemprop D-MPNN + RDKit | 0.625 | 0.61 | 12.2 | 30 | 54 |
| MLP + Mordred | 0.614 | 0.62 | 13.0 | **35** | **33** |
| Validation-weighted ensemble | **0.605** | **0.63** | 12.2 | 31 | 36 |

The full table for all 18 models is in `results/advanced_v1/comparison.csv`. **The lowest error doesn't mean the best decisions:** the ensemble has the lowest error, but the Mordred MLP picks the best solvent most often.

### Stage 2

| Finding | Value | File |
|---|---|---|
| Share of model error that shifts all solvents equally (can't change the winner) | 71–74% | `g1_results.json` |
| Errors are correlated between chemically similar solvents (held-out log-likelihood gain, kernel vs shuffled control) | +0.14 vs ≈ 0 | `g1_verification.json` |
| Fraction of all measurements needed to find a near-best solvent for 90% of solutes (equal hot/cold cost) | 0.40 vs 0.67 (best simple strategy) | `g2_results.json` |
| Budget saving vs best simple strategy | 33–49% at equal cost; none when hot measurements cost ≥ 5× | `g2_results.json` |
| Final test, 36 new solutes: success, policy with stopping rule vs measuring everything | 0.93 vs 0.94, with about half the measurements | `final_test/final_results.json` |

The pre-registered G2 target (≥ 20% saving at **every** cost ratio) was **not met**. We report it as such. Details are in [`docs/v1/`](docs/v1/).

## 4. Repository layout

```
.
├── BigSolDBv2.0.csv                  input: 103,944 solubility measurements
├── BigSolDBv2.0_densities.csv        input: solvent densities vs temperature
├── requirements-tabular.txt          environment for data, features, trees, stage-2 XGBoost
├── requirements-torch.txt            environment for neural nets, Chemprop, stage-2 MLP and policies
│
├── scripts/
│   ├── train_linear.py               linear baseline (pair split)
│   ├── train_class_v1.py             cleaning, molecule split, ridge + XGBoost, density ablation
│   ├── audit_density.py              density/unit audit
│   ├── prepare_advanced.py           RDKit descriptors, Morgan fingerprints, Chemprop input
│   ├── prepare_mordred.py            Mordred descriptors
│   ├── advanced_common.py            shared loading / feature / scoring helpers
│   ├── train_advanced_tabular.py     XGBoost (RDKit, Morgan, combined), extra trees, HistGB
│   ├── train_advanced_mlp.py         two-tower MLPs (RDKit, combined, van 't Hoff head)
│   ├── train_mordred_models.py       Mordred MLP and gradient-supervised MLP
│   ├── train_mordred_tree.py         XGBoost on Mordred
│   ├── train_advanced_gnn.py         Chemprop D-MPNN (± RDKit features)
│   ├── evaluate_advanced.py          metrics, recovery, best-solvent decisions, bootstrap
│   ├── plot_advanced.py              analysis figures
│   ├── v1/                           stage 2: cohort, cross-fitting, G1, G2, freeze, final test
│   └── soltext/g2_text_augmentation.py   shared network + descriptor code imported by stage 2
│
├── results/
│   ├── linear_v1/                    baseline metrics, predictions, models
│   ├── class_v1/                     split.csv, ridge/XGBoost models, metrics, report.md
│   ├── density_audit_v1/             density coverage and conflicts
│   ├── advanced_v1/                  features, every stage-1 model (weights + predictions),
│   │                                 comparison.csv, environment snapshots
│   ├── v1_dtmd/                      stage 2 outputs (see section 2), frozen/ models, final_test/
│   ├── soltext_g2/features.npz       cached descriptor table used by stage 2
│   └── publication_v1/               BigSolDB 2.2 file + final-test annotations (inputs to the final test)
│
└── docs/
    ├── DATA.md                       dataset inventory and cleaning
    ├── DENSITY_AUDIT.md              density audit
    ├── ADVANCED_MODELS.md            stage-1 model protocol and runtime record
    ├── ADVANCED_FINDINGS.md          stage-1 findings and interpretation
    ├── PUBLISHED_COMPARISON.md       comparison with published solubility models
    ├── DIGITAL_DISCOVERY_PLAN.md     design of the stage-2 study
    └── v1/                           stage-2 protocol (hash-locked), G1/G2/final-test reports
```

Each model folder in `results/advanced_v1/` holds its predictions, metrics and fitted weights. Per-seed runs are in `seed_20261106/`, `seed_20261107/` and `seed_20261108/`.

## 5. Trained models (weights)

| Model | Location | Format |
|---|---|---|
| Linear baseline | `results/linear_v1/*.joblib` | scikit-learn (joblib) |
| Ridge, XGBoost (class models) | `results/class_v1/*.joblib` | scikit-learn / XGBoost (joblib) |
| XGBoost variants, HistGB | `results/advanced_v1/{xgb_*,histgb_rdkit}/` | joblib / XGBoost |
| MLPs (3 seeds each) | `results/advanced_v1/{rdkit_mlp,combined_mlp,rdkit_vanthoff_mlp,mordred_mlp,mordred_sobolev_mlp}/seed_*/` | PyTorch state dicts + preprocessing |
| Chemprop D-MPNN (3 seeds) | `results/advanced_v1/chemprop_dmpnn*/seed_*/fold_0/model_0/model.pt` | Chemprop 1.6.1 checkpoints |
| Stage-2 frozen MLP (3 seeds), scaler, residual model | `results/v1_dtmd/frozen/` | PyTorch + JSON/NPZ |

Example: loading a joblib model

```python
import joblib
model = joblib.load("results/class_v1/xgboost.joblib")
```

## 6. Setup

The work used two Python environments:

```bash
# tabular: data, features, trees, stage-2 XGBoost
python3.11 -m venv .venv-tabular && .venv-tabular/bin/pip install -r requirements-tabular.txt
# torch: neural networks, Chemprop, Mordred, stage-2 MLP and policies
python3.10 -m venv .venv-torch && .venv-torch/bin/pip install -r requirements-torch.txt
```

Full package snapshots from the original runs are in `results/advanced_v1/environment_tabular.txt` and `environment_graph.txt`. Neural models were trained on an NVIDIA RTX 3090 GPU.

## 7. How to reproduce

Run everything from the repository root. Outputs go to `results/`. Several training scripts refuse to overwrite a completed run, so move or rename the existing output folder before you rerun a step. `train_advanced_gnn.py` calls the `chemprop_train` command from the same environment as the Python that runs it.

**Stage 1**

```bash
.venv-tabular/bin/python scripts/train_linear.py
.venv-tabular/bin/python scripts/audit_density.py
.venv-tabular/bin/python scripts/train_class_v1.py
.venv-tabular/bin/python scripts/prepare_advanced.py
.venv-torch/bin/python   scripts/prepare_mordred.py
.venv-tabular/bin/python scripts/train_advanced_tabular.py
.venv-torch/bin/python   scripts/train_advanced_mlp.py
.venv-torch/bin/python   scripts/train_mordred_models.py
.venv-tabular/bin/python scripts/train_mordred_tree.py
.venv-torch/bin/python   scripts/train_advanced_gnn.py
.venv-tabular/bin/python scripts/evaluate_advanced.py
.venv-tabular/bin/python scripts/plot_advanced.py
```

**Stage 2** (`BB` = `mlp` or `xgb`, `K` = fold 0–4)

```bash
.venv-torch/bin/python   scripts/v1/v1_build_cohort.py
.venv-torch/bin/python   scripts/v1/v1_crossfit_mlp.py
.venv-tabular/bin/python scripts/v1/v1_crossfit_xgb.py
.venv-torch/bin/python   scripts/v1/v1_g1_residual.py fold BB K    # then: v1_g1_residual.py aggregate
.venv-torch/bin/python   scripts/v1/v1_g1_verify.py fold BB K      # then: v1_g1_verify.py aggregate
.venv-torch/bin/python   scripts/v1/v1_g2_policy.py fold BB K      # then: v1_g2_policy.py aggregate
.venv-torch/bin/python   scripts/v1/v1_g2b_scenarios.py fold BB K  # then: v1_g2b_scenarios.py aggregate
.venv-torch/bin/python   scripts/v1/v1_freeze.py mlp_resid
.venv-tabular/bin/python scripts/v1/v1_freeze.py xgb
.venv-torch/bin/python   scripts/v1/v1_freeze.py manifest
.venv-torch/bin/python   scripts/v1/v1_final_test.py prep
.venv-tabular/bin/python scripts/v1/v1_final_test.py xgb_predict
.venv-torch/bin/python   scripts/v1/v1_final_test.py evaluate
```

**Integrity check.** The stage-2 protocol and policy code are hash-locked:

```bash
sha256sum -c docs/v1/V1_PROTOCOL_A1.sha256 results/v1_dtmd/g2_protocol_script.sha256 results/soltext_g2_protocol.sha256
```

## 8. What is not included

GitHub limits file size, so three large files are left out. All three can be regenerated:

| File | Size | Regenerate with |
|---|---:|---|
| `results/advanced_v1/extra_trees_rdkit/model.joblib` | 360 MB | `train_advanced_tabular.py` (its predictions and metrics *are* included) |
| `results/advanced_v1/gnn_data/all_rdkit.npy`, `train_rdkit.npy` | 120 MB, 86 MB | `train_advanced_gnn.py` (cached Chemprop feature files) |

Exploratory side studies that are not part of the class project stay on the group's lab machine.

## 9. Data sources and references

- **BigSolDB 2.0 / 2.2:** Krasnov et al., *Scientific Data* **2025**, 12, 1236. Used as published; see the original record for its licence.
- Chemprop: Yang et al., Analyzing learned molecular representations for property prediction, *J. Chem. Inf. Model.* **2019**, 59, 3370.
- Mordred: Moriwaki et al., *J. Cheminform.* **2018**, 10, 4.
- XGBoost: Chen & Guestrin, *KDD* **2016**.
- Knowledge gradient: Frazier, Powell & Dayanik, *INFORMS J. Comput.* **2009**, 21, 599.
- Crystallization background: Nagy & Braatz, *Annu. Rev. Chem. Biomol. Eng.* **2012**, 3, 55.
