# Linear regression baseline v1

Target: log10 solubility in mole fraction. Inputs: temperature (K), plus solute and solvent molecular weight, logP, TPSA, H-bond donors/acceptors, rotatable bonds and ring count. Ordinary least squares with training-only scaling; 15 features. No measured solubility or density is an input. Mean and temperature-only references use the same split.

Split: 80/20 canonical solute–solvent pairs, seed 42. Every temperature/source record of a pair stays in one partition. This tests new pairs, not necessarily unseen individual molecules. Raw data is preserved; invalid target/temperature/SMILES rows and exact chemical/temperature/target/source duplicates are removed. Conflicting measurements remain. No hyperparameter tuning or test-based selection.

Results:
{
  "input_sha256": "879f55f39db977e295c705e304d011d89d6e10da1daaa3f069181108c1de38b0",
  "raw_rows": 103944,
  "valid_rows_before_dedup": 103645,
  "rows_after_cleaning": 103645,
  "train_rows": 82911,
  "test_rows": 20734,
  "train_pairs": 8962,
  "test_pairs": 2241,
  "solutes": 1448,
  "solvents": 207,
  "temperature_range_K": [
    243.15,
    425.77
  ],
  "seed": 42,
  "metrics": {
    "mean": {
      "MAE_log10_x": 0.9774720392206295,
      "RMSE_log10_x": 1.2515905961155156,
      "R2": -0.002664667590187131
    },
    "temperature_only": {
      "MAE_log10_x": 0.9716904808152165,
      "RMSE_log10_x": 1.2419991436626912,
      "R2": 0.012644109841871587
    },
    "linear_descriptors": {
      "MAE_log10_x": 0.8304381014178964,
      "RMSE_log10_x": 1.076556921080457,
      "R2": 0.2581687517820075
    }
  },
  "recovery_endpoint_groups": 2320,
  "invalid_predicted_endpoint_groups": 0,
  "recovery_MAE_percentage_points": 15.898321752895514,
  "versions": {
    "python": "3.11.16",
    "sklearn": "1.9.1",
    "rdkit": "2026.03.6",
    "numpy": "2.4.6",
    "pandas": "3.0.6"
  }
}

Recovery is an ideal equilibrium estimate for a hot saturated solution cooled at constant solvent amount: r=x/(1-x), recovery=max(0,1-r_cold/r_hot). Endpoints are measured minimum/maximum temperatures within the same test pair and source; replicate log targets are median-aggregated. It is not measured crystallization yield. Assumes equilibrium, no solvent loss, no polymorph changes, and pure-solvent interpretation. No fixed universal process temperatures are imposed. Endpoint intervals differ, so these rows are not a universal solvent ranking.

Limitations: an additive linear model has one common temperature slope for all pairs, cannot capture pair-specific interactions/curvature, and may predict x outside (0,1). Invalid endpoint predictions are excluded from recovery MAE and counted explicitly. Results are one split, not uncertainty estimates or publication-ready validation. Recovery error compounds solubility errors. We have not modeled energy duty, kinetics, safety, or process economics. Densities are intentionally unused: mole-fraction conversion already gives mol solute per mol solvent.

Run from the project root:
`/home/puqing/anaconda3/envs/jupyter/bin/python scripts/train_linear.py`
