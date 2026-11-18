# CLASS-001: class-project model comparison

Models predict log10 mole-fraction solubility. Split by neutral single-fragment canonical solute, approximate 70/15/15 train/validation/test groups. Models fit train only; XGBoost selected among two prespecified configurations on validation MAE. Test set not used for hyperparameter selection. This is an internal class benchmark, not a chemically independent external dataset; earlier project-wide exploratory analyses have occurred.

Features: 12 descriptors per molecule, temperature and inverse temperature; XGBoost additionally uses separate 256-bit radius-2 Morgan fingerprints. Ridge uses descriptor/inverse-T interactions, alpha=10 fixed. Density ablation fits otherwise identical selected XGBoost settings on identical covered rows, with/without interpolated density. Ablation settings were selected on the broad training/validation cohort; ablation is not separately optimized.

Chemistry limits: neutral molecular structures can still ionize; pH, solid form and equilibrium protocols are unverified in v2.0. Neutral single-fragment filtering is not proof of thermodynamic homogeneity. No polymorph or melting data inferred. In-range median-density interpolation is an approximation; source conflicts retained in raw data and separate audit. Density uncertainty not quantified. No universal monotonicity imposed.

Recovery is an ideal constant-solvent-amount equilibrium estimate, not measured isolated yield. All endpoints in one evaluation group share source and pair, but same solid form is not confirmed. Density-derived mass uses one litre of pure solvent initially at the hot temperature. Endpoint intervals vary, so aggregate endpoint rows are not a fair universal solvent ranking. The 25/50°C case requires measured endpoint data and in-range hot density and is illustrative, not optimized or established safe/feasible. Case chosen deterministically by largest number of solvent alternatives in a held-out solute/source group; no error-based cherry-picking. No new experimental validation or publication claim.

Rows with invalid predicted mole fractions are counted, not clipped into physical plausibility. Macro-MAE intervals bootstrap held-out solutes (1000 draws) and quantify sampling variation conditional on this fitted model, not prediction intervals. No calibrated model uncertainty is claimed.

Artifacts: protocol.json, split.csv, validation_comparison.csv, metrics.json, test_predictions.csv, endpoint_recovery.csv, common_temperature_cases.csv and illustrative_case.csv where supported; fitted joblib models and data-derived figures. To rerun, use a new output version; completed runs are protected from overwrite.

## Results

{
  "run_id": "CLASS-001",
  "seed": 20261106,
  "filters": {
    "raw": 103944,
    "numeric_and_present_smiles": 103944,
    "valid_smiles": 103645,
    "neutral_single_fragment": 81202,
    "exclude_variable_composition_solvent_names": 81197,
    "deduplicated": 81197
  },
  "solutes": 1130,
  "solvents": 179,
  "splits": {
    "train": {
      "rows": 57976,
      "solutes": 790
    },
    "validation": {
      "rows": 11608,
      "solutes": 170
    },
    "test": {
      "rows": 11613,
      "solutes": 170
    }
  },
  "density_covered_rows": 64088,
  "metrics": {
    "mean": {
      "MAE_log10_x": 1.0572176578493782,
      "RMSE_log10_x": 1.3289559554939148,
      "R2": -0.0018811840612953379,
      "macro_solute_MAE": 1.1749889898205097,
      "macro_solute_MAE_bootstrap95": [
        1.0663461998869348,
        1.2931230030690397
      ],
      "n_rows": 11613,
      "n_solutes": 170
    },
    "ridge_inverseT_interactions": {
      "MAE_log10_x": 0.8173292130552521,
      "RMSE_log10_x": 1.0649907746403124,
      "R2": 0.356591585258231,
      "macro_solute_MAE": 0.9676745613166762,
      "macro_solute_MAE_bootstrap95": [
        0.8591087012681217,
        1.082105448359413
      ],
      "n_rows": 11613,
      "n_solutes": 170
    },
    "xgboost": {
      "MAE_log10_x": 0.6502901374262556,
      "RMSE_log10_x": 0.8379055946247501,
      "R2": 0.601723001715661,
      "macro_solute_MAE": 0.6869092541566925,
      "macro_solute_MAE_bootstrap95": [
        0.6275781556344632,
        0.745045067992316
      ],
      "n_rows": 11613,
      "n_solutes": 170
    }
  },
  "selected_xgb": {
    "config": {
      "max_depth": 6,
      "min_child_weight": 10
    },
    "n_trees": 327,
    "validation_MAE": 0.6691583309647885
  },
  "density_ablation": {
    "without_density": {
      "MAE_log10_x": 0.6520268402630844,
      "RMSE_log10_x": 0.8430169077774399,
      "R2": 0.6212783049267239,
      "macro_solute_MAE": 0.6855814428391737,
      "macro_solute_MAE_bootstrap95": [
        0.6257357264043277,
        0.7482080964067553
      ],
      "n_rows": 9250,
      "n_solutes": 168
    },
    "with_density": {
      "MAE_log10_x": 0.6597512128271498,
      "RMSE_log10_x": 0.8495750813010279,
      "R2": 0.6153629228799732,
      "macro_solute_MAE": 0.6888301509624826,
      "macro_solute_MAE_bootstrap95": [
        0.6293271116597087,
        0.7491380519302641
      ],
      "n_rows": 9250,
      "n_solutes": 168
    }
  },
  "recovery": {
    "mean": {
      "endpoint_groups": 1316,
      "valid_groups": 1316,
      "invalid_predicted_groups": 0,
      "MAE_percentage_points": 72.82655481957491,
      "density_covered_groups": 957,
      "recovered_mass_MAE_g_per_L_initial_solvent": 529.4371053405184
    },
    "ridge_inverseT_interactions": {
      "endpoint_groups": 1316,
      "valid_groups": 1316,
      "invalid_predicted_groups": 0,
      "MAE_percentage_points": 18.538964399651483,
      "density_covered_groups": 957,
      "recovered_mass_MAE_g_per_L_initial_solvent": 515.0652432771234
    },
    "xgboost": {
      "endpoint_groups": 1316,
      "valid_groups": 1316,
      "invalid_predicted_groups": 0,
      "MAE_percentage_points": 17.629877170827033,
      "density_covered_groups": 957,
      "recovered_mass_MAE_g_per_L_initial_solvent": 490.1101508169647
    }
  },
  "common_temperature_rows": 371,
  "illustrative_case_rows": 13,
  "versions": {
    "python": "3.11.16",
    "sklearn": "1.9.1",
    "rdkit": "2026.03.6",
    "xgboost": "3.2.0"
  },
  "runtime_seconds": 25.13636302947998
}