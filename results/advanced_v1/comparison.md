# Advanced-model comparison

| model | validation_MAE | MAE_log10_x | RMSE_log10_x | R2 | shared_recovery_MAE_pp | top_choice_matches | shared_decision_groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chemprop_dmpnn | 0.6373 | 0.6472 | 0.8418 | 0.5980 | 12.5867 | 33 | 53 |
| chemprop_dmpnn_rdkit | 0.5847 | 0.6246 | 0.8327 | 0.6067 | 12.2175 | 30 | 53 |
| combined_mlp | 0.6912 | 0.6835 | 0.9005 | 0.5400 | 11.0474 | 32 | 53 |
| extra_trees_rdkit | 0.6371 | 0.6480 | 0.8441 | 0.5958 | 24.2332 | 23 | 53 |
| histgb_rdkit | 0.6498 | 0.6557 | 0.8516 | 0.5886 | 16.9847 | 28 | 53 |
| mordred_mlp | 0.5971 | 0.6142 | 0.8228 | 0.6159 | 13.0075 | 35 | 53 |
| mordred_sobolev_mlp | 0.5879 | 0.6217 | 0.8307 | 0.6086 | 12.8355 | 35 | 53 |
| rdkit_mlp | 0.6144 | 0.6393 | 0.8327 | 0.6066 | 12.6827 | 31 | 53 |
| rdkit_vanthoff_mlp | 0.6331 | 0.6488 | 0.8446 | 0.5954 | 15.0823 | 31 | 53 |
| validation_convex_ensemble | 0.5651 | 0.6047 | 0.8070 | 0.6306 | 12.1812 | 31 | 53 |
| xgb_combined | 0.6618 | 0.6761 | 0.8680 | 0.5726 | 18.8401 | 27 | 53 |
| xgb_mordred | 0.6195 | 0.6290 | 0.8195 | 0.6191 | 14.8603 | 27 | 53 |
| xgb_morgan2 | 0.7070 | 0.7615 | 0.9704 | 0.4658 | 21.3451 | 19 | 53 |
| xgb_morgan3 | 0.7136 | 0.7465 | 0.9483 | 0.4898 | 18.1454 | 18 | 53 |
| xgb_rdkit | 0.6305 | 0.6559 | 0.8540 | 0.5863 | 14.5771 | 28 | 53 |
| class_mean | None | 1.0572 | 1.3290 | -0.0019 | 72.8833 | 4 | 53 |
| class_ridge_inverseT_interactions | None | 0.8173 | 1.0650 | 0.3566 | 18.4986 | 18 | 53 |
| class_xgboost | None | 0.6503 | 0.8379 | 0.6017 | 17.5874 | 25 | 53 |

{
  "n_model_entries": 18,
  "test_rows": 11613,
  "test_solutes": 170,
  "endpoint_groups": 1316,
  "shared_valid_endpoint_groups": 1309,
  "common_temperature_rows": 371,
  "eligible_decision_groups": 53,
  "shared_valid_decision_groups": 53,
  "ensemble": {
    "members": [
      "chemprop_dmpnn_rdkit",
      "mordred_sobolev_mlp",
      "mordred_mlp"
    ],
    "weights": [
      0.5,
      0.5,
      0.0
    ],
    "validation_MAE": 0.5650777624827016,
    "rule": "Top three families by validation MAE; nonnegative weights on 0.1 grid summing to 1; no test-based selection. Not an independent new model or calibrated uncertainty."
  },
  "checks": {
    "chemprop_dmpnn": {
      "seeds": 3,
      "max_ensemble_difference": 1.7763568394002505e-15
    },
    "chemprop_dmpnn_rdkit": {
      "seeds": 3,
      "max_ensemble_difference": 1.7763568394002505e-15
    },
    "combined_mlp": {
      "seeds": 3,
      "max_ensemble_difference": 1.0000000010279564e-06
    },
    "mordred_mlp": {
      "seeds": 3,
      "max_ensemble_difference": 8.333333330057258e-07
    },
    "mordred_sobolev_mlp": {
      "seeds": 3,
      "max_ensemble_difference": 8.000000004670937e-07
    },
    "rdkit_mlp": {
      "seeds": 3,
      "max_ensemble_difference": 7.999999995789153e-07
    },
    "rdkit_vanthoff_mlp": {
      "seeds": 3,
      "max_ensemble_difference": 9.000000016357035e-07
    }
  },
  "caveats": "One internal solute-group development split; source and phase dependence unresolved. Ideal equilibrium recovery, not isolated yield. Density interpolation uncertainty not included. Invalid mole-fraction endpoint predictions excluded and counted; shared comparisons restrict to all-method-valid groups. Bootstrap is conditional sampling variability, not a model prediction interval or multiplicity-corrected significance test."
}
