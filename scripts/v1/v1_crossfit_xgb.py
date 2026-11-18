"""DTMD-CROSSFIT (XGBoost backbone): same folds/inner early-stopping as the MLP script.

Usage (lab root, after v1_crossfit_mlp.py): .venv-advanced/bin/python -I scripts/v1/v1_crossfit_xgb.py
Features: raw RDKit descriptors of solute and solvent + 1000/T. Target ln x. One fixed configuration
(no tuning): 3000 trees max, lr 0.03, depth 8, subsample 0.8, colsample 0.5, early stopping 100 rounds.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v1_dtmd"
TC, TH, LN10 = 298.15, 323.15, np.log(10)

d = np.load(OUT / "design.npz", allow_pickle=True)
X = np.nan_to_num(d["X"].astype(np.float32), nan=np.nan)
idx = {m: i for i, m in enumerate(d["mols"])}
rows = pd.read_csv(OUT / "rows.csv.gz")
cur = pd.read_csv(OUT / "curves.csv")


def feat(df, T=None):
    Tv = df["T"].values if T is None else np.full(len(df), T)
    return np.hstack([X[df.solute.map(idx).values], X[df.solvent.map(idx).values], (1000 / Tv)[:, None]]).astype(np.float32)


out, log = [], []
for k in range(5):
    inner = (k + 1) % 5
    tr, va = rows[(rows.fold != k) & (rows.fold != inner)], rows[rows.fold == inner]
    m = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.5,
                         min_child_weight=5, tree_method="hist", device="cuda", early_stopping_rounds=100,
                         eval_metric="mae", random_state=0)
    m.fit(feat(tr), tr.y.values, eval_set=[(feat(va), va.y.values)], verbose=False)
    held = cur[cur.fold == k]
    pc = np.minimum(m.predict(feat(held, TC)), 0.0)
    ph = np.minimum(m.predict(feat(held, TH)), 0.0)
    out.append(held.assign(lnx_cold_pred=pc, lnx_hot_pred=ph))
    log.append({"fold": k, "best_iteration": int(m.best_iteration)})
    print("fold", k, m.best_iteration, flush=True)
res = pd.concat(out)
res.to_csv(OUT / "oof_xgb.csv", index=False)
pd.DataFrame(log).to_csv(OUT / "crossfit_xgb_log.csv", index=False)
err = np.abs(np.r_[res.lnx_cold_pred - np.log(res.x_cold), res.lnx_hot_pred - np.log(res.x_hot)]) / LN10
print(json.dumps({"oof_endpoint_mae_log10": round(float(err.mean()), 4), "curves": len(res)}, indent=1))
