"""DTMD-CROSSFIT (MLP backbone): solute-grouped 5-fold cross-fitting; out-of-fold endpoint predictions.

Usage (lab root): .venv-graph/bin/python -I scripts/v1/v1_crossfit_mlp.py
Also writes results/v1_dtmd/design.npz (raw descriptors, molecule index, row table) for the XGBoost script.
Model/optimizer identical to SOLTEXT-G2 arm A0; 3 seeds averaged; inner early-stopping fold = (k+1) mod 5.
"""
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "soltext"))
from g2_text_augmentation import Net, desc  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v1_dtmd"
TC, TH, LN10 = 298.15, 323.15, np.log(10)
dev = "cuda"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m is not None else None


def fold_of(smi):
    return int(hashlib.sha256(smi.encode()).hexdigest(), 16) % 5


coh = pd.read_csv(ROOT / "results/advanced_v1/cohort.csv")
coh["solute"] = coh.solute.map({s: canon(s) for s in coh.solute.unique()})
coh["solvent"] = coh.solvent.map({s: canon(s) for s in coh.solvent.unique()})
coh = coh.dropna(subset=["solute", "solvent"])
rows = pd.DataFrame({"solute": coh.solute, "solvent": coh.solvent, "T": coh.Temperature_K,
                     "y": np.log(coh["Solubility(mole_fraction)"])})
rows["fold"] = rows.solute.map(fold_of)
cur = pd.read_csv(OUT / "curves.csv")

mols = sorted(set(rows.solute) | set(rows.solvent))
g2 = np.load(ROOT / "results/soltext_g2/features.npz", allow_pickle=True)
g2X, g2mols = g2["X"], g2["mols"]  # load once: NpzFile re-reads the archive on every key access
have = {m: i for i, m in enumerate(g2mols)}
missing = [m for m in mols if m not in have]
print("molecules", len(mols), "missing features", len(missing), flush=True)
if missing:
    with Pool(14) as p:
        extra = dict(zip(missing, p.map(desc, missing)))
else:
    extra = {}
X = np.stack([g2X[have[m]] if m in have else extra[m] for m in mols]).astype(np.float64)
del g2X
idx = {m: i for i, m in enumerate(mols)}
np.savez_compressed(OUT / "design.npz", X=X, mols=np.array(mols, dtype=object))
rows.to_csv(OUT / "rows.csv.gz", index=False, compression="gzip")


def scaled(train_mols):
    tr = X[[idx[m] for m in train_mols]]
    med = np.nanmedian(tr, 0)
    q1, q3 = np.nanpercentile(tr, 25, 0), np.nanpercentile(tr, 75, 0)
    iqr = np.where(q3 - q1 > 1e-9, q3 - q1, np.nanstd(tr, 0) + 1e-9)
    keep = np.nanstd(tr, 0) > 1e-9
    Z = np.nan_to_num((X - med) / iqr, nan=0.0).clip(-10, 10)[:, keep]
    return torch.tensor(Z, dtype=torch.float32, device=dev)


def T_(df, cols=("solute", "solvent", "T")):
    return (torch.tensor(df[cols[0]].map(idx).values, device=dev), torch.tensor(df[cols[1]].map(idx).values, device=dev),
            torch.tensor(df[cols[2]].values, dtype=torch.float32, device=dev))


@torch.no_grad()
def pred(net, Z, s, v, T):
    net.eval()
    return torch.cat([net(Z[s[i:i + 8192]], Z[v[i:i + 8192]], T[i:i + 8192]) for i in range(0, len(s), 8192)]).cpu().numpy()


def macro(df, p):
    return float(pd.Series(np.abs(p - df.y.values) / LN10).groupby(df.solute.values).mean().mean())


out, log = [], []
for k in range(5):
    inner = (k + 1) % 5
    tr = rows[(rows.fold != k) & (rows.fold != inner)]
    va = rows[rows.fold == inner]
    Z = scaled(sorted(set(tr.solute) | set(tr.solvent)))
    ts, tv, tT = T_(tr)
    ty = torch.tensor(tr.y.values, dtype=torch.float32, device=dev)
    vs, vv, vT = T_(va)
    held = cur[cur.fold == k]
    hc = T_(held.assign(T=TC))
    hh = T_(held.assign(T=TH))
    pc, ph = [], []
    for seed in range(3):
        torch.manual_seed(seed)
        net = Net(Z.shape[1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        best, state, bad = 1e9, None, 0
        for ep in range(60):
            net.train()
            perm = torch.randperm(len(tr), device=dev)
            for i in range(len(tr) // 512):
                b = perm[i * 512:(i + 1) * 512]
                loss = ((net(Z[ts[b]], Z[tv[b]], tT[b]) - ty[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            vm = macro(va, pred(net, Z, vs, vv, vT))
            if vm < best - 1e-4:
                best, bad, state = vm, 0, {a: b.detach().clone() for a, b in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break
        net.load_state_dict(state)
        pc.append(pred(net, Z, *hc))
        ph.append(pred(net, Z, *hh))
        log.append({"fold": k, "seed": seed, "inner_val_macro_mae": best, "epochs": ep + 1})
        print("fold", k, "seed", seed, round(best, 4), flush=True)
    out.append(held.assign(lnx_cold_pred=np.mean(pc, 0), lnx_hot_pred=np.mean(ph, 0)))
res = pd.concat(out)
res.to_csv(OUT / "oof_mlp.csv", index=False)
pd.DataFrame(log).to_csv(OUT / "crossfit_mlp_log.csv", index=False)
err = np.abs(np.r_[res.lnx_cold_pred - np.log(res.x_cold), res.lnx_hot_pred - np.log(res.x_hot)]) / LN10
print(json.dumps({"oof_endpoint_mae_log10": round(float(err.mean()), 4), "curves": len(res)}, indent=1))
