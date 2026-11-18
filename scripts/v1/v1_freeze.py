"""DTMD amendment A1.4 freeze: final backbones on all development rows + residual hyperparameters on all groups.

Usage (lab root):
  .venv-graph/bin/python -I scripts/v1/v1_freeze.py mlp_resid   # MLP (3 seeds) + residual fits + phi stats + density
  .venv-advanced/bin/python -I scripts/v1/v1_freeze.py xgb      # XGBoost
  .venv-graph/bin/python -I scripts/v1/v1_freeze.py manifest    # SHA-256 of every frozen file and script
Fixed training lengths (no inner validation): MLP epochs = median(best epoch) from crossfit_mlp_log
(epochs run minus the 8-epoch patience); XGBoost trees = median(best_iteration)+1 from crossfit_xgb_log.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v1_dtmd"
FZ = OUT / "frozen"
FZ.mkdir(exist_ok=True)


def mlp_resid():
    import torch
    sys.path.insert(0, str(ROOT / "scripts" / "soltext"))
    sys.path.insert(0, str(ROOT / "scripts" / "v1"))
    from g2_text_augmentation import Net
    import v1_g1_verify as V
    from rdkit import Chem
    from rdkit.Chem import Crippen, rdMolDescriptors

    torch.set_default_dtype(torch.float32)  # v1_g1_verify sets float64 at import
    d = np.load(OUT / "design.npz", allow_pickle=True)
    X, mols = d["X"].astype(np.float64), list(d["mols"])
    idx = {m: i for i, m in enumerate(mols)}
    rows = pd.read_csv(OUT / "rows.csv.gz")
    med = np.nanmedian(X, 0)
    q1, q3 = np.nanpercentile(X, 25, 0), np.nanpercentile(X, 75, 0)
    iqr = np.where(q3 - q1 > 1e-9, q3 - q1, np.nanstd(X, 0) + 1e-9)
    keep = np.nanstd(X, 0) > 1e-9
    np.savez(FZ / "mlp_scaler.npz", med=med, iqr=iqr, keep=keep)
    Z = torch.tensor(np.nan_to_num((X - med) / iqr, nan=0.0).clip(-10, 10)[:, keep], dtype=torch.float32, device="cuda")
    log = pd.read_csv(OUT / "crossfit_mlp_log.csv")
    n_ep = int(np.median(log.epochs - 8).clip(min=5))
    s = torch.tensor(rows.solute.map(idx).values, device="cuda")
    v = torch.tensor(rows.solvent.map(idx).values, device="cuda")
    T = torch.tensor(rows["T"].values, dtype=torch.float32, device="cuda")
    y = torch.tensor(rows.y.values, dtype=torch.float32, device="cuda")
    for seed in range(3):
        torch.manual_seed(seed)
        net = Net(Z.shape[1]).cuda()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        for ep in range(n_ep):
            net.train()
            perm = torch.randperm(len(rows), device="cuda")
            for i in range(len(rows) // 512):
                b = perm[i * 512:(i + 1) * 512]
                loss = ((net(Z[s[b]], Z[v[b]], T[b]) - y[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        torch.save(net.state_dict(), FZ / f"mlp_seed{seed}.pt")
    # solvent phi statistics over development solvents (raw descriptors, then dev mean/std)
    solvents = sorted(pd.read_csv(OUT / "oof_mlp.csv").solvent.unique())

    def raw_phi(ss):
        out = []
        for x in ss:
            m = Chem.MolFromSmiles(x)
            out.append([Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m),
                        rdMolDescriptors.CalcNumHBA(m), Crippen.MolMR(m), rdMolDescriptors.CalcFractionCSP3(m),
                        rdMolDescriptors.CalcNumAromaticRings(m)])
        return np.array(out, float)

    Araw = raw_phi(solvents)
    mu, sd = Araw.mean(0), Araw.std(0) + 1e-9
    np.savez(FZ / "phi_stats.npz", mu=mu, sd=sd)
    keys = ["sb", "sd", "su", "sv", "se", "au", "av", "ell", "mb", "md"]
    resid = {"mlp_epochs": n_ep}
    for bb in ["mlp", "xgb"]:
        df = pd.read_csv(OUT / f"oof_{bb}.csv")
        G = V.groups_from(df)
        A = (raw_phi(sorted(df.solvent.unique())) - mu) / sd
        P = {x: torch.tensor(A[i], dtype=torch.float64) for i, x in enumerate(sorted(df.solvent.unique()))}
        PHI = lambda ss: torch.stack([P[x] for x in ss])  # noqa: E731
        torch.set_default_dtype(torch.float64)
        m0, m1 = V.fit(G, PHI, False), V.fit(G, PHI, True)
        resid[bb] = {"null": dict(zip(keys, [float(x) for x in m0.parts()])),
                     "kernel": dict(zip(keys, [float(x) for x in m1.parts()]))}
        torch.set_default_dtype(torch.float32)
    json.dump(resid, open(FZ / "residual_params.json", "w"), indent=1)
    # density fits (solvent -> slope, intercept, Tmin, Tmax) from the development cohort only
    coh = pd.read_csv(ROOT / "results/advanced_v1/cohort.csv")
    canon = {x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in coh.solvent.unique() if Chem.MolFromSmiles(x)}
    coh["solvent_c"] = coh.solvent.map(canon)
    dens = {}
    for x, g in coh.dropna(subset=["rho_g_cm3"]).groupby("solvent_c"):
        if g.Temperature_K.nunique() >= 2:
            b = np.polyfit(g.Temperature_K, g.rho_g_cm3, 1)
            dens[x] = [float(b[0]), float(b[1]), float(g.Temperature_K.min()), float(g.Temperature_K.max())]
    json.dump(dens, open(FZ / "density_fits.json", "w"), indent=1)
    print("mlp epochs", n_ep, "residual params", json.dumps(resid)[:400])


def xgb_fit():
    import xgboost as xgb
    d = np.load(OUT / "design.npz", allow_pickle=True)
    X = d["X"].astype(np.float32)
    idx = {m: i for i, m in enumerate(d["mols"])}
    rows = pd.read_csv(OUT / "rows.csv.gz")
    F = np.hstack([X[rows.solute.map(idx).values], X[rows.solvent.map(idx).values], (1000 / rows["T"].values)[:, None]])
    n_trees = int(np.median(pd.read_csv(OUT / "crossfit_xgb_log.csv").best_iteration)) + 1
    m = xgb.XGBRegressor(n_estimators=n_trees, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.5,
                         min_child_weight=5, tree_method="hist", device="cuda", random_state=0)
    m.fit(F, rows.y.values)
    m.save_model(FZ / "xgb.json")
    json.dump({"n_trees": n_trees}, open(FZ / "xgb_meta.json", "w"))
    print("xgb trees", n_trees)


def manifest():
    files = sorted(FZ.glob("*")) + sorted((ROOT / "scripts" / "v1").glob("*.py")) + \
        [ROOT / "scripts/soltext/g2_text_augmentation.py", ROOT / "docs/v1/V1_PROTOCOL.md"]
    man = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
    json.dump(man, open(OUT / "freeze_manifest.json", "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    {"mlp_resid": mlp_resid, "xgb": xgb_fit, "manifest": manifest}[sys.argv[1]]()
