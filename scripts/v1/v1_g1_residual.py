"""DTMD-G1: residual structure and solvent-kernel test (protocol docs/v1/V1_PROTOCOL.md).

Usage (lab root): .venv-graph/bin/python -I scripts/v1/v1_g1_residual.py
For each backbone (mlp, xgb): exact error-mode variance shares; 5-fold cross-validated fit of the
null (independent solvent errors) and kernel residual models; held-out per-group log-likelihood and
conditional-prediction RMSE (2 observed cold endpoints -> remaining solvents' cold+hot residuals).
Outputs results/v1_dtmd/g1_*.json/csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v1_dtmd"
rng = np.random.default_rng(20261111)


def solvent_phi(solvents):
    rows = []
    for s in solvents:
        m = Chem.MolFromSmiles(s)
        rows.append([Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m),
                     rdMolDescriptors.CalcNumHBA(m), Crippen.MolMR(m), rdMolDescriptors.CalcFractionCSP3(m),
                     rdMolDescriptors.CalcNumAromaticRings(m)])
    A = np.array(rows, float)
    return (A - A.mean(0)) / (A.std(0) + 1e-9)


def groups_from(df):
    out = []
    for g, d in df.groupby("group"):
        x_hat_c, x_hat_h = np.exp(d.lnx_cold_pred.values), np.exp(d.lnx_hot_pred.values)
        lr = lambda x: np.log(x) - np.log1p(-np.minimum(x, 1 - 1e-9))  # noqa: E731
        e = np.r_[lr(x_hat_c) - lr(d.x_cold.values), lr(x_hat_h) - lr(d.x_hot.values)]
        out.append({"group": g, "solute": d.solute.iloc[0], "fold": int(d.fold.iloc[0]),
                    "solvents": d.solvent.tolist(), "e": e})
    return out


def shares(G):
    tot = {"common_level": 0.0, "solvent_level": 0.0, "common_contrast": 0.0, "solvent_contrast": 0.0}
    for g in G:
        n = len(g["solvents"])
        c, h = g["e"][:n], g["e"][n:]
        b, d = (c + h).mean() / 2, (h - c).mean() / 2
        u, v = (c + h) / 2 - b, (h - c) / 2 - d
        tot["common_level"] += 2 * n * b ** 2
        tot["common_contrast"] += 2 * n * d ** 2
        tot["solvent_level"] += 2 * (u ** 2).sum()
        tot["solvent_contrast"] += 2 * (v ** 2).sum()
    s = sum(tot.values())
    return {k: round(v / s, 4) for k, v in tot.items()}


class ResidualModel(torch.nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.p = torch.nn.Parameter(torch.tensor([np.log(1.0), np.log(0.2), np.log(0.5), np.log(0.2), np.log(0.1),
                                                  0.0, 0.0, np.log(1.5), 0.0, 0.0]))

    def parts(self):
        sb, sd, su, sv, se = torch.exp(self.p[:5])
        au, av = torch.sigmoid(self.p[5]), torch.sigmoid(self.p[6])
        if not self.kernel:
            au, av = au * 0, av * 0
        return sb, sd, su, sv, se, au, av, torch.exp(self.p[7]), self.p[8], self.p[9]

    def mean_cov(self, phi):
        sb, sd, su, sv, se, au, av, ell, mb, md = self.parts()
        n = phi.shape[0]
        D2 = torch.cdist(phi, phi) ** 2
        K = torch.exp(-D2 / (2 * ell ** 2))
        I = torch.eye(n)
        Ku, Kv = (1 - au) * I + au * K, (1 - av) * I + av * K
        tau = torch.cat([-torch.ones(n), torch.ones(n)])
        one = torch.ones(2 * n)
        Su = torch.cat([torch.cat([Ku, Ku], 1), torch.cat([Ku, Ku], 1)], 0)
        Sv = torch.cat([torch.cat([Kv, -Kv], 1), torch.cat([-Kv, Kv], 1)], 0)
        S = sb ** 2 * torch.outer(one, one) + sd ** 2 * torch.outer(tau, tau) + su ** 2 * Su + sv ** 2 * Sv \
            + se ** 2 * torch.eye(2 * n)
        return mb * one + md * tau, S

    def nll(self, phi, e):
        mu, S = self.mean_cov(phi)
        L = torch.linalg.cholesky(S)
        r = (e - mu).unsqueeze(1)
        a = torch.cholesky_solve(r, L)
        return 0.5 * (2 * torch.log(torch.diagonal(L)).sum() + (r * a).sum() + len(e) * np.log(2 * np.pi))


def fit(G, PHI, kernel, iters=300):
    mdl = ResidualModel(kernel)
    opt = torch.optim.Adam(mdl.parameters(), lr=0.05)
    data = [(PHI[g["solvents"]], torch.tensor(g["e"])) for g in G]
    for it in range(iters):
        opt.zero_grad()
        loss = sum(mdl.nll(p, e) for p, e in data) / len(data)
        loss.backward()
        opt.step()
    return mdl


@torch.no_grad()
def cond_rmse(mdl, phi, e, obs_idx):
    """Observe cold residuals of solvents obs_idx; predict cold+hot of the other solvents."""
    n = phi.shape[0]
    mu, S = mdl.mean_cov(phi)
    o = list(obs_idx)
    t = [i for i in range(n) if i not in o] + [n + i for i in range(n) if i not in o]
    Soo, Sto = S[o][:, o], S[t][:, o]
    pred = mu[t] + Sto @ torch.linalg.solve(Soo, e[o] - mu[o])
    return float(torch.sqrt(((pred - e[t]) ** 2).mean())), float(torch.sqrt(((mu[t] - e[t]) ** 2).mean()))


def boot_ci(per_solute):
    v = np.asarray(per_solute)
    bs = rng.choice(v, (4000, len(v))).mean(1)
    return [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)]


import sys  # noqa: E402

torch.set_num_threads(1)
# Parallel mode: "fold BB K" fits one backbone/fold and writes g1_part_BB_K.csv / .json;
# "aggregate" (default) combines all parts. Same computation as the sequential version.
MODE = sys.argv[1] if len(sys.argv) > 1 else "aggregate"
if MODE == "fold":
    BB_ONLY, K_ONLY = sys.argv[2], int(sys.argv[3])
results = {}
for bb in (["mlp", "xgb"] if MODE != "fold" else [BB_ONLY]):
    df = pd.read_csv(OUT / f"oof_{bb}.csv")
    G = groups_from(df)
    solvents = sorted(df.solvent.unique())
    A = solvent_phi(solvents)
    PHI = {}
    for i, s in enumerate(solvents):
        PHI[s] = torch.tensor(A[i])

    class _P(dict):
        def __getitem__(self, keys):
            return torch.stack([dict.__getitem__(self, k) for k in keys])

    PHI = _P(PHI)
    rec, params = [], []
    if MODE == "aggregate":
        parts = [pd.read_csv(OUT / f"g1_part_{bb}_{k}.csv") for k in range(5)]
        params = [json.load(open(OUT / f"g1_part_{bb}_{k}.json")) for k in range(5)]
        rec = pd.concat(parts).to_dict("records")
    for k in (range(5) if MODE == "fold" else []):
        if k != K_ONLY:
            continue
        rng = np.random.default_rng(20261111 + k)
        trG, teG = [g for g in G if g["fold"] != k], [g for g in G if g["fold"] == k]
        m0, m1 = fit(trG, PHI, False), fit(trG, PHI, True)
        params.append({"fold": k, "null": dict(zip(["sb", "sd", "su", "sv", "se", "au", "av", "ell", "mb", "md"],
                                                   [round(float(x), 4) for x in m0.parts()])),
                       "kernel": dict(zip(["sb", "sd", "su", "sv", "se", "au", "av", "ell", "mb", "md"],
                                          [round(float(x), 4) for x in m1.parts()]))})
        for g in teG:
            phi, e = PHI[g["solvents"]], torch.tensor(g["e"])
            with torch.no_grad():
                l0, l1 = -float(m0.nll(phi, e)), -float(m1.nll(phi, e))
            n = len(g["solvents"])
            r0s, r1s, rps = [], [], []
            if n >= 4:
                for _ in range(20):
                    o = rng.choice(n, 2, replace=False)
                    r1, rp = cond_rmse(m1, phi, e, o)
                    r0, _ = cond_rmse(m0, phi, e, o)
                    r0s.append(r0)
                    r1s.append(r1)
                    rps.append(rp)
            rec.append({"backbone": bb, "fold": k, "group": g["group"], "solute": g["solute"], "n": n,
                        "ll_null": l0 / (2 * n), "ll_kernel": l1 / (2 * n),
                        "rmse_null_cond": np.mean(r0s) if r0s else np.nan,
                        "rmse_kernel_cond": np.mean(r1s) if r1s else np.nan,
                        "rmse_prior_only": np.mean(rps) if rps else np.nan})
        print(bb, "fold", k, "done", flush=True)
    if MODE == "fold":
        pd.DataFrame(rec).to_csv(OUT / f"g1_part_{bb}_{K_ONLY}.csv", index=False)
        json.dump(params[0], open(OUT / f"g1_part_{bb}_{K_ONLY}.json", "w"), indent=1)
        sys.exit(0)
    rng = np.random.default_rng(20261111)
    R = pd.DataFrame(rec)
    R.to_csv(OUT / f"g1_groups_{bb}.csv", index=False)
    ps = R.groupby("solute").mean(numeric_only=True)
    dll = ps.ll_kernel - ps.ll_null
    c = ps.dropna(subset=["rmse_null_cond"])
    drm = c.rmse_kernel_cond - c.rmse_null_cond
    results[bb] = {
        "groups": int(len(R)), "solutes": int(len(ps)),
        "error_mode_shares_pooled_sq": shares(G),
        "heldout_ll_per_endpoint_null": round(float(ps.ll_null.mean()), 4),
        "heldout_ll_per_endpoint_kernel": round(float(ps.ll_kernel.mean()), 4),
        "delta_ll_kernel_minus_null": round(float(dll.mean()), 4), "delta_ll_ci95": boot_ci(dll),
        "cond_rmse_null": round(float(c.rmse_null_cond.mean()), 4),
        "cond_rmse_kernel": round(float(c.rmse_kernel_cond.mean()), 4),
        "cond_rmse_prior_only": round(float(c.rmse_prior_only.mean()), 4),
        "delta_cond_rmse_kernel_minus_null": round(float(drm.mean()), 4), "delta_cond_rmse_ci95": boot_ci(drm),
        "fold_params": params,
    }
    results[bb]["G1_pass"] = bool(results[bb]["delta_ll_ci95"][0] > 0 and results[bb]["delta_cond_rmse_ci95"][1] < 0)
results["G1_pass_both"] = bool(results["mlp"]["G1_pass"] and results["xgb"]["G1_pass"])
json.dump(results, open(OUT / "g1_results.json", "w"), indent=1)
print(json.dumps({b: {k: v for k, v in r.items() if k != "fold_params"} if isinstance(r, dict) else r
                  for b, r in results.items()}, indent=1))
