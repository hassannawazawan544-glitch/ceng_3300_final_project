"""DTMD-G1 verification: negative control and ranking-relevant conditional test.

Usage (lab root):
  .venv-graph/bin/python -I scripts/v1/v1_g1_verify.py fold BB K     # one backbone/fold job
  .venv-graph/bin/python -I scripts/v1/v1_g1_verify.py aggregate

Per job, refit on training folds exactly as v1_g1_residual.py: null, kernel (real solvent descriptors),
and kernel with SHUFFLED descriptors (descriptor rows permuted among solvents; fixed seed per fold).
Held-out metrics:
  - per-endpoint log-likelihood (null, kernel, shuffled kernel);
  - conditional prediction after 2 observed cold residuals (same subsets for all three models):
      raw RMSE on the remaining solvents' cold+hot residuals, and
      RANKING-RELEVANT RMSE: errors centred within each temperature across the target solvents, so a
      shared shift (which cannot change rankings) is removed;
      Spearman between predicted and true centred residuals (cold) of target solvents (n_target >= 3).
Verification only; the G1 decision rule is unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v1_dtmd"


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
    lr = lambda x: np.log(x) - np.log1p(-np.minimum(x, 1 - 1e-9))  # noqa: E731
    out = []
    for g, d in df.groupby("group"):
        e = np.r_[lr(np.exp(d.lnx_cold_pred.values)) - lr(d.x_cold.values),
                  lr(np.exp(d.lnx_hot_pred.values)) - lr(d.x_hot.values)]
        out.append({"group": g, "solute": d.solute.iloc[0], "fold": int(d.fold.iloc[0]),
                    "solvents": d.solvent.tolist(), "e": e})
    return out


class ResidualModel(torch.nn.Module):  # identical to v1_g1_residual.py
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
        K = torch.exp(-torch.cdist(phi, phi) ** 2 / (2 * ell ** 2))
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
    data = [(PHI(g["solvents"]), torch.tensor(g["e"])) for g in G]
    for _ in range(iters):
        opt.zero_grad()
        (sum(mdl.nll(p, e) for p, e in data) / len(data)).backward()
        opt.step()
    return mdl


@torch.no_grad()
def cond(mdl, phi, e, o):
    n = phi.shape[0]
    mu, S = mdl.mean_cov(phi)
    o = list(o)
    tc = [i for i in range(n) if i not in o]
    t = tc + [n + i for i in tc]
    pred = (mu[t] + S[t][:, o] @ torch.linalg.solve(S[o][:, o], e[o] - mu[o])).numpy()
    true = e[t].numpy()
    m = len(tc)
    raw = float(np.sqrt(((pred - true) ** 2).mean()))
    err = pred - true
    cen = np.r_[err[:m] - err[:m].mean(), err[m:] - err[m:].mean()]
    rank = float(np.sqrt((cen ** 2).mean()))
    sp = np.nan
    if m >= 3:
        sp = float(pd.Series(pred[:m]).rank().corr(pd.Series(true[:m]).rank()))
    return raw, rank, sp


def job(bb, k):
    df = pd.read_csv(OUT / f"oof_{bb}.csv")
    G = groups_from(df)
    solvents = sorted(df.solvent.unique())
    A = solvent_phi(solvents)
    perm = np.random.default_rng(777 + k).permutation(len(solvents))
    real = {s: torch.tensor(A[i]) for i, s in enumerate(solvents)}
    shuf = {s: torch.tensor(A[perm[i]]) for i, s in enumerate(solvents)}
    P_real = lambda ss: torch.stack([real[s] for s in ss])  # noqa: E731
    P_shuf = lambda ss: torch.stack([shuf[s] for s in ss])  # noqa: E731
    trG, teG = [g for g in G if g["fold"] != k], [g for g in G if g["fold"] == k]
    m0, m1, m2 = fit(trG, P_real, False), fit(trG, P_real, True), fit(trG, P_shuf, True)
    rng = np.random.default_rng(20261111 + k)
    rec = []
    for g in teG:
        e, n = torch.tensor(g["e"]), len(g["solvents"])
        pr, ps = P_real(g["solvents"]), P_shuf(g["solvents"])
        with torch.no_grad():
            ll = [-float(m0.nll(pr, e)), -float(m1.nll(pr, e)), -float(m2.nll(ps, e))]
        row = {"backbone": bb, "fold": k, "group": g["group"], "solute": g["solute"], "n": n,
               "ll_null": ll[0] / (2 * n), "ll_kernel": ll[1] / (2 * n), "ll_shuffled": ll[2] / (2 * n)}
        if n >= 4:
            acc = {key: [] for key in ["raw_null", "raw_kernel", "raw_shuf", "rank_null", "rank_kernel",
                                       "rank_shuf", "sp_null", "sp_kernel", "sp_shuf"]}
            for _ in range(20):
                o = rng.choice(n, 2, replace=False)
                for tag, mdl, ph in [("null", m0, pr), ("kernel", m1, pr), ("shuf", m2, ps)]:
                    raw, rank, sp = cond(mdl, ph, e, o)
                    acc["raw_" + tag].append(raw)
                    acc["rank_" + tag].append(rank)
                    acc["sp_" + tag].append(sp)
            row.update({key: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for key, v in acc.items()})
        rec.append(row)
    pd.DataFrame(rec).to_csv(OUT / f"g1v_part_{bb}_{k}.csv", index=False)
    json.dump({"fold": k, "shuffled_kernel": [round(float(x), 4) for x in m2.parts()],
               "kernel": [round(float(x), 4) for x in m1.parts()]}, open(OUT / f"g1v_part_{bb}_{k}.json", "w"))


def aggregate():
    rng = np.random.default_rng(20261111)

    def ci(v):
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        bs = rng.choice(v, (4000, len(v))).mean(1)
        return [round(float(v.mean()), 4), round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)]

    res = {}
    for bb in ["mlp", "xgb"]:
        R = pd.concat([pd.read_csv(OUT / f"g1v_part_{bb}_{k}.csv") for k in range(5)])
        ps = R.groupby("solute").mean(numeric_only=True)
        c = ps.dropna(subset=["raw_null"])
        res[bb] = {
            "dLL_kernel_minus_null": ci(ps.ll_kernel - ps.ll_null),
            "dLL_shuffled_minus_null": ci(ps.ll_shuffled - ps.ll_null),
            "dLL_kernel_minus_shuffled": ci(ps.ll_kernel - ps.ll_shuffled),
            "raw_rmse": {"null": round(float(c.raw_null.mean()), 4), "kernel": round(float(c.raw_kernel.mean()), 4),
                         "shuffled": round(float(c.raw_shuf.mean()), 4)},
            "rank_rmse": {"null": round(float(c.rank_null.mean()), 4), "kernel": round(float(c.rank_kernel.mean()), 4),
                          "shuffled": round(float(c.rank_shuf.mean()), 4)},
            "d_rank_rmse_kernel_minus_null": ci(c.rank_kernel - c.rank_null),
            "d_rank_rmse_shuffled_minus_null": ci(c.rank_shuf - c.rank_null),
            "d_rank_rmse_kernel_minus_shuffled": ci(c.rank_kernel - c.rank_shuf),
            "spearman_target_cold": {"null": round(float(c.sp_null.mean()), 4), "kernel": round(float(c.sp_kernel.mean()), 4),
                                     "shuffled": round(float(c.sp_shuf.mean()), 4)},
            "d_spearman_kernel_minus_null": ci(c.sp_kernel - c.sp_null),
        }
    json.dump(res, open(OUT / "g1_verification.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "fold":
        job(sys.argv[2], int(sys.argv[3]))
    else:
        aggregate()
