"""DTMD-G2: equal-cost, anytime comparison of measurement policies for recovery-solvent selection.

PROTOCOL (fixed and hashed before any policy was scored; 2026-11-11)
  Groups     DTMD cohort groups with complete hot density (U defined for every solvent), both backbones'
             out-of-fold predictions (oof_mlp.csv / oof_xgb.csv). Truth = curve-fit endpoints.
  Objective  U_s = 1000 rho_hot,s (M_solute/M_solvent,s) max(0, r_hot,s - r_cold,s); choose argmax U.
  Prior      ln r ~ N(ln r_hat - mu, Sigma) with the G1 residual model, hyperparameters fitted on the
             training folds of the group's fold (g1_part_<bb>_<k>.json), null or kernel version.
             Observations are the true ln r endpoints with the model's noise variance se^2.
  Actions    measure one endpoint (solvent, cold|hot); cost 1 (cold) or h (hot), h in {0.5,1,2,5,10}.
  Policies   KG_kernel  cost-normalized knowledge gradient on E[U] under the kernel model (proposed)
             KG_null    same under the independent-residual model (ablation: value of solvent transfer)
             US_kernel  largest posterior variance of ln r per unit cost, kernel model
             CF_classic cold endpoints of all solvents (order: predicted U), then hot endpoints in
                        descending measured-cold-capacity order; decision = best fully measured solvent
                        (PUB-011 practice), backbone prediction with measured values plugged in before that
             CF_bayes   CF_classic order, decision = kernel posterior
             PAIR       model-led pair screen: solvents in predicted-U order, cold then hot; classic decision
             RANDOM     uniformly random order, kernel-posterior decision; 10 replicates (mean)
             Bayesian decisions: argmax posterior E[U] with lognormal moments, clipped at 0.
  Curves     after every action record the current choice; relative regret = (U* - U_choice)/U*.
             Budget axis = spent cost / full-screen cost n(1+h). Success = relative regret <= 5%.
             Solute-macro success S(f) on a 0..1 grid (step 0.01), step-function in spent budget.
  Endpoint   f90 = smallest budget fraction with S(f) >= 0.90. Best heuristic = min f90 over
             {CF_classic, CF_bayes, PAIR, RANDOM}. Reduction = 1 - f90(KG_kernel)/f90(best heuristic).
             95% CI from 2,000 solute-cluster bootstrap resamples (best heuristic re-chosen per resample).
  Pass (G2)  reduction >= 20% with CI lower bound > 0, for every h in {1,2,5}, both backbones.
             Also reported: KG_kernel vs KG_null (kernel contribution), AUC of S(f), success at f=0.25/0.5.

Usage (lab root): .venv-graph/bin/python -I scripts/v1/v1_g2_policy.py fold BB K   |   aggregate
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v1_dtmd"
HS = [0.5, 1.0, 2.0, 5.0, 10.0]
GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(16)
GH_W = GH_W / GH_W.sum()
POLICIES = ["KG_kernel", "KG_null", "US_kernel", "CF_classic", "CF_bayes", "PAIR", "RANDOM"]


def solvent_phi(solvents):  # identical to G1
    rows = []
    for s in solvents:
        m = Chem.MolFromSmiles(s)
        rows.append([Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m),
                     rdMolDescriptors.CalcNumHBA(m), Crippen.MolMR(m), rdMolDescriptors.CalcFractionCSP3(m),
                     rdMolDescriptors.CalcNumAromaticRings(m)])
    A = np.array(rows, float)
    return (A - A.mean(0)) / (A.std(0) + 1e-9)


def prior(params, phi, lnr_hat):
    sb, sd, su, sv, se, au, av, ell, mb, md = params
    n = phi.shape[0]
    D2 = ((phi[:, None, :] - phi[None, :, :]) ** 2).sum(-1)
    K = np.exp(-D2 / (2 * ell ** 2))
    I = np.eye(n)
    Ku, Kv = (1 - au) * I + au * K, (1 - av) * I + av * K
    tau = np.r_[-np.ones(n), np.ones(n)]
    one = np.ones(2 * n)
    S = (sb ** 2 * np.outer(one, one) + sd ** 2 * np.outer(tau, tau) + su ** 2 * np.block([[Ku, Ku], [Ku, Ku]])
         + sv ** 2 * np.block([[Kv, -Kv], [-Kv, Kv]]))
    mu_e = mb * one + md * tau
    return lnr_hat - mu_e, S, se ** 2  # prior over ln r (residual e = ln r_hat - ln r)


def update(m, C, j, y, s2):
    c = C[:, j]
    g = c / (C[j, j] + s2)
    return m + g * (y - m[j]), C - np.outer(g, c)


def expU(m, C, A, n):
    v = np.diag(C)
    eh, ec = np.exp(m[n:] + v[n:] / 2), np.exp(m[:n] + v[:n] / 2)
    return np.clip(A * (eh - ec), 0, None)


def kg_scores(m, C, A, n, s2, cand, cost):
    base = expU(m, C, A, n).max()
    out = np.full(len(cand), -np.inf)
    for i, j in enumerate(cand):
        c = C[:, j]
        sd = np.sqrt(C[j, j] + s2)
        g = c / sd
        Cn = C - np.outer(c, c) / (C[j, j] + s2)
        vals = [expU(m + g * z, Cn, A, n).max() for z in GH_X]
        out[i] = (np.dot(GH_W, vals) - base) / cost[j]
    return out


def run_group(gd, params_k, params_n, PHI, hs, rng):
    n = len(gd)
    phi = PHI(gd.solvent.tolist())
    lr = lambda x: np.log(x) - np.log1p(-np.minimum(x, 1 - 1e-12))  # noqa: E731
    lnr_hat = np.r_[lr(np.exp(gd.lnx_cold_pred.values)), lr(np.exp(gd.lnx_hot_pred.values))]
    truth = np.r_[lr(gd.x_cold.values), lr(gd.x_hot.values)]
    A = (1000 * gd.rho_hot.values * gd.solute_mw.values / gd.solvent_mw.values)
    U = np.clip(A * (np.exp(truth[n:]) - np.exp(truth[:n])), 0, None)
    Ustar = U.max()
    regret = lambda s: (Ustar - U[s]) / Ustar if Ustar > 0 else 0.0  # noqa: E731
    pred_U = np.clip(A * (np.exp(lnr_hat[n:]) - np.exp(lnr_hat[:n])), 0, None)
    out = []
    for h in hs:
        cost = np.r_[np.ones(n), np.full(n, h)]
        full = cost.sum()

        def classic_choice(meas):
            done = [s for s in range(n) if meas[s] and meas[n + s]]
            if done:
                return max(done, key=lambda s: U[s])
            z = np.where(meas, truth, lnr_hat)
            return int(np.argmax(A * (np.exp(z[n:]) - np.exp(z[:n]))))

        for pol in POLICIES:
            reps = 10 if pol == "RANDOM" else 1
            for rep in range(reps):
                mdl = params_n if pol == "KG_null" else params_k
                m, C, s2 = prior(mdl, phi, lnr_hat)
                meas = np.zeros(2 * n, bool)
                spent = 0.0
                if pol in ("CF_classic", "CF_bayes"):
                    order_c = list(np.argsort(-pred_U))
                elif pol == "PAIR":
                    seq = [j for s in np.argsort(-pred_U) for j in (s, n + s)]
                elif pol == "RANDOM":
                    seq = list(rng.permutation(2 * n))
                bayes = pol not in ("CF_classic", "PAIR")
                choice = int(np.argmax(expU(m, C, A, n))) if bayes else classic_choice(meas)
                curve = [(0.0, regret(choice))]
                for step in range(2 * n):
                    cand = np.where(~meas)[0]
                    if pol in ("KG_kernel", "KG_null"):
                        sc = kg_scores(m, C, A, n, s2, cand, cost)
                        if not np.isfinite(sc).any() or sc.max() <= 1e-12:
                            sc = np.diag(C)[cand] / cost[cand]
                        j = int(cand[np.argmax(sc)])
                    elif pol == "US_kernel":
                        j = int(cand[np.argmax(np.diag(C)[cand] / cost[cand])])
                    elif pol in ("CF_classic", "CF_bayes"):
                        cold_left = [s for s in order_c if not meas[s]]
                        if cold_left:
                            j = int(cold_left[0])
                        else:
                            capc = A * np.exp(truth[:n])  # measured cold capacity proxy
                            hot_left = [s for s in np.argsort(-capc) if not meas[n + s]]
                            j = int(n + hot_left[0])
                    else:
                        j = int([x for x in seq if not meas[x]][0])
                    meas[j] = True
                    spent += cost[j]
                    m, C = update(m, C, j, truth[j], s2)
                    choice = int(np.argmax(expU(m, C, A, n))) if bayes else classic_choice(meas)
                    curve.append((spent / full, regret(choice)))
                out.append({"h": h, "policy": pol, "rep": rep, "curve": curve})
    return out


def success_on_grid(curve):
    f = np.array([c[0] for c in curve])
    ok = np.array([c[1] <= 0.05 for c in curve])
    # success at budget g: the last action taken with spent fraction <= g
    idx = np.searchsorted(f, GRID + 1e-9, side="right") - 1
    return ok[np.clip(idx, 0, None)].astype(float)


def job(bb, k):
    cur = pd.read_csv(OUT / f"oof_{bb}.csv")
    grp = pd.read_csv(OUT / "groups.csv", index_col=0)
    cur = cur[cur.group.isin(grp.index[grp.U_complete])]
    solvents = sorted(pd.read_csv(OUT / f"oof_{bb}.csv").solvent.unique())  # same phi basis as G1
    A = solvent_phi(solvents)
    PHI_d = {s: A[i] for i, s in enumerate(solvents)}
    PHI = lambda ss: np.stack([PHI_d[s] for s in ss])  # noqa: E731
    mw = {s: Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in set(cur.solute) | set(cur.solvent)}
    cur = cur.assign(solute_mw=cur.solute.map(mw), solvent_mw=cur.solvent.map(mw))
    pj = json.load(open(OUT / f"g1_part_{bb}_{k}.json"))
    keys = ["sb", "sd", "su", "sv", "se", "au", "av", "ell", "mb", "md"]
    pk, pn = [pj["kernel"][x] for x in keys], [pj["null"][x] for x in keys]
    rng = np.random.default_rng(4242 + k)
    rows = []
    for g, gd in cur[cur.fold == k].groupby("group"):
        for r in run_group(gd.reset_index(drop=True), pk, pn, PHI, HS, rng):
            s = success_on_grid(r["curve"])
            rows.append({"backbone": bb, "fold": k, "group": g, "solute": gd.solute.iloc[0], "n": len(gd),
                         "h": r["h"], "policy": r["policy"], "rep": r["rep"],
                         "final_regret": r["curve"][-1][1], **{f"s{int(round(x * 100)):03d}": v for x, v in zip(GRID, s)}})
    pd.DataFrame(rows).to_csv(OUT / f"g2_part_{bb}_{k}.csv.gz", index=False, compression="gzip")
    print(bb, k, "groups", cur[cur.fold == k].group.nunique(), flush=True)


def f90(S):
    hit = np.where(S >= 0.90)[0]
    return GRID[hit[0]] if len(hit) else np.nan


def aggregate():
    rng = np.random.default_rng(20261111)
    scol = [f"s{int(round(x * 100)):03d}" for x in GRID]
    res = {}
    heur = ["CF_classic", "CF_bayes", "PAIR", "RANDOM"]
    for bb in ["mlp", "xgb"]:
        D = pd.concat([pd.read_csv(OUT / f"g2_part_{bb}_{k}.csv.gz") for k in range(5)])
        D = D.groupby(["solute", "group", "h", "policy"])[scol].mean().reset_index()  # RANDOM reps averaged
        res[bb] = {}
        for h in HS:
            d = D[D.h == h]
            # solute-macro curves: groups -> solute mean
            sol = d.groupby(["policy", "solute"])[scol].mean()
            sols = sol.loc["KG_kernel"].index.values
            M = {p: sol.loc[p].loc[sols].values for p in POLICIES}
            curves = {p: M[p].mean(0) for p in POLICIES}
            F = {p: f90(curves[p]) for p in POLICIES}
            best_h = min(heur, key=lambda p: F[p] if np.isfinite(F[p]) else 9)
            red = 1 - F["KG_kernel"] / F[best_h]
            bs_red, bs_kern = [], []
            for _ in range(2000):
                ix = rng.integers(0, len(sols), len(sols))
                Fb = {p: f90(M[p][ix].mean(0)) for p in POLICIES}
                bh = min(heur, key=lambda p: Fb[p] if np.isfinite(Fb[p]) else 9)
                bs_red.append(1 - Fb["KG_kernel"] / Fb[bh])
                bs_kern.append(Fb["KG_null"] - Fb["KG_kernel"])
            res[bb][str(h)] = {
                "solutes": int(len(sols)), "groups": int(d.group.nunique()),
                "f90": {p: (None if not np.isfinite(F[p]) else float(F[p])) for p in POLICIES},
                "auc_success": {p: round(float(curves[p].mean()), 4) for p in POLICIES},
                "success_at_0.25": {p: round(float(curves[p][25]), 3) for p in POLICIES},
                "success_at_0.50": {p: round(float(curves[p][50]), 3) for p in POLICIES},
                "best_heuristic": best_h, "reduction_vs_best_heuristic": round(float(red), 4),
                "reduction_ci95": [round(float(np.nanquantile(bs_red, 0.025)), 4), round(float(np.nanquantile(bs_red, 0.975)), 4)],
                "f90_KGnull_minus_KGkernel": round(float(F["KG_null"] - F["KG_kernel"]), 3),
                "f90_KGnull_minus_KGkernel_ci95": [round(float(np.nanquantile(bs_kern, 0.025)), 3),
                                                   round(float(np.nanquantile(bs_kern, 0.975)), 3)],
            }
            pd.DataFrame({p: curves[p] for p in POLICIES}, index=GRID).to_csv(OUT / f"g2_curves_{bb}_h{h}.csv")
    passed = {bb: all(res[bb][str(h)]["reduction_vs_best_heuristic"] >= 0.2 and res[bb][str(h)]["reduction_ci95"][0] > 0
                      for h in [1.0, 2.0, 5.0]) for bb in res}
    res["G2_pass_by_backbone"] = passed
    res["G2_pass"] = all(passed.values())
    json.dump(res, open(OUT / "g2_results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "fold":
        job(sys.argv[2], int(sys.argv[3]))
    else:
        aggregate()
