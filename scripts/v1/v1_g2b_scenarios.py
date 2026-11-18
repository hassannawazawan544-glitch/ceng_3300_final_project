"""DTMD amendment A1.3: realistic measurement regimes, noise and stopping (development folds, descriptive).

Usage (lab root): .venv-graph/bin/python -I scripts/v1/v1_g2b_scenarios.py fold BB K | aggregate
Reuses the frozen G2 code (v1_g2_policy.py) for the prior, updates and expected-U computation.
Regimes (protocol A1.1): ISO-1 (h=1), ISO-2 (h=2, hot noise x2), POLY (one action = both endpoints of a
solvent, cost 1). Observation noise sigma_meas in {0.04, 0.10, 0.20} (ln r). Stopping rule (A1.2):
P(choice within 5% of best U) >= 0.90 from 400 posterior samples. Also run as fixed-rule heuristics:
CF_top2, PAIR_half, FULL.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v1_g2_policy as G  # noqa: E402

OUT = G.OUT
REGIMES = {"ISO-1": (1.0, 1.0, False), "ISO-2": (2.0, 2.0, False), "POLY": (None, 1.0, True)}  # h, hot-noise mult, pair
NOISES = [0.04, 0.10, 0.20]
GH = np.array(np.meshgrid(G.GH_X, G.GH_X)).reshape(2, -1).T
GHW = np.outer(G.GH_W, G.GH_W).ravel()


def upd_multi(m, C, o, y, s2o):
    P = C[np.ix_(o, o)] + np.diag(s2o)
    K = C[:, o] @ np.linalg.inv(P)
    return m + K @ (y - m[o]), C - K @ C[o, :]


def kg_iso(m, C, A, n, s2, cand, cost):
    base = G.expU(m, C, A, n).max()
    out = np.full(len(cand), -np.inf)
    for i, j in enumerate(cand):
        c = C[:, j]
        var = C[j, j] + s2[j]
        g = c / np.sqrt(var)
        Cn = C - np.outer(c, c) / var
        out[i] = (np.dot(G.GH_W, [G.expU(m + g * z, Cn, A, n).max() for z in G.GH_X]) - base) / cost[j]
    return out


def kg_pair(m, C, A, n, s2, cand_s):
    base = G.expU(m, C, A, n).max()
    out = np.full(len(cand_s), -np.inf)
    for i, s in enumerate(cand_s):
        o = [s, n + s]
        P = C[np.ix_(o, o)] + np.diag(s2[o])
        L = np.linalg.cholesky(P)
        Kg = C[:, o] @ np.linalg.inv(P)
        Cn = C - Kg @ C[o, :]
        vals = [G.expU(m + Kg @ (L @ z), Cn, A, n).max() for z in GH]
        out[i] = np.dot(GHW, vals) - base
    return out


def p_near(m, C, A, n, choice, rng, ns=400):
    z = rng.multivariate_normal(m, C + 1e-10 * np.eye(len(m)), ns, method="cholesky")
    U = np.clip(A * (np.exp(z[:, n:]) - np.exp(z[:, :n])), 0, None)
    return float((U[:, choice] >= 0.95 * U.max(1)).mean())


def simulate(gd, params_k, params_n, PHI, rng):
    n = len(gd)
    phi = PHI(gd.solvent.tolist())
    lr = lambda x: np.log(x) - np.log1p(-np.minimum(x, 1 - 1e-12))  # noqa: E731
    lnr_hat = np.r_[lr(np.exp(gd.lnx_cold_pred.values)), lr(np.exp(gd.lnx_hot_pred.values))]
    truth = np.r_[lr(gd.x_cold.values), lr(gd.x_hot.values)]
    A = 1000 * gd.rho_hot.values * gd.solute_mw.values / gd.solvent_mw.values
    U = np.clip(A * (np.exp(truth[n:]) - np.exp(truth[:n])), 0, None)
    Ust = U.max()
    reg = lambda s: (Ust - U[s]) / Ust if Ust > 0 else 0.0  # noqa: E731
    pred_U = np.clip(A * (np.exp(lnr_hat[n:]) - np.exp(lnr_hat[:n])), 0, None)
    rows = []
    for rname, (h, hmult, pair) in REGIMES.items():
        for sig in NOISES:
            obs_sd = np.r_[np.full(n, sig), np.full(n, sig * hmult)]
            y_obs = truth + rng.normal(0, 1, 2 * n) * obs_sd  # one noisy realization shared by all policies
            if pair:
                cost_s = np.ones(n)
                full = float(n)
            else:
                cost = np.r_[np.ones(n), np.full(n, h)]
                full = cost.sum()
            pols = ["KG_kernel", "KG_null", "PAIR", "RANDOM", "FULL"] + ([] if pair else ["CF_classic", "CF_bayes"])
            for pol in pols:
                mdl = params_n if pol == "KG_null" else params_k
                m, C, se2 = G.prior(mdl, phi, lnr_hat)
                s2 = se2 + obs_sd ** 2
                meas = np.zeros(2 * n, bool)
                spent, curve, stop = 0.0, [], None
                bayes = pol not in ("CF_classic", "PAIR")

                def decide():
                    if bayes:
                        return int(np.argmax(G.expU(m, C, A, n)))
                    done = [s for s in range(n) if meas[s] and meas[n + s]]
                    if done:
                        return max(done, key=lambda s: A[s] * (np.exp(y_obs[n + s]) - np.exp(y_obs[s])))
                    z = np.where(meas, y_obs, lnr_hat)
                    return int(np.argmax(A * (np.exp(z[n:]) - np.exp(z[:n]))))

                ch = decide()
                curve.append((0.0, reg(ch)))
                if pol.startswith("KG") and p_near(m, C, A, n, ch, rng) >= 0.9:
                    stop = (0.0, reg(ch))
                order_rand = list(rng.permutation(n if pair else 2 * n))
                order_pair = list(np.argsort(-pred_U))
                rule_end = None
                if pol == "PAIR":
                    rule_end = (n + 2) // 2 if not pair else int(np.ceil(n / 2))
                steps = n if pair else 2 * n
                for step in range(steps):
                    if pair:
                        cand = [s for s in range(n) if not meas[s]]
                        if pol.startswith("KG"):
                            s = cand[int(np.argmax(kg_pair(m, C, A, n, s2, cand)))]
                        elif pol == "RANDOM":
                            s = [x for x in order_rand if not meas[x]][0]
                        else:  # PAIR / FULL: predicted-U order
                            s = [x for x in order_pair if not meas[x]][0]
                        o = [s, n + s]
                        meas[o] = True
                        spent += 1.0
                        m, C = upd_multi(m, C, o, y_obs[o], s2[o])
                    else:
                        cand = np.where(~meas)[0]
                        if pol.startswith("KG"):
                            sc = kg_iso(m, C, A, n, s2, cand, cost)
                            if not np.isfinite(sc).any() or sc.max() <= 1e-12:
                                sc = np.diag(C)[cand] / cost[cand]
                            j = int(cand[np.argmax(sc)])
                        elif pol in ("CF_classic", "CF_bayes"):
                            cl = [s for s in np.argsort(-pred_U) if not meas[s]]
                            if cl:
                                j = int(cl[0])
                            else:
                                capc = A * np.exp(y_obs[:n])
                                j = int(n + [s for s in np.argsort(-capc) if not meas[n + s]][0])
                        elif pol == "RANDOM":
                            j = int([x for x in order_rand if not meas[x]][0])
                        else:  # PAIR / FULL: predicted-U order, cold then hot
                            seq = [x for s in order_pair for x in (s, n + s)]
                            j = int([x for x in seq if not meas[x]][0])
                        meas[j] = True
                        spent += cost[j]
                        m, C = G.update(m, C, j, y_obs[j], s2[j])
                    ch = decide()
                    curve.append((spent / full, reg(ch)))
                    if pol.startswith("KG") and stop is None and p_near(m, C, A, n, ch, rng) >= 0.9:
                        stop = (spent / full, reg(ch))
                    if pol == "PAIR" and rule_end is not None:
                        done_pairs = sum(meas[s] and meas[n + s] for s in range(n))
                        if done_pairs == rule_end and stop is None:
                            stop = (spent / full, reg(ch))
                    if pol == "CF_classic" and stop is None and meas[:n].all() and meas[n:].sum() == min(2, n):
                        stop = (spent / full, reg(ch))
                if stop is None:
                    stop = curve[-1]
                s = G.success_on_grid(curve)
                rows.append({"regime": rname, "noise": sig, "policy": pol, "stop_cost": stop[0],
                             "stop_regret": stop[1], "stop_success": float(stop[1] <= 0.05),
                             **{f"s{int(round(x * 100)):03d}": v for x, v in zip(G.GRID, s)}})
    return rows


def job(bb, k):
    cur = pd.read_csv(OUT / f"oof_{bb}.csv")
    grp = pd.read_csv(OUT / "groups.csv", index_col=0)
    cur = cur[cur.group.isin(grp.index[grp.U_complete])]
    solvents = sorted(pd.read_csv(OUT / f"oof_{bb}.csv").solvent.unique())
    Aphi = G.solvent_phi(solvents)
    d = {s: Aphi[i] for i, s in enumerate(solvents)}
    PHI = lambda ss: np.stack([d[s] for s in ss])  # noqa: E731
    mw = {s: Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in set(cur.solute) | set(cur.solvent)}
    cur = cur.assign(solute_mw=cur.solute.map(mw), solvent_mw=cur.solvent.map(mw))
    pj = json.load(open(OUT / f"g1_part_{bb}_{k}.json"))
    keys = ["sb", "sd", "su", "sv", "se", "au", "av", "ell", "mb", "md"]
    pk, pn = [pj["kernel"][x] for x in keys], [pj["null"][x] for x in keys]
    rng = np.random.default_rng(9000 + k)
    out = []
    for g, gd in cur[cur.fold == k].groupby("group"):
        for r in simulate(gd.reset_index(drop=True), pk, pn, PHI, rng):
            out.append({"backbone": bb, "fold": k, "group": g, "solute": gd.solute.iloc[0], **r})
    pd.DataFrame(out).to_csv(OUT / f"g2b_part_{bb}_{k}.csv.gz", index=False, compression="gzip")
    print(bb, k, "done", flush=True)


def aggregate():
    rng = np.random.default_rng(20261111)
    scol = [f"s{int(round(x * 100)):03d}" for x in G.GRID]
    res = []
    for bb in ["mlp", "xgb"]:
        D = pd.concat([pd.read_csv(OUT / f"g2b_part_{bb}_{k}.csv.gz") for k in range(5)])
        for (rg, nz), d in D.groupby(["regime", "noise"]):
            sol = d.groupby(["policy", "solute"])[scol + ["stop_cost", "stop_success", "stop_regret"]].mean()
            pols = d.policy.unique()
            sols = sol.loc["KG_kernel"].index.values
            for p in pols:
                x = sol.loc[p].loc[sols]
                curve = x[scol].values.mean(0)
                bs = rng.integers(0, len(sols), (2000, len(sols)))
                auc_b = x[scol].values.mean(1)[bs].mean(1)
                res.append({"backbone": bb, "regime": rg, "noise": nz, "policy": p, "solutes": len(sols),
                            "auc": round(float(curve.mean()), 4),
                            "auc_ci95": [round(float(np.quantile(auc_b, 0.025)), 4), round(float(np.quantile(auc_b, 0.975)), 4)],
                            "f90": float(G.f90(curve)) if np.isfinite(G.f90(curve)) else None,
                            "stop_success": round(float(x.stop_success.mean()), 3),
                            "stop_cost": round(float(x.stop_cost.mean()), 3),
                            "stop_mean_regret": round(float(x.stop_regret.mean()), 4)})
    R = pd.DataFrame(res)
    R.to_csv(OUT / "g2b_summary.csv", index=False)
    print(R.drop(columns=["auc_ci95"]).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "fold":
        job(sys.argv[2], int(sys.argv[3]))
    else:
        aggregate()
