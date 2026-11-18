"""DTMD-COHORT: curve-fit decision cohort from the BigSolDB curated cohort (protocol V1_PROTOCOL.md).

Usage (lab root): .venv-graph/bin/python -I scripts/v1/v1_build_cohort.py
Outputs results/v1_dtmd/: curves.csv (one row per accepted curve with both endpoints), groups.csv,
cohort_summary.json, aleatoric_limit.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v1_dtmd"
OUT.mkdir(parents=True, exist_ok=True)
TC, TH, TOL, RMSE_MAX = 298.15, 323.15, 0.5, 0.15


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m is not None else None


def fold_of(smi):
    return int(hashlib.sha256(smi.encode()).hexdigest(), 16) % 5


coh = pd.read_csv(ROOT / "results/advanced_v1/cohort.csv")
coh["solute_c"] = coh.solute.map({s: canon(s) for s in coh.solute.unique()})
coh["solvent_c"] = coh.solvent.map({s: canon(s) for s in coh.solvent.unique()})
coh = coh.dropna(subset=["solute_c", "solvent_c"])
coh["lnx"] = np.log(coh["Solubility(mole_fraction)"])
coh["u"] = 1000.0 / coh.Temperature_K

# density per solvent: linear rho(T) from cohort rows, used only within +-15 K of its data
rho_fit = {}
for s, g in coh.dropna(subset=["rho_g_cm3"]).groupby("solvent_c"):
    if g.Temperature_K.nunique() >= 2:
        b = np.polyfit(g.Temperature_K, g.rho_g_cm3, 1)
        rho_fit[s] = (b, g.Temperature_K.min(), g.Temperature_K.max())


def rho_at(s, T):
    f = rho_fit.get(s)
    if f is None or T < f[1] - 15 or T > f[2] + 15:
        return np.nan
    return float(np.polyval(f[0], T))


rows, n_curves, rej = [], 0, {"range": 0, "rmse": 0, "points": 0}
for (sol, src, sv), g in coh.groupby(["solute_c", "Source", "solvent_c"]):
    n_curves += 1
    g = g.groupby("Temperature_K", as_index=False).lnx.median()
    if len(g) < 2:
        rej["points"] += 1
        continue
    if g.Temperature_K.min() > TC + TOL or g.Temperature_K.max() < TH - TOL:
        rej["range"] += 1
        continue
    u, y = 1000 / g.Temperature_K.values, g.lnx.values
    b1 = np.polyfit(u, y, 1)
    res1 = y - np.polyval(b1, u)
    best, deg = b1, 1
    if len(g) >= 5:
        b2 = np.polyfit(u, y, 2)
        res2 = y - np.polyval(b2, u)
        n = len(g)
        bic1 = n * np.log(np.mean(res1 ** 2) + 1e-12) + 2 * np.log(n)
        bic2 = n * np.log(np.mean(res2 ** 2) + 1e-12) + 3 * np.log(n)
        if bic2 < bic1 - 2:
            best, deg = b2, 2
    rmse = float(np.sqrt(np.mean((y - np.polyval(best, u)) ** 2)))
    if len(g) > deg + 1 and rmse > RMSE_MAX:
        rej["rmse"] += 1
        continue
    xc, xh = np.exp(np.polyval(best, 1000 / TC)), np.exp(np.polyval(best, 1000 / TH))
    if not (0 < xc < 1 and 0 < xh < 1):
        rej["range"] += 1
        continue
    rows.append({"solute": sol, "source": src, "solvent": sv, "n_points": len(g), "deg": deg, "fit_rmse": rmse,
                 "x_cold": xc, "x_hot": xh, "rho_hot": rho_at(sv, TH)})
cur = pd.DataFrame(rows)
ns = cur.groupby(["solute", "source"]).solvent.nunique()
keep = ns[ns >= 3].index
cur = cur.set_index(["solute", "source"]).loc[keep].reset_index()
cur["group"] = cur.solute + " || " + cur.source
cur["fold"] = cur.solute.map(fold_of)
cur["r_cold"], cur["r_hot"] = cur.x_cold / (1 - cur.x_cold), cur.x_hot / (1 - cur.x_hot)
cur["R"] = np.clip(1 - cur.r_cold / cur.r_hot, 0, None)
mw = {s: Chem.Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in set(cur.solute) | set(cur.solvent)} if False else None
from rdkit.Chem import Descriptors  # noqa: E402
mw = {s: Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in set(cur.solute) | set(cur.solvent)}
cur["U"] = 1000 * cur.rho_hot * cur.solute.map(mw) / cur.solvent.map(mw) * np.clip(cur.r_hot - cur.r_cold, 0, None)
cur.to_csv(OUT / "curves.csv", index=False)
grp = cur.groupby("group").agg(solute=("solute", "first"), source=("source", "first"), fold=("fold", "first"),
                               n_solvents=("solvent", "nunique"), U_complete=("U", lambda s: s.notna().all()))
grp.to_csv(OUT / "groups.csv")

summ = {"curves_considered": n_curves, "rejected": rej, "accepted_curves_in_groups": int(len(cur)),
        "groups": int(len(grp)), "solutes": int(grp.solute.nunique()), "sources": int(grp.source.nunique()),
        "mean_solvents_per_group": round(float(grp.n_solvents.mean()), 2),
        "groups_with_complete_hot_density": int(grp.U_complete.sum()),
        "groups_per_fold": grp.fold.value_counts().sort_index().to_dict(),
        "quadratic_curves": int((cur.deg == 2).sum()), "distinct_solvents": int(cur.solvent.nunique())}

# ------------------------------------------------------------- decision aleatoric limit
agree = []
for sol, g in cur.groupby("solute"):
    srcs = g.source.unique()
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            a, b = g[g.source == srcs[i]].set_index("solvent"), g[g.source == srcs[j]].set_index("solvent")
            common = a.index.intersection(b.index)
            if len(common) < 3:
                continue
            a, b = a.loc[common], b.loc[common]
            rec = {"solute": sol, "n_common": len(common),
                   "agree_best_R": bool(a.R.idxmax() == b.R.idxmax()),
                   "agree_best_cold_x": bool(a.x_cold.idxmax() == b.x_cold.idxmax()),
                   "spearman_lnx_cold": float(np.log(a.x_cold).rank().corr(np.log(b.x_cold).rank())),
                   "mad_lnx_cold": float(np.median(np.abs(np.log(a.x_cold) - np.log(b.x_cold)))),
                   "mad_lnx_cold_centered": float(np.median(np.abs((np.log(a.x_cold) - np.log(a.x_cold).mean())
                                                                   - (np.log(b.x_cold) - np.log(b.x_cold).mean()))))}
            if a.U.notna().all() and b.U.notna().all():
                rec["agree_best_U"] = bool(a.U.idxmax() == b.U.idxmax())
                rel = (a.U.max() - a.U.loc[b.U.idxmax()]) / a.U.max()
                rec["regret_rel_U_using_other_lab"] = float(rel)
            rec["chance_best"] = 1 / len(common)
            agree.append(rec)
ag = pd.DataFrame(agree)
ag.to_csv(OUT / "aleatoric_pairs.csv", index=False)
per_sol = ag.groupby("solute").mean(numeric_only=True)
lim = {"source_pairs": int(len(ag)), "solutes": int(ag.solute.nunique()),
       "solute_macro_agree_best_R": round(float(per_sol.agree_best_R.mean()), 3),
       "solute_macro_agree_best_cold_x": round(float(per_sol.agree_best_cold_x.mean()), 3),
       "solute_macro_chance": round(float(per_sol.chance_best.mean()), 3),
       "median_spearman_lnx_cold": round(float(ag.spearman_lnx_cold.median()), 3),
       "median_abs_lnx_diff_cold": round(float(ag.mad_lnx_cold.median()), 3),
       "median_abs_lnx_diff_cold_after_removing_shared_level": round(float(ag.mad_lnx_cold_centered.median()), 3)}
if "agree_best_U" in ag:
    u = ag.dropna(subset=["agree_best_U"])
    pu = u.groupby("solute").mean(numeric_only=True)
    lim.update({"U_pairs": int(len(u)), "U_solutes": int(len(pu)),
                "solute_macro_agree_best_U": round(float(pu.agree_best_U.mean()), 3),
                "median_rel_regret_U_using_other_lab": round(float(u.regret_rel_U_using_other_lab.median()), 4),
                "mean_rel_regret_U_using_other_lab": round(float(u.regret_rel_U_using_other_lab.mean()), 4)})
json.dump(summ, open(OUT / "cohort_summary.json", "w"), indent=1)
json.dump(lim, open(OUT / "aleatoric_limit.json", "w"), indent=1)
print(json.dumps({"cohort": summ, "aleatoric_limit": lim}, indent=1))
