"""DTMD amendment A1.4: frozen final confirmation test (run ONCE, after freeze_manifest.json exists).

Usage (lab root):
  .venv-graph/bin/python -I scripts/v1/v1_final_test.py prep        # reserved outcomes read here
  .venv-advanced/bin/python -I scripts/v1/v1_final_test.py xgb_predict
  .venv-graph/bin/python -I scripts/v1/v1_final_test.py evaluate
Cohort: BigSolDB v2.2 calibration + final_candidate partitions (external_candidate_annotations.csv),
valid target & neutral single-fragment structures (as PUB-003), development solutes excluded
(InChIKey connectivity block), curves/groups exactly as v1_build_cohort.py, hot density from the frozen
development density fits. No refitting of any kind on this cohort.
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
FT = OUT / "final_test"
FT.mkdir(exist_ok=True)
TC, TH, TOL, RMSE_MAX = 298.15, 323.15, 0.5, 0.15


def prep():
    import csv
    import torch
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    sys.path.insert(0, str(ROOT / "scripts" / "soltext"))
    from g2_text_augmentation import Net, desc

    assert (OUT / "freeze_manifest.json").exists(), "freeze first"
    base = ROOT / "results/publication_v1"
    a = pd.read_csv(base / "external_candidate_annotations.csv", keep_default_na=False)
    a = a[a.partition.isin(["calibration", "final_candidate"])].copy()
    ids = set(a.v22_source_row.astype(int))
    xs = []
    with (base / "metadata/BigSolDBv2.2.csv").open() as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i in ids:
                xs.append({"v22_source_row": i, "x": float(r["Solubility(mole_fraction)"])})
    a = a.merge(pd.DataFrame(xs), on="v22_source_row", validate="one_to_one")
    a["valid_target"] = np.isfinite(a.x) & a.x.gt(0) & a.x.lt(1)
    a["eligible_structure"] = a.neutral_single_solute.astype(str).eq("True") & a.neutral_single_solvent.astype(str).eq("True")
    n0 = len(a)
    a = a[a.valid_target & a.eligible_structure].copy()
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None  # noqa: E731
    a["solute"] = a.solute.map(canon)
    a["solvent"] = a.solvent.map(canon)
    a = a.dropna(subset=["solute", "solvent"])
    a["T"] = pd.to_numeric(a.Temperature_K)
    dev = pd.read_csv(ROOT / "results/advanced_v1/cohort.csv", usecols=["solute"]).solute.unique()
    k14 = lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14]  # noqa: E731
    dev_keys = {k14(s) for s in dev}
    a["ikey14"] = a.solute.map(k14)
    n_overlap = a.loc[a.ikey14.isin(dev_keys), "solute"].nunique()
    a = a[~a.ikey14.isin(dev_keys)]
    dens = json.load(open(FZ / "density_fits.json"))

    def rho(s):
        f = dens.get(s)
        if f is None or TH < f[2] - 15 or TH > f[3] + 15:
            return np.nan
        return f[0] * TH + f[1]

    recs = []
    for (sol, src, sv), g in a.groupby(["solute", "doi", "solvent"]):
        g = g.assign(lnx=np.log(g.x)).groupby("T", as_index=False).lnx.median()
        if len(g) < 2 or g["T"].min() > TC + TOL or g["T"].max() < TH - TOL:
            continue
        u, y = 1000 / g["T"].values, g.lnx.values
        b1 = np.polyfit(u, y, 1)
        best, deg = b1, 1
        if len(g) >= 5:
            b2 = np.polyfit(u, y, 2)
            n = len(g)
            bic1 = n * np.log(np.mean((y - np.polyval(b1, u)) ** 2) + 1e-12) + 2 * np.log(n)
            bic2 = n * np.log(np.mean((y - np.polyval(b2, u)) ** 2) + 1e-12) + 3 * np.log(n)
            if bic2 < bic1 - 2:
                best, deg = b2, 2
        rmse = float(np.sqrt(np.mean((y - np.polyval(best, u)) ** 2)))
        if len(g) > deg + 1 and rmse > RMSE_MAX:
            continue
        xc, xh = np.exp(np.polyval(best, 1000 / TC)), np.exp(np.polyval(best, 1000 / TH))
        if 0 < xc < 1 and 0 < xh < 1:
            recs.append({"solute": sol, "source": src, "solvent": sv, "x_cold": xc, "x_hot": xh, "rho_hot": rho(sv)})
    cur = pd.DataFrame(recs)
    ns = cur.groupby(["solute", "source"]).solvent.nunique()
    cur = cur.set_index(["solute", "source"]).loc[ns[ns >= 3].index].reset_index()
    cur["group"] = cur.solute + " || " + cur.source
    cur["fold"] = -1
    # frozen MLP predictions
    mols = sorted(set(cur.solute) | set(cur.solvent))
    Xn = np.stack([desc(m) for m in mols]).astype(np.float64)
    sc = np.load(FZ / "mlp_scaler.npz")
    Z = torch.tensor(np.nan_to_num((Xn - sc["med"]) / sc["iqr"], nan=0.0).clip(-10, 10)[:, sc["keep"]],
                     dtype=torch.float32, device="cuda")
    idx = {m: i for i, m in enumerate(mols)}
    s = torch.tensor(cur.solute.map(idx).values, device="cuda")
    v = torch.tensor(cur.solvent.map(idx).values, device="cuda")
    preds = {"c": [], "h": []}
    for seed in range(3):
        net = Net(Z.shape[1]).cuda()
        net.load_state_dict(torch.load(FZ / f"mlp_seed{seed}.pt"))
        net.eval()
        with torch.no_grad():
            for key, T in (("c", TC), ("h", TH)):
                preds[key].append(net(Z[s], Z[v], torch.full((len(cur),), T, device="cuda")).cpu().numpy())
    cur.assign(lnx_cold_pred=np.mean(preds["c"], 0), lnx_hot_pred=np.mean(preds["h"], 0)).to_csv(FT / "final_mlp.csv", index=False)
    np.savez_compressed(FT / "final_design.npz", X=Xn, mols=np.array(mols, dtype=object))
    cur.to_csv(FT / "final_curves.csv", index=False)
    summ = {"rows_after_eligibility": int(len(a)), "rows_before": int(n0), "dev_overlap_solutes_removed": int(n_overlap),
            "groups": int(cur.group.nunique()), "solutes": int(cur.solute.nunique()), "sources": int(cur.source.nunique()),
            "groups_with_complete_density": int(cur.groupby("group").rho_hot.apply(lambda r: r.notna().all()).sum()),
            "curves": int(len(cur)), "curves_sha256": hashlib.sha256((FT / "final_curves.csv").read_bytes()).hexdigest()}
    json.dump(summ, open(FT / "final_cohort_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


def xgb_predict():
    import xgboost as xgb
    cur = pd.read_csv(FT / "final_curves.csv")
    d = np.load(FT / "final_design.npz", allow_pickle=True)
    X = d["X"].astype(np.float32)
    idx = {m: i for i, m in enumerate(d["mols"])}
    m = xgb.XGBRegressor()
    m.load_model(FZ / "xgb.json")

    def F(T):
        return np.hstack([X[cur.solute.map(idx).values], X[cur.solvent.map(idx).values], np.full((len(cur), 1), 1000 / T)])
    cur.assign(lnx_cold_pred=np.minimum(m.predict(F(TC)), 0), lnx_hot_pred=np.minimum(m.predict(F(TH)), 0)).to_csv(
        FT / "final_xgb.csv", index=False)
    print("xgb predictions", len(cur))


def evaluate():
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    sys.path.insert(0, str(ROOT / "scripts" / "v1"))
    import v1_g2_policy as G
    import v1_g2b_scenarios as S

    ph = np.load(FZ / "phi_stats.npz")
    rp = json.load(open(FZ / "residual_params.json"))
    keys = ["sb", "sd", "su", "sv", "se", "au", "av", "ell", "mb", "md"]

    def PHI(ss):
        out = []
        for x in ss:
            m = Chem.MolFromSmiles(x)
            out.append([Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m),
                        rdMolDescriptors.CalcNumHBA(m), Crippen.MolMR(m), rdMolDescriptors.CalcFractionCSP3(m),
                        rdMolDescriptors.CalcNumAromaticRings(m)])
        return (np.array(out, float) - ph["mu"]) / ph["sd"]

    rng = np.random.default_rng(20261112)
    scol = [f"s{int(round(x * 100)):03d}" for x in G.GRID]
    res = {}
    for bb in ["mlp", "xgb"]:
        cur = pd.read_csv(FT / f"final_{bb}.csv")
        ok = cur.groupby("group").rho_hot.apply(lambda r: r.notna().all())
        cur = cur[cur.group.isin(ok.index[ok])]
        mw = {s: Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in set(cur.solute) | set(cur.solvent)}
        cur = cur.assign(solute_mw=cur.solute.map(mw), solvent_mw=cur.solvent.map(mw))
        pk, pn = [rp[bb]["kernel"][k] for k in keys], [rp[bb]["null"][k] for k in keys]
        sc_rows, g2_rows = [], []
        for g, gd in cur.groupby("group"):
            gd = gd.reset_index(drop=True)
            for r in S.simulate(gd, pk, pn, PHI, rng):
                sc_rows.append({"group": g, "solute": gd.solute.iloc[0], **r})
            for r in G.run_group(gd, pk, pn, PHI, G.HS, rng):
                s = G.success_on_grid(r["curve"])
                g2_rows.append({"group": g, "solute": gd.solute.iloc[0], "h": r["h"], "policy": r["policy"],
                                **{f"s{int(round(x * 100)):03d}": v for x, v in zip(G.GRID, s)}})
        SC, G2 = pd.DataFrame(sc_rows), pd.DataFrame(g2_rows)
        SC.to_csv(FT / f"final_scenarios_{bb}.csv.gz", index=False, compression="gzip")
        G2.to_csv(FT / f"final_g2_{bb}.csv.gz", index=False, compression="gzip")
        # PRIMARY: ISO-1, noise 0.04 - AUC(KG_kernel) - AUC(best heuristic), heuristic re-chosen per resample
        d = SC[(SC.regime == "ISO-1") & (SC.noise == 0.04)]
        sol = d.groupby(["policy", "solute"])[scol].mean()
        sols = sol.loc["KG_kernel"].index.values
        auc = {p: sol.loc[p].loc[sols][scol].values.mean(1) for p in d.policy.unique()}
        heur = ["CF_classic", "CF_bayes", "PAIR", "RANDOM"]
        point_best = max(heur, key=lambda p: auc[p].mean())
        diffs = []
        for _ in range(4000):
            ix = rng.integers(0, len(sols), len(sols))
            bh = max(heur, key=lambda p: auc[p][ix].mean())
            diffs.append(auc["KG_kernel"][ix].mean() - auc[bh][ix].mean())
        prim = {"solutes": int(len(sols)), "groups": int(d.group.nunique()),
                "auc": {p: round(float(v.mean()), 4) for p, v in auc.items()},
                "best_heuristic": point_best,
                "delta_auc": round(float(auc["KG_kernel"].mean() - auc[point_best].mean()), 4),
                "delta_auc_ci95": [round(float(np.quantile(diffs, 0.025)), 4), round(float(np.quantile(diffs, 0.975)), 4)]}
        prim["confirmed"] = bool(prim["delta_auc_ci95"][0] > 0)
        # secondary: f90 by h, stopping summary by regime/noise
        sec_f90 = {}
        for h in G.HS:
            dd = G2[G2.h == h].groupby(["policy", "solute"])[scol].mean()
            sec_f90[str(h)] = {p: (lambda f: None if not np.isfinite(f) else float(f))(G.f90(dd.loc[p][scol].values.mean(0)))
                               for p in G.POLICIES}
        stop = SC.groupby(["regime", "noise", "policy", "solute"])[["stop_success", "stop_cost"]].mean() \
            .groupby(["regime", "noise", "policy"]).mean().round(3).reset_index().to_dict("records")
        res[bb] = {"primary": prim, "f90_by_h": sec_f90, "stopping": stop}
    res["confirmed_both"] = bool(res["mlp"]["primary"]["confirmed"] and res["xgb"]["primary"]["confirmed"])
    json.dump(res, open(FT / "final_results.json", "w"), indent=1)
    print(json.dumps({b: (r["primary"], r["f90_by_h"]) if isinstance(r, dict) else r for b, r in res.items()}, indent=1))


if __name__ == "__main__":
    {"prep": prep, "xgb_predict": xgb_predict, "evaluate": evaluate}[sys.argv[1]]()
