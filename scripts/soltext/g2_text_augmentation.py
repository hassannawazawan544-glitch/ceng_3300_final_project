"""SOLTEXT-G2: does adding SolText constraints improve held-out solubility prediction?

PROTOCOL (fixed before any arm was scored; written 2026-11-11)
  Data      BigSolDB 2.0 curated cohort (results/advanced_v1/cohort.csv) with its existing
            solute-disjoint split: train 790 / validation 170 / internal test 170 solutes.
            External development: results/publication_v1/external_development/cohort.csv
            (BigSolDB v2.2 new solutes + new papers, rows with valid_target & eligible_structure).
            Reserved calibration/final cohorts are NOT touched.
  Leakage   Text constraints whose compound (InChIKey connectivity block) is a validation, test or
            external solute are removed. Train-solute overlap is allowed.
  Model     One backbone: MLP on RDKit 2D descriptors of solute and solvent (217 each, robust-scaled,
            clipped), van't Hoff head ln x = -softplus(-(a - b*(1000/T - 1000/298.15))).
            Adam, lr 1e-3, batch 512, <= 60 epochs, early stopping on validation solute-macro MAE.
  Arms      A0  BigSolDB only
            A1  + quantitative bounds: upper (suspended, rt), lower (name-linked dissolved, rt/cold),
                  hot lower (recrystallization mass/volume at solvent bp)
            A2  A1 + within-compound orderings: ln x(S_dissolved) > ln x(S_washed/triturated/
                  precipitated/insoluble) at 298 K; recrystallization: ln x(bp) - ln x(298) >= ln 2
            A3  A2 + absolute "low" for washed/triturated/precipitated/insoluble: ln x(298) <= ln 0.02
            Constraint loss = smooth hinge (softplus, scale 0.5) on the violation, weight lambda.
            lambda in {0.03, 0.1, 0.3, 1.0} chosen per arm by validation solute-macro MAE (seed 0),
            then 3 seeds at the chosen lambda. A0 has 3 seeds.
  Endpoints Primary: solute-macro MAE in log10 x on (i) internal test and (ii) external development,
            mean over 3 seeds, paired solute bootstrap (4,000) of arm - A0.
            Secondary: "far" test solutes (max Morgan-2 Tanimoto to train solutes < 0.4);
            within-solute Spearman of solvents at fixed temperature (sources >= 3 solvents).
  Pass (G2) a text arm improves BOTH primary endpoints with 95% CI excluding zero.
            This is a single-backbone screening test ("G2-lite"); a pass motivates the full study,
            a fail is reported as such.

Usage (lab root):  .venv-graph/bin/python -I scripts/soltext/g2_text_augmentation.py prep
                   .venv-graph/bin/python -I scripts/soltext/g2_text_augmentation.py run
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from torch import nn

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "soltext_g2"
OUT.mkdir(parents=True, exist_ok=True)
T0 = 1000 / 298.15
LN10 = np.log(10)
DESC = [n for n, _ in Descriptors.descList]


def desc(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return np.full(len(DESC), np.nan, np.float32)
    out = []
    for _, f in Descriptors.descList:
        try:
            out.append(f(m))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, np.float64).clip(-1e8, 1e8).astype(np.float32)


def ikey14(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    try:
        return Chem.MolToInchiKey(m)[:14] if m is not None else None
    except Exception:
        return None


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m is not None else None


# ====================================================================== prep
def prep():
    t = time.time()
    coh = pd.read_csv(ROOT / "results/advanced_v1/cohort.csv")
    coh = coh[["solute", "solvent", "Temperature_K", "Solubility(mole_fraction)", "split", "Source"]].rename(
        columns={"Temperature_K": "T", "Solubility(mole_fraction)": "x", "Source": "source"})
    ext = pd.read_csv(ROOT / "results/publication_v1/external_development/cohort.csv")
    ext = ext[ext.valid_target.astype(bool) & ext.eligible_structure.astype(bool)]
    ext = ext[["solute", "solvent", "Temperature_K", "x", "doi"]].rename(columns={"Temperature_K": "T", "doi": "source"})
    ext["split"] = "external"
    db = pd.concat([coh, ext], ignore_index=True)
    db["solute"] = db.solute.map(canon)
    db["solvent"] = db.solvent.map(canon)
    db = db.dropna(subset=["solute", "solvent"])
    db["y"] = np.log(db.x)

    held = set(db.loc[db.split.isin(["validation", "test", "external"]), "solute"])
    held_keys = {ikey14(s) for s in held}

    core = pd.read_csv(ROOT / "results/soltext_v1/soltext_v1_core.csv.gz", low_memory=False)
    core = core[~core.solvents.isin(["PETROL", "XYLENE"])].copy()
    core["solvent"] = core.solvents.map(canon)
    core["solute"] = core.solute_canon
    core = core.dropna(subset=["solute", "solvent"])
    n_before = core.inchikey.nunique()
    core = core[~core.ikey14.isin(held_keys)]
    leak_removed = n_before - core.inchikey.nunique()

    cons = []  # (kind, solute, solvent, T, value, solute2/solvent2, T2)
    low_types = {"washed", "triturated", "precipitated_by", "insoluble"}
    T_ctx = {"rt": 298.15, "cold": 278.15}
    for r in core.itertuples():
        if r.bound_kind == "upper" and r.T_context == "rt":
            cons.append(("upper", r.solute, r.solvent, 298.15, r.lnx_bound, None, None))
        elif r.bound_kind == "lower" and r.link == "name" and r.T_context in T_ctx:
            cons.append(("lower", r.solute, r.solvent, T_ctx[r.T_context], r.lnx_bound, None, None))
        elif r.bound_kind == "lower_hot" and np.isfinite(r.T_hot_K):
            cons.append(("lower", r.solute, r.solvent, r.T_hot_K, r.lnx_bound, None, None))
        if r.type == "recrystallized" and np.isfinite(r.T_hot_K) and r.T_hot_K > 310:
            cons.append(("tcontrast", r.solute, r.solvent, r.T_hot_K, np.log(2.0), r.solvent, 298.15))
        if r.type in low_types and r.T_context in T_ctx:
            cons.append(("abs_low", r.solute, r.solvent, 298.15, np.log(0.02), None, None))
    hi = core[(core.type == "dissolved") & (core.link == "name") & core.T_context.isin(["rt", "cold"])]
    lo = core[core.type.isin(low_types) & core.T_context.isin(["rt", "cold"])]
    pairs = hi[["solute", "solvent"]].drop_duplicates().merge(
        lo[["solute", "solvent"]].drop_duplicates(), on="solute", suffixes=("_hi", "_lo"))
    pairs = pairs[pairs.solvent_hi != pairs.solvent_lo]
    for r in pairs.itertuples():
        cons.append(("order", r.solute, r.solvent_hi, 298.15, 0.0, r.solvent_lo, 298.15))
    cons = pd.DataFrame(cons, columns=["kind", "solute", "solvent", "T", "value", "solvent2", "T2"])
    cons = cons.drop_duplicates(["kind", "solute", "solvent", "T", "solvent2"])

    mols = sorted(set(db.solute) | set(db.solvent) | set(cons.solute) | set(cons.solvent) | set(cons.solvent2.dropna()))
    with Pool(14) as p:
        X = np.stack(p.map(desc, mols, chunksize=200))
    np.savez_compressed(OUT / "features.npz", X=X, mols=np.array(mols, dtype=object))

    # far test solutes: max Morgan-2 Tanimoto to train solutes < 0.4
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    tr_fps = [fpg.GetFingerprint(Chem.MolFromSmiles(s)) for s in db.loc[db.split == "train", "solute"].unique()]
    far = {}
    for s in db.loc[db.split.isin(["test", "external"]), "solute"].unique():
        far[s] = max(DataStructs.BulkTanimotoSimilarity(fpg.GetFingerprint(Chem.MolFromSmiles(s)), tr_fps))
    db["maxsim_train"] = db.solute.map(far)
    db.to_csv(OUT / "bigsoldb_rows.csv.gz", index=False, compression="gzip")
    cons.to_csv(OUT / "constraints.csv.gz", index=False, compression="gzip")
    info = {"bigsoldb_rows_by_split": db.split.value_counts().to_dict(),
            "solutes_by_split": db.groupby("split").solute.nunique().to_dict(),
            "text_compounds_after_leak_filter": int(core.inchikey.nunique()),
            "text_compounds_removed_for_leakage": int(leak_removed),
            "constraints_by_kind": cons.kind.value_counts().to_dict(),
            "constraint_compounds": int(cons.solute.nunique()),
            "far_test_solutes_lt_0.4": int(sum(v < 0.4 for k, v in far.items()
                                              if k in set(db.loc[db.split == 'test', 'solute']))),
            "far_external_solutes_lt_0.4": int(sum(v < 0.4 for k, v in far.items()
                                                  if k in set(db.loc[db.split == 'external', 'solute']))),
            "molecules_featurized": len(mols), "seconds": round(time.time() - t)}
    json.dump(info, open(OUT / "prep_summary.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


# ====================================================================== model
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * d, 512), nn.SiLU(), nn.Dropout(0.1),
                               nn.Linear(512, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 2))

    def forward(self, xs, xv, T):
        h = self.f(torch.cat([xs, xv], 1))
        a, b = h[:, 0], nn.functional.softplus(h[:, 1])  # b >= 0: solubility rises with T (endothermic)
        z = a - b * (1000.0 / T - T0)
        return -nn.functional.softplus(-z)  # ln x <= 0


def shinge(v, s=0.5):
    return (s * nn.functional.softplus(v / s)).mean()


def load():
    f = np.load(OUT / "features.npz", allow_pickle=True)
    X, mols = f["X"].astype(np.float64), list(f["mols"])
    db = pd.read_csv(OUT / "bigsoldb_rows.csv.gz")
    cons = pd.read_csv(OUT / "constraints.csv.gz")
    idx = {m: i for i, m in enumerate(mols)}
    tr_mols = sorted(set(db.loc[db.split == "train", "solute"]) | set(db.loc[db.split == "train", "solvent"]))
    tr = X[[idx[m] for m in tr_mols]]
    med = np.nanmedian(tr, 0)
    q1, q3 = np.nanpercentile(tr, 25, 0), np.nanpercentile(tr, 75, 0)
    iqr = np.where(q3 - q1 > 1e-9, q3 - q1, np.nanstd(tr, 0) + 1e-9)
    Z = np.nan_to_num((X - med) / iqr, nan=0.0).clip(-10, 10)
    keep = np.nanstd(tr, 0) > 1e-9
    return torch.tensor(Z[:, keep], dtype=torch.float32), idx, db, cons


def tensors(df, idx, dev, cols=("solute", "solvent", "T")):
    s = torch.tensor(df[cols[0]].map(idx).values, device=dev)
    v = torch.tensor(df[cols[1]].map(idx).values, device=dev)
    T = torch.tensor(df[cols[2]].values, dtype=torch.float32, device=dev)
    return s, v, T


def macro_mae(df, pred):
    e = np.abs(pred - df.y.values) / LN10
    return float(pd.Series(e).groupby(df.solute.values).mean().mean())


def train(arm, lam, seed, Z, idx, db, cons, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Zd = Z.to(dev)
    net = Net(Z.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    tr = db[db.split == "train"]
    va = db[db.split == "validation"]
    ts, tv, tT = tensors(tr, idx, dev)
    ty = torch.tensor(tr.y.values, dtype=torch.float32, device=dev)
    kinds = {"A0": [], "A1": ["upper", "lower"], "A2": ["upper", "lower", "order", "tcontrast"],
             "A3": ["upper", "lower", "order", "tcontrast", "abs_low"]}[arm]
    C = {k: cons[cons.kind == k] for k in kinds}
    Ct = {}
    for k, c in C.items():
        s, v, T = tensors(c, idx, dev)
        val = torch.tensor(c.value.values, dtype=torch.float32, device=dev)
        if k in ("order", "tcontrast"):
            v2 = torch.tensor(c.solvent2.map(idx).values, device=dev)
            T2 = torch.tensor(c.T2.values, dtype=torch.float32, device=dev)
            Ct[k] = (s, v, T, val, v2, T2)
        else:
            Ct[k] = (s, v, T, val)
    best, best_state, bad = 1e9, None, 0
    n = len(tr)
    steps = max(1, n // 512)
    for ep in range(60):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(steps):
            b = perm[i * 512:(i + 1) * 512]
            pred = net(Zd[ts[b]], Zd[tv[b]], tT[b])
            loss = ((pred - ty[b]) ** 2).mean()
            for k, tpl in Ct.items():
                m = torch.randint(0, len(tpl[0]), (256,), device=dev)
                s, v, T, val = tpl[0][m], tpl[1][m], tpl[2][m], tpl[3][m]
                p1 = net(Zd[s], Zd[v], T)
                if k == "upper" or k == "abs_low":
                    loss = loss + lam * shinge(p1 - val)
                elif k == "lower":
                    loss = loss + lam * shinge(val - p1)
                elif k == "order":
                    p2 = net(Zd[s], Zd[tpl[4][m]], tpl[5][m])
                    loss = loss + lam * shinge(p2 - p1)
                elif k == "tcontrast":
                    p2 = net(Zd[s], Zd[tpl[4][m]], tpl[5][m])
                    loss = loss + lam * shinge(val - (p1 - p2))
            opt.zero_grad()
            loss.backward()
            opt.step()
        vm = macro_mae(va, predict(net, va, Zd, idx, dev))
        if vm < best - 1e-4:
            best, bad = vm, 0
            best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best_state)
    return net, best


@torch.no_grad()
def predict(net, df, Zd, idx, dev):
    net.eval()
    s, v, T = tensors(df, idx, dev)
    out = [net(Zd[s[i:i + 8192]], Zd[v[i:i + 8192]], T[i:i + 8192]).cpu().numpy() for i in range(0, len(df), 8192)]
    return np.concatenate(out)


def spearman_within(df, pred):
    """Mean Spearman of solvent ordering within (solute, source, T) groups with >= 3 solvents."""
    d = df.assign(p=pred).copy()
    d["Tr"] = d["T"].round(0)
    rs = []
    for _, g in d.groupby(["solute", "source", "Tr"]):
        if g.solvent.nunique() >= 3:
            g = g.groupby("solvent")[["y", "p"]].mean()
            r = g.y.rank().corr(g.p.rank())
            if np.isfinite(r):
                rs.append((_[0], r))
    if not rs:
        return np.nan
    return float(pd.DataFrame(rs, columns=["s", "r"]).groupby("s").r.mean().mean())


def run():
    dev = "cuda"
    Z, idx, db, cons = load()
    lams = [0.03, 0.1, 0.3, 1.0]
    log = []
    choice = {"A0": 0.0}
    for arm in ["A1", "A2", "A3"]:
        scores = {}
        for lam in lams:
            _, vm = train(arm, lam, 0, Z, idx, db, cons, dev)
            scores[lam] = vm
            log.append({"stage": "select", "arm": arm, "lambda": lam, "val_macro_mae": vm})
            print("select", arm, lam, round(vm, 4), flush=True)
        choice[arm] = min(scores, key=scores.get)
    preds = {}
    for arm, lam in choice.items():
        for seed in range(3):
            net, vm = train(arm, lam, seed, Z, idx, db, cons, dev)
            for split in ["validation", "test", "external"]:
                d = db[db.split == split]
                preds[(arm, seed, split)] = predict(net, d, Z.to(dev), idx, dev)
            log.append({"stage": "final", "arm": arm, "lambda": lam, "seed": seed, "val_macro_mae": vm})
            print("final", arm, lam, seed, round(vm, 4), flush=True)
    rng = np.random.default_rng(20261111)
    res = {"lambda_choice": choice, "arms": {}}
    rows = []
    for split in ["validation", "test", "external"]:
        d = db[db.split == split].reset_index(drop=True)
        far_mask = (d.maxsim_train < 0.4).values if split != "validation" else np.zeros(len(d), bool)
        err = {arm: np.mean([np.abs(preds[(arm, s, split)] - d.y.values) / LN10 for s in range(3)], 0) for arm in choice}
        pm = {arm: np.mean([preds[(arm, s, split)] for s in range(3)], 0) for arm in choice}
        per_sol = {arm: pd.Series(e).groupby(d.solute.values).mean() for arm, e in err.items()}
        sols = per_sol["A0"].index.values
        for arm in choice:
            entry = {"macro_mae": round(float(per_sol[arm].mean()), 4),
                     "row_mae": round(float(err[arm].mean()), 4),
                     "spearman_within": round(spearman_within(d, pm[arm]), 4)}
            if far_mask.any():
                fs = pd.Series(err[arm][far_mask]).groupby(d.solute.values[far_mask]).mean()
                entry["far_macro_mae"] = round(float(fs.mean()), 4)
                entry["far_solutes"] = int(len(fs))
            if arm != "A0":
                diff = (per_sol[arm] - per_sol["A0"]).values
                bs = rng.choice(diff, (4000, len(diff))).mean(1)
                entry["delta_macro_vs_A0"] = round(float(diff.mean()), 4)
                entry["delta_ci95"] = [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)]
                if far_mask.any():
                    fd = (pd.Series(err[arm][far_mask]).groupby(d.solute.values[far_mask]).mean()
                          - pd.Series(err["A0"][far_mask]).groupby(d.solute.values[far_mask]).mean()).values
                    fb = rng.choice(fd, (4000, len(fd))).mean(1)
                    entry["far_delta"] = round(float(fd.mean()), 4)
                    entry["far_delta_ci95"] = [round(float(np.quantile(fb, 0.025)), 4), round(float(np.quantile(fb, 0.975)), 4)]
            res["arms"].setdefault(arm, {})[split] = entry
            rows.append({"split": split, "arm": arm, **{k: v for k, v in entry.items() if not isinstance(v, list)}})
        pd.DataFrame({"solute": d.solute, "solvent": d.solvent, "T": d["T"], "y": d.y,
                      **{f"pred_{a}": pm[a] for a in choice}}).to_csv(OUT / f"predictions_{split}.csv.gz",
                                                                     index=False, compression="gzip")
    passed = [a for a in ["A1", "A2", "A3"]
              if res["arms"][a]["test"]["delta_ci95"][1] < 0 and res["arms"][a]["external"]["delta_ci95"][1] < 0]
    res["G2_pass_arms"] = passed
    res["G2_pass"] = bool(passed)
    pd.DataFrame(log).to_csv(OUT / "training_log.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "results_table.csv", index=False)
    json.dump(res, open(OUT / "g2_results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"prep": prep, "run": run}[sys.argv[1]]()
