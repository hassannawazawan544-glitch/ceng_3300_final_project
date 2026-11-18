"""Read-only raw-data audit; density is never reconstructed as a model input."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'density_audit_v1';OUT.mkdir(parents=True,exist_ok=True)
s=pd.read_csv(ROOT/'BigSolDBv2.0.csv');d=pd.read_csv(ROOT/'BigSolDBv2.0_densities.csv',dtype=str)
raw=d['Density_g/cm^3'].copy()
d['rho']=pd.to_numeric(raw.str.replace(',','.',regex=False),errors='coerce')
d['T']=pd.to_numeric(d.Temperature_K,errors='coerce')
d['key']=d.Solvent.str.strip().str.lower();s['key']=s.Solvent.str.strip().str.lower()
valid=d.loc[np.isfinite(d.rho)&(d.rho>0)&np.isfinite(d['T'])&(d['T']>0)].copy()
span=valid.groupby('key').agg(T_min=('T','min'),T_max=('T','max'),rows=('T','size'),temperatures=('T','nunique'))
a=s.merge(span,on='key',how='left')
matched=a.T_min.notna();inside=matched&a.Temperature_K.between(a.T_min,a.T_max)
exact=set(zip(valid.key,valid['T']));a['exact_density_temperature']=[(k,t) in exact for k,t in zip(a.key,a.Temperature_K)]
# Density matching is by trimmed lower-case name only; no fuzzy identity guesses.
rhomap={k:g.groupby('T').rho.median().sort_index() for k,g in valid.groupby('key')}
a['rho_interpolated']=np.nan
for k,g in a.loc[inside].groupby('key'):
 points=rhomap[k]
 a.loc[g.index,'rho_interpolated']=np.interp(g.Temperature_K,points.index,points.values)
cache={}
for smi in s.SMILES_Solvent.dropna().unique():
 mol=Chem.MolFromSmiles(smi)
 cache[smi]=Descriptors.MolWt(mol) if mol is not None else np.nan
mw=a.SMILES_Solvent.map(cache)
x=pd.to_numeric(a['Solubility(mole_fraction)'],errors='coerce')
c=pd.to_numeric(a['Solubility(mol/L)'],errors='coerce')
recomputed=x/(1-x)*1000*a.rho_interpolated/mw
ok=inside&(x>0)&(x<1)&(c>0)&(mw>0)&(recomputed>0)&np.isfinite(recomputed)
delta=np.log10(recomputed[ok])-np.log10(c[ok])
missing=sorted(set(s.key)-set(valid.key))
conflicts=valid.groupby(['key','T']).rho.agg(['min','max','size']);conflicts=conflicts.loc[(conflicts['max']-conflicts['min'])>1e-8]
summary={'main_rows':len(s),'main_solvent_names':s.Solvent.nunique(),'density_rows':len(d),'density_solvent_names':d.Solvent.nunique(),
 'decimal_comma_density_rows':int(raw.str.contains(',',regex=False).sum()),'invalid_density_rows':len(d)-len(valid),
 'density_range_g_cm3':[float(valid.rho.min()),float(valid.rho.max())],'density_temperature_range_K':[float(valid['T'].min()),float(valid['T'].max())],
 'main_rows_name_matched':int(matched.sum()),'main_rows_within_density_temperature_span':int(inside.sum()),
 'main_rows_exact_density_temperature':int(a.exact_density_temperature.sum()),'main_rows_outside_matched_density_span':int((matched&~inside).sum()),
 'main_rows_unmatched':int((~matched).sum()),'unmatched_solvent_names':missing,
 'solvent_temperature_groups_with_density_disagreement':len(conflicts),
 'conversion_check_rows':int(ok.sum()),'median_abs_log10_difference_solvent_volume_formula':float(np.median(np.abs(delta))),
 'p95_abs_log10_difference_solvent_volume_formula':float(np.quantile(np.abs(delta),.95)),
 'main_file_md5':hashlib.md5((ROOT/'BigSolDBv2.0.csv').read_bytes()).hexdigest(),
 'density_file_md5':hashlib.md5((ROOT/'BigSolDBv2.0_densities.csv').read_bytes()).hexdigest(),
 'notes':'Name-only match. Interpolation uses median density at each measured temperature, within range only. Conversion discrepancies may reflect publisher linear fits versus our piecewise interpolation. Converted labels are not independent density measurements.'}
(OUT/'audit.json').write_text(json.dumps(summary,indent=2))
span.to_csv(OUT/'density_coverage.csv');conflicts.to_csv(OUT/'density_conflicts.csv')
print(json.dumps(summary,indent=2))
