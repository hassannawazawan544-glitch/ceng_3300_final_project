"""Reproducible ordinary least squares baseline; run on lab machine."""
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
import sklearn, rdkit, joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'linear_v1'
OUT.mkdir(parents=True, exist_ok=True)
source = ROOT / 'BigSolDBv2.0.csv'
raw = pd.read_csv(source)
df = raw.copy()
df['source_row'] = np.arange(len(df)) + 2
for col in ['Temperature_K', 'Solubility(mole_fraction)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
valid = (np.isfinite(df.Temperature_K) & (df.Temperature_K > 0)
         & np.isfinite(df['Solubility(mole_fraction)'])
         & df['Solubility(mole_fraction)'].between(0, 1, inclusive='neither')
         & df.SMILES_Solute.notna() & df.SMILES_Solvent.notna())
df = df.loc[valid].copy()
def canonical(s):
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol) if mol is not None else None
smiles = set(df.SMILES_Solute) | set(df.SMILES_Solvent)
canon = {s: canonical(s) for s in smiles}
df['solute'] = df.SMILES_Solute.map(canon)
df['solvent'] = df.SMILES_Solvent.map(canon)
df = df.dropna(subset=['solute', 'solvent']).copy()
before_dedup = len(df)
df = df.drop_duplicates(['solute','solvent','Temperature_K','Solubility(mole_fraction)','Source']).copy()
df['pair'] = df.solute + ' | ' + df.solvent
df['target'] = np.log10(df['Solubility(mole_fraction)'])
names = ['MolWt','MolLogP','TPSA','HBD','HBA','RotatableBonds','RingCount']
def descriptors(s):
    m = Chem.MolFromSmiles(s)
    return [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
            Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
            Lipinski.NumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m)]
cache = {s: descriptors(s) for s in set(df.solute) | set(df.solvent)}
X = pd.DataFrame({'Temperature_K': df.Temperature_K}, index=df.index)
for role in ['solute','solvent']:
    vals = np.array([cache[s] for s in df[role]])
    for i,n in enumerate(names):
        X[role+'_'+n] = vals[:,i]
train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, groups=df.pair))
assert not set(df.iloc[train].pair) & set(df.iloc[test].pair)
y = df.target.to_numpy()
models = {'mean': DummyRegressor(),
          'temperature_only': make_pipeline(StandardScaler(), LinearRegression()),
          'linear_descriptors': make_pipeline(StandardScaler(), LinearRegression())}
metrics = {}
preds = {}
for name, model in models.items():
    columns = ['Temperature_K'] if name == 'temperature_only' else list(X.columns)
    model.fit(X.iloc[train][columns], y[train])
    p = model.predict(X.iloc[test][columns])
    preds[name] = p
    metrics[name] = {'MAE_log10_x': float(mean_absolute_error(y[test],p)),
                     'RMSE_log10_x': float(np.sqrt(mean_squared_error(y[test],p))),
                     'R2': float(r2_score(y[test],p))}
    joblib.dump({'model':model,'features':columns,'target':'log10(mole_fraction)'}, OUT/(name+'.joblib'))
prediction = df.iloc[test][['source_row','Compound_Name','Solvent','solute','solvent','pair','Source','Temperature_K','Solubility(mole_fraction)','target']].copy()
for name,p in preds.items(): prediction['pred_'+name] = p
prediction.to_csv(OUT/'test_predictions.csv',index=False)
split = df[['source_row','pair']].copy()
split['split'] = 'train'
split.iloc[test,split.columns.get_loc('split')] = 'test'
split.to_csv(OUT/'split.csv',index=False)
model = models['linear_descriptors']
coef = model[-1].coef_ / model[0].scale_
intercept = model[-1].intercept_ - np.sum(coef*model[0].mean_)
pd.DataFrame({'feature':['intercept']+list(X.columns),'coefficient':[intercept]+list(coef)}).to_csv(OUT/'coefficients.csv',index=False)
# Compare measured endpoints within a held-out pair AND publication source.
recovery = []
for (pair,src), group in prediction.groupby(['pair','Source']):
    endpoints = group.groupby('Temperature_K')[['target','pred_linear_descriptors']].median().sort_index()
    if len(endpoints)<2: continue
    cold,hot = endpoints.iloc[0],endpoints.iloc[-1]
    observed_x = 10**np.array([cold.target,hot.target])
    predicted_x = 10**np.array([cold.pred_linear_descriptors,hot.pred_linear_descriptors])
    def fraction(x):
        if not np.all((x>0)&(x<1)): return np.nan
        ratio = x/(1-x)
        return max(0.,1-ratio[0]/ratio[1])
    recovery.append({'pair':pair,'Source':src,'Compound_Name':group.Compound_Name.iloc[0],
      'Solvent':group.Solvent.iloc[0],'T_cold_K':endpoints.index[0],'T_hot_K':endpoints.index[-1],
      'observed_recovery':fraction(observed_x),'predicted_recovery':fraction(predicted_x),
      'observed_hot_mol_per_mol_solvent':observed_x[1]/(1-observed_x[1]),
      'predicted_x_cold':predicted_x[0],'predicted_x_hot':predicted_x[1]})
rec = pd.DataFrame(recovery)
rec.to_csv(OUT/'recovery_endpoints.csv',index=False)
good = rec.dropna(subset=['observed_recovery','predicted_recovery'])
summary = {'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'raw_rows':len(raw),'valid_rows_before_dedup':before_dedup,'rows_after_cleaning':len(df),
 'train_rows':len(train),'test_rows':len(test),'train_pairs':df.iloc[train].pair.nunique(),
 'test_pairs':df.iloc[test].pair.nunique(),'solutes':df.solute.nunique(),'solvents':df.solvent.nunique(),
 'temperature_range_K':[float(df.Temperature_K.min()),float(df.Temperature_K.max())],
 'seed':42,'metrics':metrics,'recovery_endpoint_groups':len(rec),
 'invalid_predicted_endpoint_groups':len(rec)-len(good),
 'recovery_MAE_percentage_points':float(100*mean_absolute_error(good.observed_recovery,good.predicted_recovery)) if len(good) else None,
 'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'rdkit':rdkit.__version__,'numpy':np.__version__,'pandas':pd.__version__}}
(OUT/'metrics.json').write_text(json.dumps(summary,indent=2))
fig,ax = plt.subplots(1,2,figsize=(10,4.3))
ax[0].hexbin(y[test],preds['linear_descriptors'],gridsize=55,mincnt=1,cmap='viridis',bins='log')
lims=[min(y[test].min(),preds['linear_descriptors'].min()),max(y[test].max(),preds['linear_descriptors'].max())]
ax[0].plot(lims,lims,'r--'); ax[0].set(xlabel='Measured log10(mole fraction)',ylabel='Predicted log10(mole fraction)',title='Held-out solute–solvent pairs')
ax[1].scatter(100*good.observed_recovery,100*good.predicted_recovery,s=6,alpha=.25)
ax[1].plot([0,100],[0,100],'r--');ax[1].set(xlabel='Measured-endpoint recovery (%)',ylabel='Predicted-endpoint recovery (%)',title='Cooling recovery: same-source endpoints')
fig.tight_layout();fig.savefig(OUT/'baseline_results.png',dpi=180);plt.close(fig)
report = f'''# Linear regression baseline v1

Target: log10 solubility in mole fraction. Inputs: temperature (K), plus solute and solvent molecular weight, logP, TPSA, H-bond donors/acceptors, rotatable bonds and ring count. Ordinary least squares with training-only scaling; 15 features. No measured solubility or density is an input. Mean and temperature-only references use the same split.

Split: 80/20 canonical solute–solvent pairs, seed 42. Every temperature/source record of a pair stays in one partition. This tests new pairs, not necessarily unseen individual molecules. Raw data is preserved; invalid target/temperature/SMILES rows and exact chemical/temperature/target/source duplicates are removed. Conflicting measurements remain. No hyperparameter tuning or test-based selection.

Results:
{json.dumps(summary,indent=2)}

Recovery is an ideal equilibrium estimate for a hot saturated solution cooled at constant solvent amount: r=x/(1-x), recovery=max(0,1-r_cold/r_hot). Endpoints are measured minimum/maximum temperatures within the same test pair and source; replicate log targets are median-aggregated. It is not measured crystallization yield. Assumes equilibrium, no solvent loss, no polymorph changes, and pure-solvent interpretation. No fixed universal process temperatures are imposed. Endpoint intervals differ, so these rows are not a universal solvent ranking.

Limitations: an additive linear model has one common temperature slope for all pairs, cannot capture pair-specific interactions/curvature, and may predict x outside (0,1). Invalid endpoint predictions are excluded from recovery MAE and counted explicitly. Results are one split, not uncertainty estimates or publication-ready validation. Recovery error compounds solubility errors. We have not modeled energy duty, kinetics, safety, or process economics. Densities are intentionally unused: mole-fraction conversion already gives mol solute per mol solvent.

Run from the project root:
`/home/puqing/anaconda3/envs/jupyter/bin/python scripts/train_linear.py`
'''
(OUT/'report.md').write_text(report)
assert np.isfinite(preds['linear_descriptors']).all()
assert good.observed_recovery.between(0,1).all() and good.predicted_recovery.between(0,1).all()
assert np.allclose(model.predict(X.iloc[test]),intercept+X.iloc[test].to_numpy()@coef)
print(json.dumps(summary,indent=2))
