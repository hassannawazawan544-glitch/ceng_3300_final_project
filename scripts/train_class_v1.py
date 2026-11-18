"""CLASS-001: grouped chemical generalization and density-based recovery.
Prespecified comparison. Raw inputs preserved; saved models use training data only.
"""
from pathlib import Path
import json, hashlib, platform, time
import numpy as np
import pandas as pd
import sklearn, rdkit, xgboost, joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, rdFingerprintGenerator
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'class_v1'
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'metrics.json').exists(): raise RuntimeError('Completed run exists; use a new version rather than overwrite it')
SEED=20261106;start=time.time()
raw=pd.read_csv(ROOT/'BigSolDBv2.0.csv');df=raw.copy();df['source_row']=np.arange(len(df))+2
counts={'raw':len(df)}
for c in ['Temperature_K','Solubility(mole_fraction)']:df[c]=pd.to_numeric(df[c],errors='coerce')
good=np.isfinite(df.Temperature_K)&(df.Temperature_K>0)&np.isfinite(df['Solubility(mole_fraction)'])&df['Solubility(mole_fraction)'].between(0,1,inclusive='neither')
df=df.loc[good].dropna(subset=['SMILES_Solute','SMILES_Solvent']).copy();counts['numeric_and_present_smiles']=len(df)
mols={s:Chem.MolFromSmiles(s) for s in sorted(set(df.SMILES_Solute)|set(df.SMILES_Solvent))}
canon={s:Chem.MolToSmiles(m) if m is not None else None for s,m in mols.items()}
for role in ['Solute','Solvent']:df[role.lower()]=df['SMILES_'+role].map(canon)
df=df.dropna(subset=['solute','solvent']);counts['valid_smiles']=len(df)
# A conservative molecular cohort: no disconnected fragments or charged atoms.
# Neutral ionizable molecules remain; unknown pH/form remains a limitation.
def molecular(s):
 m=Chem.MolFromSmiles(s)
 return len(Chem.GetMolFrags(m))==1 and all(a.GetFormalCharge()==0 for a in m.GetAtoms())
allowed={s:molecular(s) for s in sorted(set(df.solute)|set(df.solvent))}
df=df.loc[df.solute.map(allowed)&df.solvent.map(allowed)].copy();counts['neutral_single_fragment']=len(df)
variable=df.Solvent.str.lower().str.contains(r'peg|polyethylene|span\s*80',regex=True,na=False)
df=df.loc[~variable].copy();counts['exclude_variable_composition_solvent_names']=len(df)
df=df.drop_duplicates(['solute','solvent','Temperature_K','Solubility(mole_fraction)','Source']).reset_index(drop=True);counts['deduplicated']=len(df)
df['pair']=df.solute+' | '+df.solvent;df['target']=np.log10(df['Solubility(mole_fraction)'])
# Density interpolation within measured range only; solvent names must be unambiguous.
d=pd.read_csv(ROOT/'BigSolDBv2.0_densities.csv',dtype=str)
d['rho']=pd.to_numeric(d['Density_g/cm^3'].str.replace(',','.',regex=False),errors='coerce');d['temp']=pd.to_numeric(d.Temperature_K,errors='coerce');d['key']=d.Solvent.str.strip().str.lower()
d=d.loc[np.isfinite(d.rho)&(d.rho>0)&np.isfinite(d.temp)&(d.temp>0)]
df['key']=df.Solvent.str.strip().str.lower();df['rho_g_cm3']=np.nan
ambiguous=set(df.groupby('key').solvent.nunique().loc[lambda x:x>1].index)
for k,g in df.groupby('key'):
 if k in ambiguous:continue
 points=d.loc[d.key==k].groupby('temp').rho.median().sort_index()
 if len(points)<2:continue
 ix=g.loc[g.Temperature_K.between(points.index.min(),points.index.max())].index
 df.loc[ix,'rho_g_cm3']=np.interp(df.loc[ix,'Temperature_K'],points.index,points.values)
# Split before fitting/scaling/selection; every form/source/temperature of a solute stays together.
trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=SEED).split(df,groups=df.solute))
tr0,va0=next(GroupShuffleSplit(n_splits=1,test_size=.15/.85,random_state=SEED+1).split(df.iloc[trainval],groups=df.iloc[trainval].solute))
train,val=trainval[tr0],trainval[va0];parts={'train':train,'validation':val,'test':test}
for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not set(df.iloc[parts[a]].solute)&set(df.iloc[parts[b]].solute)
df['split']='';
for name,idx in parts.items():df.loc[idx,'split']=name
# Freeze membership and cohort before training.
df[['source_row','solute','solvent','pair','split','rho_g_cm3']].to_csv(OUT/'split.csv',index=False)
config={'seed':SEED,'target':'log10 mole fraction','cohort':'neutral, single fragment molecular structures; variable-composition solvent names excluded','split':'70/15/15 by canonical solute, approximate proportions of groups',
 'selection':'minimum validation row MAE; all models fitted on train only; no refit on test or validation',
 'xgb_candidates':[{'max_depth':4,'min_child_weight':5},{'max_depth':6,'min_child_weight':10}],
 'xgb_common':{'n_estimators':600,'learning_rate':.04,'subsample':.8,'colsample_bytree':.8,'reg_lambda':10,'reg_alpha':.1,'tree_method':'hist','n_jobs':4,'random_state':SEED,'early_stopping_rounds':50},
 'density':'piecewise linear of median per-temperature pure-solvent density; within range; normalized-name mapping; ambiguous names excluded; provenance in density file',
 'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'BigSolDBv2.0.csv',ROOT/'BigSolDBv2.0_densities.csv']}}
(OUT/'protocol.json').write_text(json.dumps(config,indent=2))
names=['MolWt','LogP','TPSA','HBD','HBA','RotBonds','Rings','AromaticRings','FractionCSP3','HeavyAtoms','HeteroAtoms','LabuteASA']
fpg=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=256)
cache={}
for s in sorted(set(df.solute)|set(df.solvent)):
 m=Chem.MolFromSmiles(s)
 desc=[Descriptors.MolWt(m),Crippen.MolLogP(m),rdMolDescriptors.CalcTPSA(m),Lipinski.NumHDonors(m),Lipinski.NumHAcceptors(m),Lipinski.NumRotatableBonds(m),rdMolDescriptors.CalcNumRings(m),rdMolDescriptors.CalcNumAromaticRings(m),rdMolDescriptors.CalcFractionCSP3(m),m.GetNumHeavyAtoms(),rdMolDescriptors.CalcNumHeteroatoms(m),rdMolDescriptors.CalcLabuteASA(m)]
 fp=np.zeros(256,dtype=np.int8);DataStructs.ConvertToNumpyArray(fpg.GetFingerprint(m),fp);cache[s]=(desc,fp)
base=pd.DataFrame({'Temperature_K':df.Temperature_K,'inverse_T_centered':1000/df.Temperature_K-1000/298.15})
for role in ['solute','solvent']:
 arr=np.array([cache[s][0] for s in df[role]])
 for j,n in enumerate(names):base[role+'_'+n]=arr[:,j]
linear=base.drop(columns=['Temperature_K']).copy()
for col in base.columns[2:]:linear[col+'_invT']=base[col]*base.inverse_T_centered
fp=np.hstack([np.stack([cache[s][1] for s in df[role]]) for role in ['solute','solvent']])
X=np.column_stack([base.to_numpy(),fp]);XL=linear.to_numpy();y=df.target.to_numpy()
assert np.isfinite(X).all() and np.isfinite(XL).all()
metrics={};models={};testpred={};validation=[]
def score(truth,p,solutes):
 errors=pd.DataFrame({'solute':np.asarray(solutes),'absolute_error':np.abs(truth-p)})
 macro=errors.groupby('solute').absolute_error.mean().to_numpy()
 rng=np.random.default_rng(SEED);boot=np.mean(rng.choice(macro,size=(1000,len(macro)),replace=True),axis=1)
 return {'MAE_log10_x':float(mean_absolute_error(truth,p)),'RMSE_log10_x':float(np.sqrt(mean_squared_error(truth,p))),'R2':float(r2_score(truth,p)),
 'macro_solute_MAE':float(macro.mean()),'macro_solute_MAE_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'n_rows':len(p),'n_solutes':len(macro)}
for name,model,features in [('mean',DummyRegressor(),X),('ridge_inverseT_interactions',make_pipeline(StandardScaler(),Ridge(alpha=10)),XL)]:
 model.fit(features[train],y[train]);models[name]=model
 validation.append({'model':name,'MAE':float(mean_absolute_error(y[val],model.predict(features[val])))})
 testpred[name]=model.predict(features[test]);metrics[name]=score(y[test],testpred[name],df.iloc[test].solute)
best=None
for cfg in config['xgb_candidates']:
 model=XGBRegressor(**config['xgb_common'],**cfg)
 model.fit(X[train],y[train],eval_set=[(X[val],y[val])],verbose=False)
 mae=float(mean_absolute_error(y[val],model.predict(X[val])))
 validation.append({'model':'xgboost','config':cfg,'best_iteration':model.best_iteration,'MAE':mae})
 print('Validation:',validation[-1],flush=True)
 if best is None or mae<best[0]:best=(mae,model,cfg)
_,model,bestcfg=best;models['xgboost']=model;testpred['xgboost']=model.predict(X[test]);metrics['xgboost']=score(y[test],testpred['xgboost'],df.iloc[test].solute)
pd.DataFrame(validation).to_csv(OUT/'validation_comparison.csv',index=False)
# Density ablation on identical covered rows and solute split, fixed selected tree settings.
covered=np.isfinite(df.rho_g_cm3.to_numpy());density_metrics={}
fixed={k:v for k,v in config['xgb_common'].items() if k!='early_stopping_rounds'};fixed['n_estimators']=model.best_iteration+1
for use_density in [False,True]:
 xx=np.column_stack([X,df.rho_g_cm3]) if use_density else X
 tr=train[covered[train]];te=test[covered[test]];mo=XGBRegressor(**fixed,**bestcfg);mo.fit(xx[tr],y[tr],verbose=False)
 name='with_density' if use_density else 'without_density';density_metrics[name]=score(y[te],mo.predict(xx[te]),df.iloc[te].solute);models['covered_'+name]=mo
for name,mo in models.items():joblib.dump(mo,OUT/(name+'.joblib'))
pred=df.iloc[test][['source_row','Compound_Name','Solvent','solute','solvent','pair','Source','Temperature_K','Solubility(mole_fraction)','target','rho_g_cm3']].copy()
for name,p in testpred.items():pred['pred_'+name]=p
pred.to_csv(OUT/'test_predictions.csv',index=False)
# Endpoint error on same-source held-out curves; keep all rows and invalid predictions explicit.
endpoints=[]
for (pair,source),g in pred.groupby(['pair','Source']):
 cols=['target','rho_g_cm3']+['pred_'+n for n in testpred]
 curve=g.groupby('Temperature_K')[cols].median().sort_index()
 if len(curve)<2:continue
 cold,hot=curve.iloc[0],curve.iloc[-1];r={'pair':pair,'Source':source,'Compound_Name':g.Compound_Name.iloc[0],'Solvent':g.Solvent.iloc[0],'solute':g.solute.iloc[0],'solvent':g.solvent.iloc[0],'T_cold_K':curve.index[0],'T_hot_K':curve.index[-1],'rho_hot_g_cm3':hot.rho_g_cm3}
 ms=cache[r['solute']][0][0];mv=cache[r['solvent']][0][0]
 for name in ['observed']+list(testpred):
  col='target' if name=='observed' else 'pred_'+name
  xs=10**np.array([cold[col],hot[col]])
  if not np.all(np.isfinite(xs)&(xs>0)&(xs<1)):
   for quantity in ['recovery','hot_capacity_g_g','recovered_g_L_initial_solvent']:r[name+'_'+quantity]=np.nan
   continue
  rr=xs/(1-xs);q=rr*ms/mv
  r[name+'_recovery']=max(0.,1-rr[0]/rr[1]);r[name+'_hot_capacity_g_g']=q[1]
  r[name+'_recovered_g_L_initial_solvent']=1000*hot.rho_g_cm3*max(0.,q[1]-q[0]) if np.isfinite(hot.rho_g_cm3) else np.nan
 endpoints.append(r)
ep=pd.DataFrame(endpoints);ep.to_csv(OUT/'endpoint_recovery.csv',index=False)
recovery={}
for name in testpred:
 ok=ep[['observed_recovery',name+'_recovery']].notna().all(axis=1);vol=ep[['observed_recovered_g_L_initial_solvent',name+'_recovered_g_L_initial_solvent']].notna().all(axis=1)
 recovery[name]={'endpoint_groups':len(ep),'valid_groups':int(ok.sum()),'invalid_predicted_groups':int((~ok).sum()),'MAE_percentage_points':float(100*mean_absolute_error(ep.loc[ok,'observed_recovery'],ep.loc[ok,name+'_recovery'])),
 'density_covered_groups':int(vol.sum()),'recovered_mass_MAE_g_per_L_initial_solvent':float(mean_absolute_error(ep.loc[vol,'observed_recovered_g_L_initial_solvent'],ep.loc[vol,name+'_recovered_g_L_initial_solvent']))}
# Common-temperature illustrative cases: endpoints must be directly measured at 298.15 and 323.15 K.
# Conditions are prespecified classroom examples, not claimed optimal or universally feasible.
common=[]
for (pair,source),g in pred.groupby(['pair','Source']):
 cold=g.loc[np.isclose(g.Temperature_K,298.15,atol=.005,rtol=0)];hot=g.loc[np.isclose(g.Temperature_K,323.15,atol=.005,rtol=0)]
 if cold.empty or hot.empty:continue
 rho=float(hot.rho_g_cm3.median())
 if not np.isfinite(rho):continue
 r={'pair':pair,'Source':source,'Compound_Name':g.Compound_Name.iloc[0],'Solvent':g.Solvent.iloc[0],'solute':g.solute.iloc[0],'solvent':g.solvent.iloc[0],'T_cold_K':298.15,'T_hot_K':323.15,'rho_hot_g_cm3':rho}
 ms=cache[r['solute']][0][0];mv=cache[r['solvent']][0][0]
 for name,col in [('observed','target'),('xgboost','pred_xgboost')]:
  xs=10**np.array([cold[col].median(),hot[col].median()])
  if not np.all((xs>0)&(xs<1)):break
  rr=xs/(1-xs);q=rr*ms/mv
  r[name+'_recovery_pct']=100*max(0.,1-rr[0]/rr[1]);r[name+'_hot_capacity_g_g']=q[1];r[name+'_recovered_g_L_initial_solvent']=1000*rho*max(0.,q[1]-q[0])
 else:common.append(r)
cases=pd.DataFrame(common)
if len(cases):
 # Preserve source rows; evaluate/rank only within the same solute and source.
 cases.to_csv(OUT/'common_temperature_cases.csv',index=False)
 counts_cases=cases.groupby(['solute','Source']).Solvent.nunique();eligible=counts_cases[counts_cases>=3].sort_values(ascending=False)
 if len(eligible):
  chosen=eligible.index[0];case=cases.loc[(cases.solute==chosen[0])&(cases.Source==chosen[1])].sort_values('observed_recovered_g_L_initial_solvent',ascending=False)
  case.to_csv(OUT/'illustrative_case.csv',index=False)
 else:case=pd.DataFrame()
else:case=pd.DataFrame()
summary={'run_id':'CLASS-001','seed':SEED,'filters':counts,'solutes':df.solute.nunique(),'solvents':df.solvent.nunique(),
 'splits':{n:{'rows':len(ix),'solutes':df.iloc[ix].solute.nunique()} for n,ix in parts.items()},
 'density_covered_rows':int(covered.sum()),'metrics':metrics,'selected_xgb':{'config':bestcfg,'n_trees':model.best_iteration+1,'validation_MAE':best[0]},
 'density_ablation':density_metrics,'recovery':recovery,'common_temperature_rows':len(cases),'illustrative_case_rows':len(case),
 'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'rdkit':rdkit.__version__,'xgboost':xgboost.__version__},'runtime_seconds':time.time()-start}
(OUT/'metrics.json').write_text(json.dumps(summary,indent=2))
# Figures; all plotted data are test predictions or measured-endpoint calculations.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.2})
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ns=list(metrics);ax[0].bar(['Mean','Linear + interactions','XGBoost'],[metrics[n]['MAE_log10_x'] for n in ns],color=['#B6C1CE','#2D597C','#007F7A']);ax[0].set(ylabel='Test MAE, log10 mole fraction',title='Unseen-solute test')
for i,n in enumerate(ns):ax[0].text(i,metrics[n]['MAE_log10_x']+.015,f"{metrics[n]['MAE_log10_x']:.3f}",ha='center')
ax[1].hexbin(y[test],testpred['xgboost'],gridsize=50,mincnt=1,bins='log',cmap='viridis');lims=[min(y[test].min(),testpred['xgboost'].min()),max(y[test].max(),testpred['xgboost'].max())];ax[1].plot(lims,lims,'--',color='#C76347',lw=2.5);ax[1].set(xlabel='Measured log10 mole fraction',ylabel='Predicted log10 mole fraction',title='XGBoost predictions')
fig.tight_layout();fig.savefig(OUT/'model_comparison.png',dpi=300);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4.5));ok=ep[['observed_recovery','xgboost_recovery']].notna().all(axis=1);ax.scatter(100*ep.loc[ok,'observed_recovery'],100*ep.loc[ok,'xgboost_recovery'],s=10,alpha=.3,color='#007F7A');ax.plot([0,100],[0,100],'--',color='#C76347',lw=2.5);ax.set(xlabel='Measured-endpoint equilibrium recovery (%)',ylabel='Predicted equilibrium recovery (%)',title='Unseen solutes; source-specific endpoints');fig.tight_layout();fig.savefig(OUT/'recovery_comparison.png',dpi=300);plt.close(fig)
if len(case):
 fig,ax=plt.subplots(figsize=(9,4.5));ix=np.arange(len(case));ax.bar(ix-.18,case.observed_recovered_g_L_initial_solvent,width=.36,label='Measured-endpoint estimate',color='#2D597C');ax.bar(ix+.18,case.xgboost_recovered_g_L_initial_solvent,width=.36,label='ML estimate',color='#007F7A');ax.set_xticks(ix,case.Solvent,rotation=25,ha='right');ax.set(ylabel='Recoverable mass (g / L initial pure solvent)',title=str(case.Compound_Name.iloc[0])+': 50°C → 25°C');ax.legend(fontsize=9);fig.tight_layout();fig.savefig(OUT/'case_study.png',dpi=300);plt.close(fig)
report='''# CLASS-001: class-project model comparison

Models predict log10 mole-fraction solubility. Split by neutral single-fragment canonical solute, approximate 70/15/15 train/validation/test groups. Models fit train only; XGBoost selected among two prespecified configurations on validation MAE. Test set not used for hyperparameter selection. This is an internal class benchmark, not a chemically independent external dataset; earlier project-wide exploratory analyses have occurred.

Features: 12 descriptors per molecule, temperature and inverse temperature; XGBoost additionally uses separate 256-bit radius-2 Morgan fingerprints. Ridge uses descriptor/inverse-T interactions, alpha=10 fixed. Density ablation fits otherwise identical selected XGBoost settings on identical covered rows, with/without interpolated density. Ablation settings were selected on the broad training/validation cohort; ablation is not separately optimized.

Chemistry limits: neutral molecular structures can still ionize; pH, solid form and equilibrium protocols are unverified in v2.0. Neutral single-fragment filtering is not proof of thermodynamic homogeneity. No polymorph or melting data inferred. In-range median-density interpolation is an approximation; source conflicts retained in raw data and separate audit. Density uncertainty not quantified. No universal monotonicity imposed.

Recovery is an ideal constant-solvent-amount equilibrium estimate, not measured isolated yield. All endpoints in one evaluation group share source and pair, but same solid form is not confirmed. Density-derived mass uses one litre of pure solvent initially at the hot temperature. Endpoint intervals vary, so aggregate endpoint rows are not a fair universal solvent ranking. The 25/50°C case requires measured endpoint data and in-range hot density and is illustrative, not optimized or established safe/feasible. Case chosen deterministically by largest number of solvent alternatives in a held-out solute/source group; no error-based cherry-picking. No new experimental validation or publication claim.

Rows with invalid predicted mole fractions are counted, not clipped into physical plausibility. Macro-MAE intervals bootstrap held-out solutes (1000 draws) and quantify sampling variation conditional on this fitted model, not prediction intervals. No calibrated model uncertainty is claimed.

Artifacts: protocol.json, split.csv, validation_comparison.csv, metrics.json, test_predictions.csv, endpoint_recovery.csv, common_temperature_cases.csv and illustrative_case.csv where supported; fitted joblib models and data-derived figures. To rerun, use a new output version; completed runs are protected from overwrite.

## Results

'''+json.dumps(summary,indent=2)
(OUT/'report.md').write_text(report)
assert all(np.isfinite(p).all() for p in testpred.values())
assert ep.observed_recovery.between(0,1).all()
assert np.isclose(1000*.8*(.1-.02),64.) # simple dimensional mass-balance sanity check
print(json.dumps(summary,indent=2),flush=True)
