"""Unified paired evaluation of completed ADV-001 runs. No model fitting.
An optional convex ensemble is selected on validation solubility MAE only.
"""
from pathlib import Path
import json,itertools,hashlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from advanced_common import *

def main():
 df,f=load();test=df.split.eq('test').to_numpy();val=df.split.eq('validation').to_numpy();preds={};checks={}
 for file in sorted(OUT.glob('*/predictions.csv')):
  if not (file.parent/'metrics.json').exists():continue
  z=pd.read_csv(file)
  assert np.array_equal(z.source_row,df.source_row),file
  assert np.array_equal(z['split'],df['split']),file
  assert np.allclose(z.target,df.target,rtol=0,atol=1e-10),file
  p=z.predicted_log10_x.to_numpy();assert np.isfinite(p).all(),file
  preds[file.parent.name]=p
  # Verify reported ensembles against all seed outputs, including identical row order.
  seeds=sorted(file.parent.glob('seed_*/predictions.csv'))
  if seeds:
   q=[]
   for sf in seeds:
    sd=pd.read_csv(sf);assert np.array_equal(sd.source_row,df.source_row);q.append(sd.predicted_log10_x.to_numpy())
   assert len(seeds)==3 and np.allclose(p,np.mean(q,axis=0),atol=2e-6),file
   checks[file.parent.name]={'seeds':3,'max_ensemble_difference':float(np.max(np.abs(p-np.mean(q,axis=0))))}
 # Include comparable CLASS-001 predictors on exactly the same held-out rows.
 cls=pd.read_csv(ROOT/'results/class_v1/test_predictions.csv');assert np.array_equal(cls.source_row,df.loc[test,'source_row'])
 for name in ['mean','ridge_inverseT_interactions','xgboost']:
  p=np.full(len(df),np.nan);p[test]=cls['pred_'+name].to_numpy();preds['class_'+name]=p
 # Select top 3 independent families by validation MAE, before evaluating ensemble test.
 candidates=[n for n in preds if not n.startswith('class_') and n!='validation_convex_ensemble']
 top=sorted(candidates,key=lambda n:float(np.mean(np.abs(preds[n][val]-df.loc[val,'target']))))[:3]
 best=None
 for a in range(11):
  for b in range(11-a):
   weights=np.array([a,b,10-a-b])/10;p=sum(w*preds[n] for w,n in zip(weights,top));mae=float(np.mean(np.abs(p[val]-df.loc[val,'target'])))
   if best is None or mae<best[0]:best=(mae,weights,p)
 preds['validation_convex_ensemble']=best[2];save_prediction(df,best[2],'validation_convex_ensemble')
 ensemble={'members':top,'weights':best[1].tolist(),'validation_MAE':best[0],'rule':'Top three families by validation MAE; nonnegative weights on 0.1 grid summing to 1; no test-based selection. Not an independent new model or calibrated uncertainty.'}
 (OUT/'validation_convex_ensemble/ensemble.json').write_text(json.dumps(ensemble,indent=2))
 # Molecular weights used solely for mass-balance conversion.
 mw={s:Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in set(df.solute)|set(df.solvent)}
 base=df.loc[test].copy();base['observed']=base.target
 for n,p in preds.items():base[n]=p[test]
 groups=list(base.groupby(['pair','Source']));end=[];common=[]
 def physical(lp,ms,mv,rho):
  x=10**np.asarray(lp,dtype=float)
  if not np.all(np.isfinite(x)&(x>0)&(x<1)):return (np.nan,np.nan,np.nan)
  rr=x/(1-x);q=rr*ms/mv;recovery=max(0.,1-rr[0]/rr[1]);mass=1000*rho*max(0.,q[1]-q[0]) if np.isfinite(rho) else np.nan
  return recovery,q[1],mass
 cols=['observed']+list(preds)
 for (pair,source),g in groups:
  curve=g.groupby('Temperature_K')[cols+['rho_g_cm3']].median().sort_index()
  if len(curve)<2:continue
  cold,hot=curve.iloc[0],curve.iloc[-1];ms=mw[g.solute.iloc[0]];mv=mw[g.solvent.iloc[0]]
  rec={'pair':pair,'Source':source,'solute':g.solute.iloc[0],'Solvent':g.Solvent.iloc[0],'Compound_Name':g.Compound_Name.iloc[0],'T_cold_K':curve.index[0],'T_hot_K':curve.index[-1]}
  for n in cols:
   rv,q,mass=physical([cold[n],hot[n]],ms,mv,hot.rho_g_cm3);rec[n+'_recovery']=rv;rec[n+'_mass_g_L']=mass;rec[n+'_delta_logx']=hot[n]-cold[n]
  end.append(rec)
  coldx=g.loc[np.isclose(g.Temperature_K,298.15,atol=.005,rtol=0)];hotx=g.loc[np.isclose(g.Temperature_K,323.15,atol=.005,rtol=0)]
  if coldx.empty or hotx.empty:continue
  rho=hotx.rho_g_cm3.median()
  if not np.isfinite(rho):continue
  rec={k:rec[k] for k in ['pair','Source','solute','Solvent','Compound_Name']}
  for n in cols:
   rv,q,mass=physical([coldx[n].median(),hotx[n].median()],ms,mv,rho);rec[n+'_recovery']=rv;rec[n+'_mass_g_L']=mass
  common.append(rec)
 endpoints=pd.DataFrame(end);cases=pd.DataFrame(common);endpoints.to_csv(OUT/'recovery_all_models.csv',index=False);cases.to_csv(OUT/'common_temperature_all_models.csv',index=False)
 shared=endpoints[[n+'_recovery' for n in preds]].notna().all(axis=1);summary=[];decision_records=[]
 eligible=[g for _,g in cases.groupby(['solute','Source']) if g.Solvent.nunique()>=3]
 # For a fair comparison, every method must support every option of a shared decision group.
 shared_decisions=[g for g in eligible if g[[n+'_mass_g_L' for n in preds]].notna().all().all()]
 for n,p in preds.items():
  s={'model':n,**score(df.loc[test,'target'],p[test],df.loc[test,'solute'])}
  s['validation_MAE']=float(np.mean(np.abs(p[val]-df.loc[val,'target']))) if np.isfinite(p[val]).all() else None
  ok=endpoints[n+'_recovery'].notna();s['recovery_groups_valid']=int(ok.sum());s['recovery_groups_invalid']=int((~ok).sum());s['recovery_MAE_pp']=float(100*np.mean(np.abs(endpoints.loc[ok,n+'_recovery']-endpoints.loc[ok,'observed_recovery'])))
  s['shared_recovery_groups']=int(shared.sum());s['shared_recovery_MAE_pp']=float(100*np.mean(np.abs(endpoints.loc[shared,n+'_recovery']-endpoints.loc[shared,'observed_recovery'])))
  s['delta_logx_MAE']=float(np.mean(np.abs(endpoints[n+'_delta_logx']-endpoints.observed_delta_logx)))
  volumetric=endpoints[n+'_mass_g_L'].notna();err=np.abs(endpoints.loc[volumetric,n+'_mass_g_L']-endpoints.loc[volumetric,'observed_mass_g_L']);s['mass_groups_valid']=int(volumetric.sum());s['mass_MAE_g_L']=float(err.mean());s['mass_median_abs_error_g_L']=float(err.median())
  results=[]
  for g in shared_decisions:
   oracle=g.loc[g.observed_mass_g_L.idxmax()];choice=g.loc[g[n+'_mass_g_L'].idxmax()];r={'model':n,'solute':choice.solute,'Source':choice.Source,'n_solvents':len(g),'reference_best_solvent':oracle.Solvent,'selected_solvent':choice.Solvent,'reference_best_mass_g_L':oracle.observed_mass_g_L,'selected_reference_mass_g_L':choice.observed_mass_g_L,'regret_g_L':oracle.observed_mass_g_L-choice.observed_mass_g_L,'exact_match':int(oracle.Solvent==choice.Solvent)};decision_records.append(r);results.append(r)
  s['shared_decision_groups']=len(results);s['top_choice_matches']=sum(r['exact_match'] for r in results);s['choice_mean_regret_g_L']=float(np.mean([r['regret_g_L'] for r in results]));s['choice_median_regret_g_L']=float(np.median([r['regret_g_L'] for r in results]));summary.append(s)
 pd.DataFrame(decision_records).to_csv(OUT/'decisions_all_models.csv',index=False)
 table=pd.DataFrame(summary).sort_values('validation_MAE',na_position='last');table.to_csv(OUT/'comparison.csv',index=False)
 # Paired macro-solute MAE differences vs original class XGBoost; group bootstrap.
 te=df.loc[test].copy();ref=np.abs(df.loc[test,'target'].to_numpy()-preds['class_xgboost'][test]);rng=np.random.default_rng(20261106);intervals=[]
 for n,p in preds.items():
  diff=np.abs(te.target.to_numpy()-p[test])-ref;macro=pd.DataFrame({'solute':te.solute.to_numpy(),'difference':diff}).groupby('solute').difference.mean().to_numpy();boots=np.mean(rng.choice(macro,size=(2000,len(macro)),replace=True),axis=1)
  intervals.append({'model':n,'macro_MAE_difference_vs_class_xgb':float(macro.mean()),'lower95':float(np.quantile(boots,.025)),'upper95':float(np.quantile(boots,.975))})
 pd.DataFrame(intervals).to_csv(OUT/'paired_bootstrap.csv',index=False)
 # Save portable held-out evidence; full weights stay on the lab host.
 testoutput=df.loc[test,['source_row','solute','solvent','Source','Temperature_K','target','rho_g_cm3']].copy()
 for n,p in preds.items():testoutput[n]=p[test]
 testoutput.to_csv(OUT/'test_all_models.csv',index=False)
 meta={'n_model_entries':len(preds),'test_rows':int(test.sum()),'test_solutes':int(df.loc[test,'solute'].nunique()),'endpoint_groups':len(endpoints),'shared_valid_endpoint_groups':int(shared.sum()),'common_temperature_rows':len(cases),'eligible_decision_groups':len(eligible),'shared_valid_decision_groups':len(shared_decisions),'ensemble':ensemble,'checks':checks,'caveats':'One internal solute-group development split; source and phase dependence unresolved. Ideal equilibrium recovery, not isolated yield. Density interpolation uncertainty not included. Invalid mole-fraction endpoint predictions excluded and counted; shared comparisons restrict to all-method-valid groups. Bootstrap is conditional sampling variability, not a model prediction interval or multiplicity-corrected significance test.'}
 (OUT/'evaluation_manifest.json').write_text(json.dumps(meta,indent=2))
 # Internal consistency of independently implemented physical calculations.
 assert np.isclose(physical(np.log10([.01,.02]),100,50,.8)[2],1000*.8*(.02/.98-.01/.99)*2)
 assert endpoints.observed_recovery.between(0,1).all()
 columns=['model','validation_MAE','MAE_log10_x','RMSE_log10_x','R2','shared_recovery_MAE_pp','top_choice_matches','shared_decision_groups']
 lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
 for row in summary:lines.append('| '+' | '.join((f'{row[c]:.4f}' if isinstance(row[c],float) else str(row[c])) for c in columns)+' |')
 (OUT/'comparison.md').write_text('# Advanced-model comparison\n\n'+'\n'.join(lines)+'\n\n'+json.dumps(meta,indent=2)+'\n')
 print(table[['model','validation_MAE','MAE_log10_x','R2','shared_recovery_MAE_pp','top_choice_matches','shared_decision_groups']].to_string(index=False),flush=True);print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
