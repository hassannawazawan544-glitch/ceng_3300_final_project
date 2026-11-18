from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/advanced_v1'
def load():
 d=pd.read_csv(OUT/'cohort.csv');f=np.load(OUT/'molecular_features.npz');return d,f

def representations(df,f,kind):
 parts=[]
 for role in ['solute','solvent']:
  ids=f[role+'_idx'];v=[]
  if kind in ['rdkit','combined']:v.append(f['rdkit'][ids])
  if kind=='mordred':v.append(np.load(OUT/'mordred_features.npz')['mordred'][ids])
  if kind in ['morgan2','combined']:v.append(np.log1p(f['morgan2'][ids]))
  if kind=='morgan3':v.append(np.log1p(f['morgan3'][ids]))
  parts.append(np.column_stack(v))
 t=np.column_stack([df.Temperature_K,1000/df.Temperature_K-1000/298.15])
 X=np.column_stack(parts+[t]).astype(np.float32);X[~np.isfinite(X)]=np.nan
 return X

def score(y,p,solute):
 y=np.asarray(y);p=np.asarray(p);assert len(y)==len(p) and np.isfinite(p).all()
 macro=pd.DataFrame({'solute':np.asarray(solute),'err':np.abs(y-p)}).groupby('solute').err.mean()
 return {'MAE_log10_x':float(mean_absolute_error(y,p)),'RMSE_log10_x':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)),'macro_solute_MAE':float(macro.mean()),'n_rows':len(y),'n_solutes':len(macro),'predicted_x_ge_1':int((p>=0).sum())}

def save_prediction(df,p,name):
 z=df[['source_row','split','solute','solvent','pair','Source','Temperature_K','target','rho_g_cm3']].copy();z['predicted_log10_x']=p
 path=OUT/name;path.mkdir(parents=True,exist_ok=True);z.to_csv(path/'predictions.csv',index=False)
 metrics={}
 for split in ['train','validation','test']:
  mask=z.split.eq(split);metrics[split]=score(z.loc[mask,'target'],z.loc[mask,'predicted_log10_x'],z.loc[mask,'solute'])
 (path/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics
