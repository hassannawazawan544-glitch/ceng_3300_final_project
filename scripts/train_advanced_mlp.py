"""Paired-molecule MLP ensembles and apparent van't Hoff head.
Architecture is literature-inspired, not an exact FASTSOLV reproduction.
"""
import json,time,random,copy
import numpy as np
import torch
from torch import nn
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from advanced_common import *
torch.set_num_threads(4)
class PairNet(nn.Module):
 def __init__(self,n,thermo=False):
  super().__init__();self.thermo=thermo
  def tower():return nn.Sequential(nn.Linear(n,256),nn.SiLU(),nn.Dropout(.15),nn.Linear(256,128),nn.SiLU())
  self.solute=tower();self.solvent=tower()
  self.head=nn.Sequential(nn.Linear(256+(0 if thermo else 2),256),nn.SiLU(),nn.Dropout(.15),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,2 if thermo else 1))
 def forward(self,x):
  n=(x.shape[1]-2)//2;z=torch.cat([self.solute(x[:,:n]),self.solvent(x[:,n:2*n])],dim=1)
  if self.thermo:
   ab=self.head(z);return ab[:,:1]+ab[:,1:2]*x[:,-1:]
  return self.head(torch.cat([z,x[:,-2:]],dim=1))

def main():
 df,f=load();tr=np.flatnonzero(df.split.eq('train'));va=np.flatnonzero(df.split.eq('validation'));y=df.target.to_numpy(dtype=np.float32)
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('DEVICE',dev,flush=True)
 config={'families':['rdkit_mlp','combined_mlp','rdkit_vanthoff_mlp'],'seeds':[20261106,20261107,20261108],
 'max_epochs':120,'patience':18,'batch_size':512,'learning_rate':.001,'weight_decay':.0001,'dropout':.15,'checkpoint':'best validation MAE, train-only fitting',
 'notes':'Paired 256/128 towers, 256/128 head. Direct T plus inverseT or pair-dependent apparent vanthoff intercept/slope. Input clipping ±8 standardized units. Train-only median/variance/scaling. Not FASTSOLV reproduction: RDKit rather than Mordred, different widths, no Sobolev loss, mole-fraction target and common class split.'}
 (OUT/'mlp_protocol.json').write_text(json.dumps(config,indent=2))
 for kind,thermo,name in [('rdkit',False,'rdkit_mlp'),('combined',False,'combined_mlp'),('rdkit',True,'rdkit_vanthoff_mlp')]:
  final=OUT/name
  if (final/'metrics.json').exists():continue
  X=representations(df,f,kind)
  # Shared role preprocessing preserves matching descriptor layout in both towers.
  n=(X.shape[1]-2)//2;pre=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),VarianceThreshold(),StandardScaler())
  pre.fit(np.vstack([X[tr,:n],X[tr,n:2*n]]));a=pre.transform(X[:,:n]);b=pre.transform(X[:,n:2*n]);tp=StandardScaler().fit(X[tr,-2:]);tt=tp.transform(X[:,-2:])
  if thermo:tt[:,-1]=X[:,-1] # centered inverse T retains exact affine-in-1/T form.
  XX=np.clip(np.column_stack([a,b,tt]),-8,8).astype(np.float32)
  xt=torch.from_numpy(XX).to(dev);ym=float(y[tr].mean());ys=float(y[tr].std());yt=torch.from_numpy(((y-ym)/ys)[:,None]).to(dev)
  final.mkdir(exist_ok=True);joblib.dump({'molecular':pre,'temperature':tp,'target_mean':ym,'target_std':ys,'thermo':thermo,'clip':8},final/'preprocessing.joblib')
  predictions=[];records=[]
  for seed in config['seeds']:
   seedpath=final/f'seed_{seed}';seedpath.mkdir(exist_ok=True)
   if (seedpath/'predictions.csv').exists():
    import pandas as pd
    predictions.append(pd.read_csv(seedpath/'predictions.csv').predicted_log10_x.to_numpy());continue
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
   model=PairNet(a.shape[1],thermo).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
   sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=6)
   best=float('inf');stale=0;beststate=None;history=[];start=time.time();rng=np.random.default_rng(seed)
   for epoch in range(config['max_epochs']):
    model.train();total=0
    for ix in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/config['batch_size']))):
     opt.zero_grad(set_to_none=True);p=model(xt[ix]);loss=nn.functional.mse_loss(p,yt[ix]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ix)
    model.eval()
    with torch.no_grad():vp=torch.cat([model(xt[ix]) for ix in np.array_split(va,max(1,int(np.ceil(len(va)/2048))))]).cpu().numpy().ravel()*ys+ym
    mae=float(np.mean(np.abs(vp-y[va])));sch.step(mae);history.append({'epoch':epoch+1,'train_scaled_MSE':total/len(tr),'validation_MAE':mae})
    if mae<best-1e-5:best=mae;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0;bestepoch=epoch+1
    else:stale+=1
    if epoch%10==0:print(name,seed,'epoch',epoch+1,'val_MAE',mae,'best',best,flush=True)
    if stale>=config['patience']:break
   model.load_state_dict(beststate);model.eval()
   with torch.no_grad():pp=torch.cat([model(xt[ix]) for ix in np.array_split(np.arange(len(df)),int(np.ceil(len(df)/2048)))]).cpu().numpy().ravel()*ys+ym
   torch.save({'state_dict':beststate,'n_molecular_features':a.shape[1],'thermo':thermo,'target_mean':ym,'target_std':ys,'seed':seed},seedpath/'model.pt')
   (seedpath/'history.json').write_text(json.dumps(history,indent=2));rec={'seed':seed,'best_epoch':bestepoch,'validation_MAE':best,'runtime_seconds':time.time()-start};(seedpath/'training.json').write_text(json.dumps(rec,indent=2));records.append(rec)
   save_prediction(df,pp,name+f'/seed_{seed}');predictions.append(pp);print('SEED_DONE',name,rec,flush=True)
  ensemble=np.mean(predictions,axis=0);metrics=save_prediction(df,ensemble,name);(final/'ensemble.json').write_text(json.dumps({'seeds':config['seeds'],'records':records,'uncertainty':'Ensemble spread is uncalibrated; not a prediction interval'},indent=2))
  print('DONE',name,metrics,flush=True);del xt,yt;torch.cuda.empty_cache()
 print('MLP_COMPLETE',flush=True)
if __name__=='__main__':main()
