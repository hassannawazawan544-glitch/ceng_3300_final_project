"""FASTSOLV-family adaptation: 1613 Mordred 2D + paired towers;
controlled comparison of value-only and temperature-gradient supervision.
"""
import json,time,random
import numpy as np
import torch
from torch import nn
import joblib
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from advanced_common import *
torch.set_num_threads(4)
class MordredPair(nn.Module):
 def __init__(self,n):
  super().__init__()
  def tower():return nn.Sequential(nn.Linear(n,256),nn.LeakyReLU(.1),nn.Dropout(.2),nn.Linear(256,128),nn.LeakyReLU(.1))
  self.a=tower();self.b=tower();self.head=nn.Sequential(nn.Linear(257,512),nn.LeakyReLU(.1),nn.Dropout(.2),nn.Linear(512,256),nn.LeakyReLU(.1),nn.Linear(256,1))
 def forward(self,d,t):
  n=d.shape[1]//2;return self.head(torch.cat([self.a(d[:,:n]),self.b(d[:,n:]),t],dim=1))

def main():
 df,f=load();y=df.target.to_numpy(dtype=np.float32);tr=np.flatnonzero(df.split.eq('train'));va=np.flatnonzero(df.split.eq('validation'))
 X=representations(df,f,'mordred');n=(X.shape[1]-2)//2
 pre=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),VarianceThreshold(),StandardScaler());pre.fit(np.vstack([X[tr,:n],X[tr,n:2*n]]));ds=np.clip(np.column_stack([pre.transform(X[:,:n]),pre.transform(X[:,n:2*n])]),-3,3).astype(np.float32)
 tp=StandardScaler().fit(df.iloc[tr][['Temperature_K']]);ts=tp.transform(df[['Temperature_K']]).astype(np.float32);ym=float(y[tr].mean());ys=float(y[tr].std());scaled_y=(y-ym)/ys
 # Finite-difference gradients grouped by split/source/pair; no held-out labels used in training.
 grad=np.full(len(df),np.nan,dtype=np.float32)
 for _,g in df.groupby(['split','Source','pair']):
  med=g.groupby('Temperature_K').target.median().sort_index()
  if len(med)<3:continue
  tt=(med.index.to_numpy()-tp.mean_[0])/tp.scale_[0];yy=(med.to_numpy()-ym)/ys;gg=np.gradient(yy,tt)
  for temp,val in zip(med.index,gg):
   if np.isfinite(val) and abs(val)<5:grad[g.index[g.Temperature_K.eq(temp)]]=val
 cfg={'features':'1613 Mordred 2D per molecule before train-only variance filtering','seeds':[20261106,20261107,20261108],'max_epochs':120,'patience':20,'batch':256,'learning_rate':.0003,'weight_decay':.001,'descriptor_clip':3,
 'sobolev_weight':.1,'gradient':'finite differences of median log10(x) within source/pair, >=3 T; scaled by training target/T scales; retain either sign with abs gradient <5; no monotonicity assertion',
 'deviations_from_FASTSOLV':'Smaller 256/128 towers + 512/256 head; 3 seeds instead of 4; target log mole fraction; own frozen solute splits; derivative filter/sign and loss weight differ; no exact literature reproduction claim.'}
 (OUT/'mordred_protocol.json').write_text(json.dumps(cfg,indent=2))
 dev=torch.device('cuda');dt=torch.tensor(ds,device=dev);tt=torch.tensor(ts,device=dev);yt=torch.tensor(scaled_y[:,None],device=dev);gt=torch.tensor(grad[:,None],device=dev)
 for sobolev,name in [(False,'mordred_mlp'),(True,'mordred_sobolev_mlp')]:
  path=OUT/name
  if (path/'metrics.json').exists():continue
  path.mkdir(exist_ok=True);joblib.dump({'descriptors':pre,'temperature':tp,'target_mean':ym,'target_std':ys},path/'preprocessing.joblib');predictions=[]
  for seed in cfg['seeds']:
   sd=path/f'seed_{seed}';sd.mkdir(exist_ok=True)
   if (sd/'predictions.csv').exists():predictions.append(pd.read_csv(sd/'predictions.csv').predicted_log10_x.to_numpy());continue
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
   model=MordredPair(ds.shape[1]//2).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=6,factor=.5)
   best=float('inf');stale=0;hist=[];start=time.time()
   for epoch in range(cfg['max_epochs']):
    model.train()
    for ix in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/cfg['batch']))):
     opt.zero_grad(set_to_none=True);temp=tt[ix].detach().clone().requires_grad_(sobolev);p=model(dt[ix],temp);loss=nn.functional.mse_loss(p,yt[ix])
     if sobolev:
      mask=torch.isfinite(gt[ix]);gp=torch.autograd.grad(p.sum(),temp,create_graph=True)[0]
      if mask.any():loss=loss+.1*nn.functional.mse_loss(gp[mask],gt[ix][mask])
     loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
    model.eval()
    with torch.no_grad():pv=torch.cat([model(dt[ix],tt[ix]) for ix in np.array_split(va,6)]).cpu().numpy().ravel()*ys+ym
    mae=float(np.mean(np.abs(pv-y[va])));scheduler.step(mae);hist.append({'epoch':epoch+1,'validation_MAE':mae})
    if mae<best-1e-5:best=mae;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bestepoch=epoch+1;stale=0
    else:stale+=1
    if epoch%10==0:print(name,seed,epoch+1,mae,'best',best,flush=True)
    if stale>=cfg['patience']:break
   model.load_state_dict(state);model.eval()
   with torch.no_grad():p=torch.cat([model(dt[ix],tt[ix]) for ix in np.array_split(np.arange(len(df)),45)]).cpu().numpy().ravel()*ys+ym
   torch.save({'state_dict':state,'molecular_features':ds.shape[1]//2,'target_mean':ym,'target_std':ys,'seed':seed},sd/'model.pt');(sd/'history.json').write_text(json.dumps(hist,indent=2));(sd/'training.json').write_text(json.dumps({'best_epoch':bestepoch,'validation_MAE':best,'runtime_seconds':time.time()-start,'gradient_training_rows':int(np.isfinite(grad[tr]).sum())},indent=2));save_prediction(df,p,name+f'/seed_{seed}');predictions.append(p)
  metrics=save_prediction(df,np.mean(predictions,axis=0),name);print('DONE',name,metrics,flush=True)
 print('MORDRED_NEURAL_COMPLETE',flush=True)
if __name__=='__main__':main()
