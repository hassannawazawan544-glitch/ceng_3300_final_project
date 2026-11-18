"""Retrain official Chemprop 1.6.1 paired D-MPNN; no pretrained weights.
Same frozen solute split. This is a published-family baseline, not exact FASTSOLV.
"""
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1' # Only this script's own locally trained checkpoints.
import json,subprocess,time,sys
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from advanced_common import *
ROOT=Path(__file__).resolve().parents[1]
def main():
 torch.set_num_threads(4);df,f=load();data=OUT/'gnn_data';data.mkdir(exist_ok=True)
 tr=np.flatnonzero(df.split.eq('train'));X=representations(df,f,'rdkit')
 prep=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),VarianceThreshold());xx=prep.fit_transform(X[tr]);full=prep.transform(X)
 import joblib
 joblib.dump(prep,data/'descriptor_imputer.joblib')
 for split in ['all','train','validation','test']:
  mask=np.ones(len(df),dtype=bool) if split=='all' else df.split.eq(split).to_numpy()
  df.loc[mask,['solute','solvent','target']].to_csv(data/(split+'.csv'),index=False)
  np.save(data/(split+'_temperature.npy'),np.column_stack([df.loc[mask,'Temperature_K'],1000/df.loc[mask,'Temperature_K']-1000/298.15]).astype(np.float32))
  np.save(data/(split+'_rdkit.npy'),full[mask].astype(np.float32))
 protocol={'package':'chemprop 1.6.1 official PyPI','reason':'Working existing GPU environment uses Python 3.10; modern Chemprop requires Python >=3.11. Version difference explicitly recorded.',
  'families':['chemprop_dmpnn','chemprop_dmpnn_rdkit'],'seeds':[20261106,20261107,20261108],'epochs':35,'depth':3,'hidden_size':300,'ffn_hidden_size':300,'ffn_num_layers':2,'batch_size':256,'dropout':.1,
  'split':'Explicit separate train/validation/test paths from CLASS-001; no row random split',
  'criterion':'MSE optimization; checkpoint by validation MAE; test not used for model selection',
  'target':'log10 mole fraction','notes':'No Sobolev loss; not FASTSOLV exact reproduction or DMPNN-MoE; graph-only includes measured temperature/inverseT. Hybrid additionally receives all RDKit 2D descriptors. All custom features scaled on train by Chemprop. All raw targets and density are excluded from molecular input features.'}
 (OUT/'gnn_protocol.json').write_text(json.dumps(protocol,indent=2))
 exe=Path(sys.executable).parent
 for name,features in [('chemprop_dmpnn','temperature'),('chemprop_dmpnn_rdkit','rdkit')]:
  path=OUT/name
  if (path/'metrics.json').exists():continue
  path.mkdir(exist_ok=True);predictions=[];logs=[]
  for seed in protocol['seeds']:
   sd=path/f'seed_{seed}';sd.mkdir(exist_ok=True);checkpoint=sd/'fold_0/model_0/model.pt'
   if not (sd/'TRAINING_COMPLETE').exists():
    args=[str(exe/'chemprop_train'),'--data_path',str(data/'train.csv'),'--separate_val_path',str(data/'validation.csv'),'--separate_test_path',str(data/'test.csv'),'--dataset_type','regression','--smiles_columns','solute','solvent','--target_columns','target','--number_of_molecules','2','--features_path',str(data/f'train_{features}.npy'),'--separate_val_features_path',str(data/f'validation_{features}.npy'),'--separate_test_features_path',str(data/f'test_{features}.npy'),'--save_dir',str(sd),'--epochs','35','--hidden_size','300','--depth','3','--ffn_hidden_size','300','--ffn_num_layers','2','--dropout','0.1','--batch_size','256','--num_workers','0','--cache_cutoff','inf','--aggregation','mean','--metric','mae','--seed',str(seed),'--pytorch_seed',str(seed),'--init_lr','0.0001','--max_lr','0.001','--final_lr','0.0001','--save_preds','--quiet']
    (sd/'command.json').write_text(json.dumps(args,indent=2));start=time.time();print('START',name,seed,flush=True)
    with (sd/'console.log').open('w') as log:subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
    (sd/'TRAINING_COMPLETE').write_text('All planned epochs completed; checkpoint selected by validation MAE.\n')
    print('TRAINED',name,seed,time.time()-start,flush=True)
   predictionfile=sd/'all_predictions.csv'
   if not predictionfile.exists():
    args=[str(exe/'chemprop_predict'),'--test_path',str(data/'all.csv'),'--preds_path',str(predictionfile),'--checkpoint_path',str(checkpoint),'--features_path',str(data/f'all_{features}.npy'),'--smiles_columns','solute','solvent','--number_of_molecules','2','--num_workers','0','--batch_size','256']
    with (sd/'predict.log').open('w') as log:subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
   p=pd.read_csv(predictionfile)['target'].to_numpy(dtype=float);assert len(p)==len(df)
   save_prediction(df,p,name+f'/seed_{seed}');predictions.append(p)
  metrics=save_prediction(df,np.mean(predictions,axis=0),name);print('DONE',name,metrics,flush=True)
 print('GNN_COMPLETE',flush=True)
if __name__=='__main__':main()
