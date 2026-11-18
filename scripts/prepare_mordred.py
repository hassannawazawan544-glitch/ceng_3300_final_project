from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from mordred import Calculator,descriptors
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/advanced_v1'
if __name__=='__main__':
 start=time.time();manifest=json.loads((OUT/'feature_manifest.json').read_text());smiles=manifest['molecule_smiles'];calc=Calculator(descriptors,ignore_3D=True)
 d=calc.pandas([Chem.MolFromSmiles(s) for s in smiles],nproc=4,quiet=True);d=d.apply(pd.to_numeric,errors='coerce');a=d.to_numpy(dtype=np.float32);a[~np.isfinite(a)]=np.nan
 np.savez_compressed(OUT/'mordred_features.npz',mordred=a)
 m={'descriptor_names':[str(x) for x in calc.descriptors],'count':a.shape[1],'molecules':a.shape[0],'missing_fraction':float(np.isnan(a).mean()),'runtime_seconds':time.time()-start,'notes':'Mordredcommunity 2D only, independent of target. NaN imputation/constant selection fit on train later. Same molecule ordering as feature_manifest.'}
 (OUT/'mordred_manifest.json').write_text(json.dumps(m,indent=2));print({k:v for k,v in m.items() if k!='descriptor_names'},flush=True)
