"""ADV-001 features with unchanged CLASS-001 chemical cohort and split."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from rdkit import Chem,rdBase
from rdkit.Chem import Descriptors,rdFingerprintGenerator
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/advanced_v1';OUT.mkdir(parents=True,exist_ok=True)
raw=pd.read_csv(ROOT/'BigSolDBv2.0.csv');split=pd.read_csv(ROOT/'results/class_v1/split.csv')
df=raw.iloc[split.source_row.to_numpy()-2].reset_index(drop=True)
assert len(df)==len(split)
for c in ['source_row','solute','solvent','pair','split','rho_g_cm3']:df[c]=split[c].to_numpy()
df['target']=np.log10(df['Solubility(mole_fraction)'])
assert all(set(df.loc[df.split==a,'solute']).isdisjoint(set(df.loc[df.split==b,'solute'])) for a,b in [('train','validation'),('train','test'),('validation','test')])
df.to_csv(OUT/'cohort.csv',index=False)
smiles=sorted(set(df.solute)|set(df.solvent));index={s:i for i,s in enumerate(smiles)}
names=[n for n,_ in Descriptors._descList]
features=[];r2=[];r3=[]
fp2=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024,includeChirality=True)
fp3=rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=1024,includeChirality=True)
for s in smiles:
 m=Chem.MolFromSmiles(s);v=Descriptors.CalcMolDescriptors(m,missingVal=np.nan,silent=True)
 features.append([v.get(n,np.nan) for n in names]);r2.append(fp2.GetCountFingerprintAsNumPy(m));r3.append(fp3.GetCountFingerprintAsNumPy(m))
np.savez_compressed(OUT/'molecular_features.npz',rdkit=np.array(features,dtype=np.float32),morgan2=np.array(r2,dtype=np.float32),morgan3=np.array(r3,dtype=np.float32),solute_idx=df.solute.map(index).to_numpy(),solvent_idx=df.solvent.map(index).to_numpy())
chem=df[['source_row','solute','solvent','Temperature_K','target','split']].copy();chem['split']=chem['split'].replace({'validation':'val'});chem['inverse_T']=1000/chem.Temperature_K-1000/298.15
chem.to_csv(OUT/'chemprop_data.csv',index=False)
meta={'run_id':'ADV-001','rdkit':rdBase.rdkitVersion,'n_rdkit_descriptors':len(names),'descriptor_names':names,'molecule_smiles':smiles,
 'representations':['RDKit all registered 2D descriptors','Morgan count radius2 1024 bins, chirality','Morgan count radius3 1024 bins, chirality','RDKit + radius2 counts'],
 'split_sha256':hashlib.sha256((ROOT/'results/class_v1/split.csv').read_bytes()).hexdigest(),
 'notes':'No target-based descriptor selection. Imputation and variance/scaling must be fit on training rows. Molecular descriptors and fingerprints cached per molecule; solute/solvent roles separate. Same CLASS-001 split, not a new final test. New representations tested on the existing development benchmark.'}
(OUT/'feature_manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k not in ['descriptor_names','molecule_smiles']},indent=2))
