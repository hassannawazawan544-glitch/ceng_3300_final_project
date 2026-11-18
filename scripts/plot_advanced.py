from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/advanced_v1';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
table=pd.read_csv(OUT/'comparison.csv');show=table.loc[table.model.ne('class_mean')].copy();show['label']=show.model.str.replace('_',' ')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11,8));ax.barh(show.label,show.MAE_log10_x,color=['#007F7A' if n=='validation_convex_ensemble' else '#2D597C' for n in show.model]);ax.invert_yaxis();ax.set(xlabel='Held-out solute MAE, log10 mole fraction',title='Common cohort and split; order follows validation MAE')
for i,v in enumerate(show.MAE_log10_x):ax.text(v+.006,i,f'{v:.3f}',va='center',fontsize=10)
fig.tight_layout();fig.savefig(FIG/'solubility_comparison.png',dpi=250);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,6));ax.scatter(show.MAE_log10_x,show.shared_recovery_MAE_pp,color='#2D597C',s=40)
labels={'validation_convex_ensemble':'GNN + gradient ensemble','class_xgboost':'Initial XGBoost','chemprop_dmpnn':'D-MPNN','chemprop_dmpnn_rdkit':'D-MPNN + RDKit','mordred_sobolev_mlp':'Mordred + gradients','mordred_mlp':'Mordred MLP','extra_trees_rdkit':'ExtraTrees','combined_mlp':'Morgan + RDKit MLP'}
for _,r in show.iterrows():
 if r.model in {'validation_convex_ensemble','class_xgboost','combined_mlp','extra_trees_rdkit'}:ax.annotate(labels[r.model],(r.MAE_log10_x,r.shared_recovery_MAE_pp),xytext=(5,5),textcoords='offset points',fontsize=9)
ax.set(xlabel='Solubility MAE, log10 mole fraction',ylabel='Equilibrium recovery MAE (percentage points)',title='Lower solubility error does not guarantee lower recovery error')
fig.tight_layout();fig.savefig(FIG/'solubility_vs_recovery.png',dpi=250);plt.close(fig)
sel=show.loc[show.model.isin(labels)].copy();fig,ax=plt.subplots(figsize=(10,6));ax.barh(sel.model.map(labels),sel.top_choice_matches,color='#007F7A');ax.set(xlabel='Exact top recoverable-mass choices out of 53 groups',xlim=(0,53),title='50°C to 25°C; measured endpoints and supported density')
for i,v in enumerate(sel.top_choice_matches):ax.text(v+.5,i,str(v),va='center')
ax.invert_yaxis();fig.tight_layout();fig.savefig(FIG/'solvent_choice_comparison.png',dpi=250);plt.close(fig)
# Persist an explicit portable summary archive, excluding large model weights and full training-row predictions.
import tarfile
patterns=['comparison.csv','comparison.md','evaluation_manifest.json','*_manifest.json','*_protocol.json','paired_bootstrap.csv','test_all_models.csv','decisions_all_models.csv','common_temperature_all_models.csv','recovery_all_models.csv','environment_*.txt','*/metrics.json','*/training.json','*/ensemble.json','*/seed_*/training.json','*/seed_*/TRAINING_COMPLETE','figures/*.png']
files=set()
for p in patterns:files.update(OUT.glob(p))
with tarfile.open(ROOT/'archive'/'advanced_summary.tar','w') as tar:
 for f in sorted(files):
  if f.is_file():tar.add(f,arcname=str(f.relative_to(ROOT)))
print('Figures and portable summary saved; files:',len(files))
