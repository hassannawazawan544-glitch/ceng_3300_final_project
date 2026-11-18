from pathlib import Path
import json,time,joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from advanced_common import *
df,f=load();y=df.target.to_numpy();tr=np.flatnonzero(df.split.eq('train'));va=np.flatnonzero(df.split.eq('validation'))
protocol={'models':['XGB_RDKit','XGB_Morgan2','XGB_Morgan3','XGB_combined','ExtraTrees_RDKit','HistGB_RDKit'],
 'xgb_candidates':[{'max_depth':4,'min_child_weight':5},{'max_depth':6,'min_child_weight':10}],
 'selection':'Validation MAE only; evaluate heldout data after selection. Existing CLASS-001 split reused.',
 'feature_preprocessing':'Training-only median imputation and constant-feature removal; Morgan counts log1p transformed; no fitted transform needed for log1p.',
 'seeds':[20261106,20261107,20261108]}
(OUT/'tabular_protocol.json').write_text(json.dumps(protocol,indent=2));summary=[]
for kind in ['rdkit','morgan2','morgan3','combined']:
 name='xgb_'+kind;path=OUT/name
 if (path/'metrics.json').exists():continue
 start=time.time();X=representations(df,f,kind)
 prep=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),VarianceThreshold())
 xt=prep.fit_transform(X[tr]);xv=prep.transform(X[va]);best=None;records=[]
 for cfg in protocol['xgb_candidates']:
  model=XGBRegressor(n_estimators=800,learning_rate=.04,subsample=.8,colsample_bytree=.8,reg_lambda=10,reg_alpha=.1,tree_method='hist',n_jobs=4,random_state=20261106,early_stopping_rounds=60,**cfg)
  model.fit(xt,y[tr],eval_set=[(xv,y[va])],verbose=False)
  mae=float(mean_absolute_error(y[va],model.predict(xv)));records.append({'config':cfg,'validation_MAE':mae,'n_trees':model.best_iteration+1})
  print(name,records[-1],flush=True)
  if best is None or mae<best[0]:best=(mae,model,cfg)
 path.mkdir(exist_ok=True);joblib.dump(prep,path/'preprocessing.joblib');joblib.dump(best[1],path/'model.joblib')
 pred=best[1].predict(prep.transform(X));metrics=save_prediction(df,pred,name)
 info={'feature_family':kind,'n_features':xt.shape[1],'validation_candidates':records,'selected_config':best[2],'n_trees':best[1].best_iteration+1,'runtime_seconds':time.time()-start}
 (path/'training.json').write_text(json.dumps(info,indent=2));summary.append({'model':name,**metrics['test']});print('DONE',name,metrics,flush=True)
# Complementary descriptor models, fixed settings rather than a large tuning sweep.
X=representations(df,f,'rdkit');prep=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),VarianceThreshold());xt=prep.fit_transform(X[tr]);xx=prep.transform(X)
for name,model in [('extra_trees_rdkit',ExtraTreesRegressor(n_estimators=300,min_samples_leaf=5,max_features=.7,n_jobs=4,random_state=20261106)),('histgb_rdkit',HistGradientBoostingRegressor(max_iter=300,max_leaf_nodes=15,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20261106))]:
 path=OUT/name
 if (path/'metrics.json').exists():continue
 start=time.time();model.fit(xt,y[tr]);path.mkdir(exist_ok=True);joblib.dump(model,path/'model.joblib');joblib.dump(prep,path/'preprocessing.joblib');metrics=save_prediction(df,model.predict(xx),name)
 (path/'training.json').write_text(json.dumps({'settings':str(model),'runtime_seconds':time.time()-start},indent=2));print('DONE',name,metrics,flush=True)
print('TABULAR_COMPLETE',flush=True)
