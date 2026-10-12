"""Check saved classifier/scalar controls on the distributed numerical features."""
import json
import joblib
import numpy as np
import pandas as pd
from reproduce import REPO, data, frame, load_json, sha

metadata,x,names=data();column_map={n:i for i,n in enumerate(names)}
indices=pd.Series(np.arange(len(metadata)),index=metadata.execution_id)
reports=[]
for family in ['RandomForest','LogisticRegression','TokenMargin','EntropyProxy']:
    folder=REPO/'models/controls'/family
    points=load_json(REPO/('results/calibration/'+family+'/operating_points.json'))
    assert sha(folder/'model.joblib')==points['model_sha256']
    # joblib is loaded only after matching the distributed, trusted artifact checksum.
    bundle=joblib.load(folder/'model.joblib')
    columns=[column_map[n] for n in bundle['features']]
    for partition,relative in [('calibration','results/calibration/'+family+'/scores.csv.gz'),
                               ('test','results/test/'+family+'_binary_scores.csv.gz')]:
        expected=frame(REPO/relative);ix=indices.loc[expected.execution_id].to_numpy()
        z=x[ix][:,columns]
        if family in ['TokenMargin','EntropyProxy']:
            values=z.astype(float);n=np.isfinite(values).sum(axis=1)
            mean=np.divide(np.nansum(values,axis=1),n,out=np.full(len(values),np.nan),where=n>0)
            score=np.where(np.isfinite(mean),mean,bundle['model']['median'])*bundle['model']['direction']
        else:score=bundle['model'].predict_proba(z)[:,1]
        error=float(np.max(np.abs(score-expected.score.to_numpy(float))))
        assert error<1e-12,(family,partition,error)
        reports.append(dict(family=family,partition=partition,rows=len(expected),maximum_absolute_score_difference=error))
print(json.dumps(dict(status='PASS',comparisons=reports,models_fitted=False,qwen_loaded=False)))
