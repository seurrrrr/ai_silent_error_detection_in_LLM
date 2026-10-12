"""Reproduce the final detector from the distributed numerical artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]

def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def frame(p):
    return pd.read_csv(p, float_precision='round_trip', low_memory=False)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):h.update(block)
    return h.hexdigest()

def matrix(y, score, threshold):
    y=np.asarray(y,int); score=np.asarray(score,float)
    assert np.isfinite(score).all()
    p=score>=threshold
    tp=int(((y==1)&p).sum());fn=int(((y==1)&~p).sum())
    fp=int(((y==0)&p).sum());tn=int(((y==0)&~p).sum())
    div=lambda a,b:a/b if b else None
    return dict(TP=tp,FN=fn,FP=fp,TN=tn,Recall=div(tp,tp+fn),Precision=div(tp,tp+fp),
                F1=div(2*tp,2*tp+fn+fp),FPR=div(fp,fp+tn),FNR=div(fn,tp+fn))

def operating_point(y, score, budget):
    y=np.asarray(y,int);score=np.asarray(score,float)
    assert set(np.unique(y))=={0,1} and np.isfinite(score).all()
    order=np.argsort(-score,kind='mergesort');ys,ss=y[order],score[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
    tp=np.r_[0,np.cumsum(ys)[ends]];fp=np.r_[0,ends+1-np.cumsum(ys)[ends]]
    curve=pd.DataFrame(dict(threshold=np.r_[np.nextafter(ss[0],np.inf),ss[ends]],
        TP=tp,FN=y.sum()-tp,FP=fp,TN=(y==0).sum()-fp,
        Recall=tp/y.sum(),FPR=fp/(y==0).sum()))
    return curve.loc[curve.FPR<=budget].sort_values(['Recall','FPR','threshold'],
        ascending=[False,True,False],kind='mergesort').iloc[0].to_dict()

def data():
    metadata=frame(REPO/'data/execution_metadata.csv.gz')
    features=frame(REPO/'data/output_features_248.csv.gz')
    names=(REPO/'models/lightgbm/feature_list.txt').read_text().splitlines()
    assert len(names)==len(set(names))==248
    assert features.execution_id.is_unique and metadata.execution_id.is_unique
    assert features.execution_id.tolist()==metadata.execution_id.tolist()
    scene_map=frame(REPO/'data/splits/sample_to_scene.csv')
    assert scene_map.sample_id.is_unique and len(scene_map)==1195
    mapped=scene_map.set_index('sample_id').scene_token.reindex(metadata.sample_id)
    assert mapped.notna().all() and np.array_equal(mapped.to_numpy(),metadata.scene_token.to_numpy())
    x=features[names].to_numpy(np.float32);x[~np.isfinite(x)]=np.nan
    return metadata,x,names

def write_results(destination, name, value):
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/name
    if isinstance(value,pd.DataFrame):value.to_csv(path,index=False,float_format='%.17g')
    else:path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def validate_tables():
    rows=[]
    for kind,relative in [('binary','results/test/LightGBM_binary_scores.csv.gz'),
                          ('composite','results/composite/test_predictions.csv.gz')]:
        table=frame(REPO/relative)
        cfg=load_json(REPO/'configs/operating_points.json')
        expected=frame(REPO/('results/test/lightgbm_operating_profiles.csv' if kind=='binary'
                            else 'results/composite/operating_profiles.csv'))
        y=table.label.to_numpy(int) if kind=='binary' else table.category.isin(
            ['ade_only','structural_parsing','geometry_related']).to_numpy(int)
        for code,op in cfg['profiles'].items():
            score=table.score.to_numpy(float) if kind=='binary' else np.where(
                table.deterministic_alarm.astype(bool), np.inf, table.score_LightGBM.to_numpy(float))
            pred=score>=op['threshold']
            assert np.array_equal(pred.astype(int),table['prediction_'+code].to_numpy(int))
            m=matrix(y,pred.astype(float),.5)
            if kind=='binary':
                m.update(AUROC=roc_auc_score(y,table.score),AUPRC=average_precision_score(y,table.score))
            reference=expected.loc[expected.profile.eq(code)].iloc[0]
            for key,value in m.items():assert abs(value-float(reference[key]))<1e-12,(kind,code,key)
            rows.append(dict(population=kind,profile=code,threshold=op['threshold'],**m))
    return pd.DataFrame(rows)

def verify(out):
    entries=load_json(REPO/'MANIFEST.json')['files']
    checked=0
    for row in entries:
        path=REPO/row['path']
        assert path.is_file(),row['path']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
        checked+=1
    metadata,x,names=data()
    assert len(metadata)==8295 and int(metadata.category.eq('ade_only').sum())==235
    role_sets={p:set(metadata.loc[metadata.partition.eq(p),'scene_token']) for p in ['train','calibration','test']}
    assert [len(role_sets[p]) for p in ['train','calibration','test']]==[312,104,104]
    assert all(not role_sets[a]&role_sets[b] for a in role_sets for b in role_sets if a<b)
    assert x.shape==(8295,248)
    scores=validate_tables()
    write_results(out,'saved_score_metrics.csv',scores)
    cfg=load_json(REPO/'configs/operating_points.json')
    cal=frame(REPO/'results/calibration/LightGBM/scores.csv.gz')
    for code,budget in load_json(REPO/'configs/detector_protocol.json')['profiles'].items():
        chosen=operating_point(cal.label,cal.score,budget)
        assert chosen['threshold']==cfg['profiles'][code]['threshold'],code
    report=dict(status='PASS',hashed_files=checked,canonical_rows=len(metadata),features=len(names),
                ADE_only_positives=235,scene_overlap=0,samples_verified_against_source_scene_mapping=1195,
                calibration_threshold_rule_verified=True,
                score_metric_profiles_verified=len(scores),qwen_generation_executed=False)
    write_results(out,'verification.json',report)
    print(json.dumps(report))

def score(out):
    metadata,x,names=data()
    # Read through Python so Windows native filename encoding does not affect
    # repositories located under a Unicode path.
    booster=Booster(model_str=(REPO/'models/lightgbm/model.txt').read_text(encoding='utf-8'))
    assert booster.num_feature()==len(names)==248 and booster.num_trees()==446
    comparisons=[]
    for partition,path,col in [('calibration','results/calibration/LightGBM/scores.csv.gz','score'),
                               ('test','results/test/LightGBM_binary_scores.csv.gz','score')]:
        expected=frame(REPO/path)
        index=pd.Series(np.arange(len(metadata)),index=metadata.execution_id).loc[expected.execution_id].to_numpy()
        assert metadata.iloc[index].partition.eq(partition).all()
        values=booster.predict(x[index],num_threads=2)
        error=float(np.max(np.abs(values-expected[col].to_numpy(float))))
        assert error<1e-12,(partition,error)
        comparisons.append(dict(partition=partition,rows=len(expected),maximum_absolute_score_difference=error))
    expected=frame(REPO/'results/composite/test_predictions.csv.gz')
    index=pd.Series(np.arange(len(metadata)),index=metadata.execution_id).loc[expected.execution_id].to_numpy()
    values=booster.predict(x[index],num_threads=2)
    error=float(np.max(np.abs(values-expected.score_LightGBM.to_numpy(float))))
    assert error<1e-12,error
    comparisons.append(dict(partition='composite_test',rows=len(expected),maximum_absolute_score_difference=error))
    report=dict(status='PASS',model='saved_native_LightGBM',features=248,trees=446,comparisons=comparisons,
                model_fit_executed=False,qwen_generation_executed=False)
    write_results(out,'model_score_verification.json',report);print(json.dumps(report))

def refit(out):
    metadata,x,names=data();cfg=load_json(REPO/'configs/detector_protocol.json')
    eligible=metadata.category.isin(['ade_only','benign_fault','clean_reference']) & metadata.partition.eq('train')
    train=metadata[eligible].sort_values('execution_id').reset_index(drop=True)
    indices=pd.Series(np.arange(len(metadata)),index=metadata.execution_id).loc[train.execution_id].to_numpy()
    roles=load_json(REPO/'configs/early_stopping_indices.json');a=np.array(roles['fit']);b=np.array(roles['stopping'])
    assert len(train)==3795 and int(train.label.sum())==145
    assert not set(train.iloc[a].scene_token)&set(train.iloc[b].scene_token)
    y=train.label.to_numpy(int);xx=x[indices]
    params=dict(cfg['lightgbm_parameters']);params['scale_pos_weight']=float((y[a]==0).sum()/y[a].sum())
    initial=LGBMClassifier(**params)
    initial.fit(xx[a],y[a],eval_set=[(xx[b],y[b])],eval_metric='binary_logloss',
                callbacks=[early_stopping(80,verbose=False),log_evaluation(0)])
    iterations=int(initial.best_iteration_)
    params['n_estimators']=iterations;params['scale_pos_weight']=float((y==0).sum()/y.sum())
    fitted=LGBMClassifier(**params);fitted.fit(xx,y)
    assert iterations==446,iterations
    out.mkdir(parents=True,exist_ok=True)
    (out/'refitted_model.txt').write_text(fitted.booster_.model_to_string(),encoding='utf-8')
    expected=Booster(model_str=(REPO/'models/lightgbm/model.txt').read_text(encoding='utf-8'))
    max_error=float(np.max(np.abs(expected.predict(xx,num_threads=2)-fitted.predict_proba(xx)[:,1])))
    assert max_error<1e-12,max_error
    cal=frame(REPO/'results/calibration/LightGBM/scores.csv.gz')
    ci=pd.Series(np.arange(len(metadata)),index=metadata.execution_id).loc[cal.execution_id].to_numpy()
    cs=fitted.predict_proba(x[ci])[:,1]
    frozen=load_json(REPO/'configs/operating_points.json')['profiles']
    for code,budget in cfg['profiles'].items():
        assert operating_point(cal.label,cs,budget)['threshold']==frozen[code]['threshold']
    report=dict(status='PASS',best_iteration=iterations,training_score_max_difference=max_error,
                fitted_training_rows=3795,thresholds_reproduced_from_calibration=True,
                model_written_under_requested_output=True,original_model_replaced=False,
                qwen_generation_executed=False)
    write_results(out,'refit_verification.json',report);print(json.dumps(report))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['verify','score','refit'])
    ap.add_argument('--output',type=Path,default=REPO/'generated')
    args=ap.parse_args();out=args.output.resolve()
    # Reproduction outputs must not overwrite the frozen distributed artifacts.
    assert out!=REPO.resolve()
    reserved=[REPO/p for p in ['data','models','results','configs','scripts','paper','docs','src','licenses','provenance']]
    assert all(out!=p.resolve() and p.resolve() not in out.parents for p in reserved)
    {'verify':verify,'score':score,'refit':refit}[args.stage](out)

if __name__=='__main__':main()
