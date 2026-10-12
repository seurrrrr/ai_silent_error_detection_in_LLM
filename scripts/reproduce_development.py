"""Replay the recorded development folds for fixed LightGBM feature subsets."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from reproduce import REPO, frame, load_json, operating_point

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['search','ablation'])
    ap.add_argument('--candidate',default='G07')
    ap.add_argument('--repetition',type=int,default=0)
    ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--all-folds',action='store_true')
    ap.add_argument('--check-saved',action='store_true')
    ap.add_argument('--output',type=Path,default=REPO/'generated/development')
    args=ap.parse_args();cfg=load_json(REPO/'configs/development_protocol.json')
    tasks=load_json(REPO/('configs/'+args.phase+'_fold_tasks.json'))
    if not args.all_folds:tasks=[t for t in tasks if t['rep']==args.repetition and t['fold']==args.fold]
    assert tasks,'No recorded task matches the requested fold.'
    table=frame(REPO/'data/development_features_311.csv.gz')
    assert table.partition.eq('train').all() and len(table)==3795 and table.execution_id.is_unique
    assert table.execution_id.tolist()==sorted(table.execution_id)
    specs={s['candidate']:s for s in cfg['specs']}
    candidate=args.candidate;family='LightGBM'
    if args.phase=='search':names=specs[candidate]['features']
    else:
        names=specs['G07']['features']
        if candidate.startswith('MINUS_'):
            removed=candidate[len('MINUS_'):]
            group=frame(REPO/'configs/development_feature_group_map.csv').set_index('feature_name')['group']
            assert removed in ['SCORE_MAGNITUDE','TOKEN_COMPETITION','NUMERIC_TOKEN_EVIDENCE']
            names=[n for n in names if group[n]!=removed]
        elif candidate=='RF_SAME_FEATURES':family='RandomForest'
        else:assert candidate=='SELECTED'
    y=table.label.to_numpy(int);x=table[names].to_numpy(np.float32);x[~np.isfinite(x)]=np.nan
    expected=None
    if args.check_saved:
        relative='results/development/search_oof_predictions.csv.gz' if args.phase=='search' else 'results/ablation/postselection_oof_predictions.csv.gz'
        expected=frame(REPO/relative);expected=expected[expected.candidate.eq(candidate)]
        assert len(expected)==37950,'Expected ten complete OOF repetitions.'
    out=args.output.resolve()
    assert out!=REPO.resolve() and all((REPO/p).resolve()!=out and (REPO/p).resolve() not in out.parents
        for p in ['data','configs','models','results','scripts','paper','src','docs'])
    out.mkdir(parents=True,exist_ok=True);reports=[]
    for task in tasks:
        a,b,c=[np.asarray(task[k],int) for k in ['fit','inner_cal','outer_validation']]
        sa,sb=[np.asarray(task[k],int) for k in ['stop_fit_relative','stop_relative']]
        groups=[set(table.iloc[ix].scene_token) for ix in [a[sa],a[sb],b,c]]
        assert all(not groups[i]&groups[j] for i in range(4) for j in range(i))
        weight=lambda ix:float((y[ix]==0).sum()/y[ix].sum())
        if family=='RandomForest':
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=2,max_features='sqrt',
                    class_weight={0:1.,1:weight(a)},random_state=task['seed'],n_jobs=2))
            model.fit(x[a],y[a]);iteration=None
        else:
            params=dict(cfg['common_parameters']);params.update(random_state=task['seed'],scale_pos_weight=weight(a[sa]))
            early=LGBMClassifier(**params)
            early.fit(x[a[sa]],y[a[sa]],eval_set=[(x[a[sb]],y[a[sb]])],eval_metric='binary_logloss',
                callbacks=[early_stopping(80,verbose=False),log_evaluation(0)])
            iteration=int(early.best_iteration_ or 2000)
            params.update(n_estimators=iteration,scale_pos_weight=weight(a))
            model=LGBMClassifier(**params);model.fit(x[a],y[a])
        cs=model.predict_proba(x[b])[:,1];vs=model.predict_proba(x[c])[:,1]
        frozen={p:operating_point(y[b],cs,budget)['threshold'] for p,budget in cfg['profiles'].items()}
        values=table.iloc[c][['execution_id','sample_id','scene_token','label']].copy()
        values['score']=vs;values['candidate']=candidate;values['family']=family
        values['repetition']=task['rep'];values['fold']=task['fold']
        for profile,threshold in frozen.items():
            values['threshold_'+profile]=threshold
            values['prediction_'+profile]=(vs>=threshold).astype(int)
        error=None
        if expected is not None:
            saved=expected[expected.repetition.eq(task['rep']) & expected.fold.eq(task['fold'])].set_index('execution_id').loc[values.execution_id]
            error=float(np.max(np.abs(vs-saved.score.to_numpy())))
            assert error<1e-12,(candidate,task['rep'],task['fold'],error)
            for profile,threshold in frozen.items():
                assert np.max(np.abs(saved['threshold_'+profile].to_numpy()-threshold))<1e-12
        stem='%s_rep_%02d_fold_%s'%(candidate,task['rep'],task['fold'])
        values.to_csv(out/(stem+'.csv'),index=False,float_format='%.17g')
        report=dict(candidate=candidate,phase=args.phase,repetition=task['rep'],fold=task['fold'],
                    features=len(names),best_iteration=iteration,maximum_absolute_score_difference=error,
                    saved_predictions_checked=args.check_saved,global_calibration_or_test_used=False)
        reports.append(report);print(json.dumps(report),flush=True)
    (out/'replay_report.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
