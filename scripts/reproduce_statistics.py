"""Recompute development summaries and scene-bootstrap uncertainty from saved scores."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from reproduce import REPO, frame, load_json, matrix

METRICS=['AUROC','AUPRC','Recall','Precision','F1','FPR']

def cached(rows,scene_map,threshold):
    y=rows.label.to_numpy(int);s=rows.score.to_numpy(float)
    order=np.argsort(-s,kind='mergesort')
    ends=np.r_[np.flatnonzero(np.diff(s[order])!=0),len(s)-1]
    codes=np.asarray([scene_map[s] for s in rows.scene_token])
    return y,order,ends,s>=threshold,codes

def weighted(cache,counts):
    y,order,ends,pred,codes=cache;w=counts[codes]
    tp,fn,fp,tn=[float(w[m].sum()) for m in [(y==1)&pred,(y==1)&~pred,(y==0)&pred,(y==0)&~pred]]
    p,n=tp+fn,fp+tn;tps=np.cumsum(w[order]*y[order])[ends];fps=np.cumsum(w[order]*(1-y[order]))[ends]
    recall=tps/p if p else np.full(len(tps),np.nan)
    precision=np.divide(tps,tps+fps,out=np.zeros(len(tps)),where=(tps+fps)>0)
    auc=float(np.trapezoid(np.r_[0,recall],np.r_[0,fps/n])) if p and n else np.nan
    ap=float(np.sum(np.diff(np.r_[0,recall])*precision)) if p else np.nan
    div=lambda a,b:a/b if b else np.nan
    return np.array([auc,ap,div(tp,p),div(tp,tp+fp),div(2*tp,2*tp+fp+fn),div(fp,n)])

def summaries(out):
    checked=0;search_rows=[]
    for phase,folder,filename,expected_name in [('search','development','search_oof_predictions.csv.gz','search_repeat_metrics.csv'),
        ('ablation','ablation','postselection_oof_predictions.csv.gz','postselection_repeat_metrics.csv')]:
        table=frame(REPO/('results/'+folder+'/'+filename))
        reference=frame(REPO/('results/'+folder+'/'+expected_name)).set_index(['candidate','repetition','profile'])
        rows=[]
        for (candidate,repeat),g in table.groupby(['candidate','repetition'],sort=True):
            assert len(g)==3795 and g.execution_id.is_unique and g.scene_token.nunique()==312
            auc=roc_auc_score(g.label,g.score);ap=average_precision_score(g.label,g.score)
            for profile in ['PRIMARY_1PCT','MODERATE_2PCT','HIGH_SENSITIVITY_5PCT']:
                m=matrix(g.label,g.score,g['threshold_'+profile].to_numpy());m.update(AUROC=auc,AUPRC=ap)
                expected=reference.loc[(candidate,repeat,profile)]
                for key,value in m.items():
                    if value is None:assert pd.isna(expected[key]),(candidate,repeat,profile,key)
                    else:assert abs(value-float(expected[key]))<1e-12,(candidate,repeat,profile,key)
                checked+=1;rows.append(dict(candidate=candidate,repetition=int(repeat),profile=profile,**m))
            if phase=='search':search_rows.append(dict(candidate=candidate,repetition=int(repeat),AUPRC=ap))
        pd.DataFrame(rows).to_csv(out/(phase+'_repeat_metrics.csv'),index=False,float_format='%.17g')
    average=pd.DataFrame(search_rows).groupby('candidate').AUPRC.mean()
    expected=frame(REPO/'results/development/search_summary.csv').set_index('candidate')
    assert set(average.index)==set(expected.index) and len(average)==31
    for key,value in average.items():assert abs(value-expected.loc[key,'mean_AUPRC'])<1e-12,key
    winner=average.sort_values(ascending=False).index[0];assert winner=='G07'
    return dict(status='PASS',repeat_operating_metrics_checked=checked,candidates=31,selected_candidate=winner,
                selected_mean_development_AUPRC=float(average[winner]),fits_executed=False)

def bootstrap_test(out):
    order=['LightGBM','RandomForest','LogisticRegression','TokenMargin','EntropyProxy']
    cache={};keys=[];scenes=None
    for family in order:
        table=frame(REPO/('results/test/'+family+'_binary_scores.csv.gz'))
        if scenes is None:scenes=sorted(table.scene_token.unique());scene_map={s:i for i,s in enumerate(scenes)}
        points=load_json(REPO/('results/calibration/'+family+'/operating_points.json'))['profiles']
        for code,point in points.items():
            keys.append((family,code));cache[(family,code)]=cached(table,scene_map,point['threshold'])
    rng=np.random.default_rng(20260928);draws=np.empty((1000,len(keys),6))
    for i in range(1000):
        counts=np.bincount(rng.integers(0,len(scenes),len(scenes)),minlength=len(scenes))
        for j,key in enumerate(keys):draws[i,j]=weighted(cache[key],counts)
    expected=frame(REPO/'results/test/scene_bootstrap_ci.csv').set_index(['family','profile','metric'])
    records=[]
    for j,(family,profile) in enumerate(keys):
        for k,metric in enumerate(METRICS):
            values=draws[:,j,k];low,high=np.nanquantile(values,[.025,.975]);e=expected.loc[(family,profile,metric)]
            assert abs(low-e.CI95_low)<1e-12 and abs(high-e.CI95_high)<1e-12,(family,profile,metric)
            records.append(dict(family=family,profile=profile,metric=metric,CI95_low=low,CI95_high=high))
    pd.DataFrame(records).to_csv(out/'scene_bootstrap_ci.csv',index=False,float_format='%.17g')
    np.save(out/'scene_bootstrap.npy',draws)
    return dict(status='PASS',bootstrap_resamples=1000,scene_unit=104,CI_entries_verified=len(records),
                fits_executed=False,predictions_recomputed=False)

def bootstrap_ablation(out):
    table=frame(REPO/'results/ablation/postselection_oof_predictions.csv.gz')
    axes=load_json(REPO/'results/ablation/postselection_bootstrap_axes.json')
    candidates=axes['candidates'];scenes=sorted(table.scene_token.unique());scene_map={s:i for i,s in enumerate(scenes)}
    cache={key:cached(g,scene_map,g.threshold_PRIMARY_1PCT.to_numpy()) for key,g in table.groupby(['candidate','repetition'])}
    rng=np.random.default_rng(axes['seed']);draws=np.empty((axes['repeats'],len(candidates),6))
    for i in range(len(draws)):
        counts=np.bincount(rng.integers(0,len(scenes),len(scenes)),minlength=len(scenes))
        for j,candidate in enumerate(candidates):
            draws[i,j]=np.nanmean([weighted(cache[(candidate,r)],counts) for r in range(10)],axis=0)
        if (i+1)%250==0:print('ABLATION_BOOTSTRAP',i+1,'/',len(draws),flush=True)
    expected=np.load(REPO/'results/ablation/postselection_paired_bootstrap.npy')
    error=float(np.nanmax(np.abs(draws-expected)));assert error<1e-12,error
    np.save(out/'postselection_paired_bootstrap.npy',draws)
    return dict(status='PASS',bootstrap_resamples=len(draws),scene_unit=312,maximum_absolute_difference=error,
                fits_executed=False,predictions_recomputed=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['summaries','bootstrap-test','bootstrap-ablation'])
    ap.add_argument('--output',type=Path,default=REPO/'generated/statistics')
    args=ap.parse_args();out=args.output.resolve()
    assert out!=REPO.resolve() and all((REPO/p).resolve()!=out and (REPO/p).resolve() not in out.parents
        for p in ['data','configs','models','results','scripts','paper','src','docs'])
    out.mkdir(parents=True,exist_ok=True)
    report={'summaries':summaries,'bootstrap-test':bootstrap_test,'bootstrap-ablation':bootstrap_ablation}[args.stage](out)
    (out/(args.stage+'_verification.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
