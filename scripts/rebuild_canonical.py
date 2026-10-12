"""Rebuild categories and output features from authorized raw execution sources."""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np
import pandas as pd
from reproduce import REPO, frame, load_json

spec=importlib.util.spec_from_file_location('output_rules',REPO/'src/generation/output_rules.py')
rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)

def missing(value):return value is None or (isinstance(value,float) and np.isnan(value)) or str(value).strip()==''
def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:return [json.loads(line) for line in stream if line.strip()]
def source(root,relative):
    result=(root/relative).resolve()
    assert root.resolve() in result.parents and result.is_file(),relative
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources-root',required=True,type=Path,help='Root of lawfully obtained raw files with the manifest layout.')
    ap.add_argument('--output',type=Path,default=REPO/'generated/rebuilt')
    args=ap.parse_args();root=args.sources_root.resolve();out=args.output.resolve()
    assert out!=REPO.resolve() and all((REPO/p).resolve()!=out and (REPO/p).resolve() not in out.parents
        for p in ['data','configs','models','results','scripts','paper','src','docs'])
    refs=frame(REPO/'data/reconstruction_source_manifest.csv')
    metadata=frame(REPO/'data/execution_metadata.csv.gz').set_index('execution_id')
    names=load_json(REPO/'configs/output_feature_contract.json')['ordered_features']
    bank_paths=refs.clean_source.unique();assert len(bank_paths)==1
    bank=jsonl(source(root,bank_paths[0]));clean={}
    for row in bank:clean.setdefault(row['sample_id'],[]).append(row)
    result=[];features=[];missing_event_error_rows=0;clean_replay_errors=[]
    for relative,group in refs.groupby('compact_source',sort=True):
        cp=source(root,relative);rp=source(root,group.raw_feature_source.iloc[0]);ep=source(root,group.event_source.iloc[0])
        needed=['campaign_id','sample_id','predicted_trajectory_json','actual_injection_count','runner_error']
        header=pd.read_csv(cp,nrows=0).columns
        compact=pd.read_csv(cp,usecols=[n for n in needed+names if n in header],low_memory=False,float_precision='round_trip')
        raw=pd.read_csv(rp,usecols=['campaign_id','sample_id']+names,low_memory=False,float_precision='round_trip')
        assert compact.campaign_id.is_unique and raw.campaign_id.is_unique
        compact=compact.set_index('campaign_id');raw=raw.set_index('campaign_id')
        events={}
        for e in jsonl(ep):
            assert e['campaign_id'] not in events
            events[e['campaign_id']]=e
        for ref in group.itertuples(index=False):
            cid=ref.execution_id;sid=ref.sample_id
            assert len(clean[sid])==1 and cid in compact.index and cid in raw.index
            cr=compact.loc[cid];rr=raw.loc[cid];nominal=clean[sid][0];info=metadata.loc[cid]
            assert cr.sample_id==rr.sample_id==sid
            clean_traj=rules.normalize_trajectory(nominal['predicted_trajectory'])
            reference=rules.normalize_trajectory(nominal['reference_trajectory'])
            assert clean_traj is not None and rules.ade(clean_traj,reference)<1.0
            assert rules.self_intersection(clean_traj) is not True
            assert rules.approx_collision_eval(clean_traj,nominal['user_prompt'])[1] is not True
            parsed=None if missing(cr.predicted_trajectory_json) else rules.normalize_trajectory(json.loads(cr.predicted_trajectory_json))
            error=None if 'runner_error' not in cr or missing(cr.runner_error) else cr.runner_error
            event=events.get(cid)
            if event is None:
                assert error;missing_event_error_rows+=1
            else:
                assert event['sample_id']==sid and rules.normalize_trajectory(event.get('predicted_trajectory'))==parsed
                assert rules.parse_trajectory_prefix_array(event.get('generated_text_head',''))[0]==parsed
            a=pd.to_numeric(cr[names],errors='coerce').to_numpy(float)
            b=pd.to_numeric(rr[names],errors='coerce').to_numpy(float)
            assert np.array_equal(a,b,equal_nan=True),cid
            collision=rules.approx_collision_eval(parsed,nominal['user_prompt'])[1]
            si=rules.self_intersection(parsed);valid=parsed is not None
            geometry=valid and collision is not True and si is not True
            count=cr.actual_injection_count;effective=not missing(count) and float(count)>0
            if event is not None and not missing(count):assert event['injector_summary'].get('actual_injection_count')==int(count)
            isclean=info.source_role=='clean_reference';ade=rules.ade(parsed,clean_traj)
            if isclean:
                assert valid and geometry and ade==0.0
                clean_replay_errors.append(ade);category='clean_reference'
            elif error:category='other_unknown'
            elif not effective:category='no_effective_injection'
            elif not valid:category='structural_parsing'
            elif not geometry:category='geometry_related'
            else:category='ade_only' if ade>1.0 else 'benign_fault'
            assert category==info.category,(cid,category,info.category)
            result.append(dict(execution_id=cid,sample_id=sid,scene_token=info.scene_token,partition=info.partition,
                category=category,label=1 if category=='ade_only' else 0 if category in ['benign_fault','clean_reference'] else None,
                structure_valid=valid,geometry_valid=geometry,collision_proxy=collision,self_intersection=si,
                ade_fault_to_matched_clean=ade))
            features.append(dict(execution_id=cid,**dict(zip(names,a))))
    out.mkdir(parents=True,exist_ok=True)
    rebuilt=pd.DataFrame(result);rebuilt_features=pd.DataFrame(features)
    assert len(rebuilt)==8295 and rebuilt.execution_id.is_unique
    assert int(rebuilt.category.eq('ade_only').sum())==235
    reference_features=frame(REPO/'data/output_features_248.csv.gz').set_index('execution_id').loc[rebuilt_features.execution_id]
    assert np.array_equal(rebuilt_features[names].to_numpy(float),reference_features[names].to_numpy(float),equal_nan=True)
    rebuilt.to_csv(out/'canonical_metadata.csv',index=False,float_format='%.17g')
    rebuilt_features.to_csv(out/'output_features_248.csv.gz',index=False,float_format='%.17g')
    report=dict(status='PASS',rows=len(rebuilt),categories=rebuilt.category.value_counts().to_dict(),
                parser_geometry_pairing_and_raw_feature_checks=True,missing_events_with_recorded_runner_error=missing_event_error_rows,
                all_categories_and_features_match_numerical_release=True,
                qualified_nominal_clean_structure_geometry_and_dataset_ADE_checked=True,
                clean_noop_replay_rows=len(clean_replay_errors),
                maximum_clean_noop_ADE_to_matched_clean=max(clean_replay_errors),qwen_executed=False)
    (out/'reconstruction_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
