"""Reconstruct matched-clean categories from authorized saved trajectory records."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

def truth(value):return str(value).strip().lower() in {'true','1','1.0'}

def category(row,delta):
    if row.source_role=='clean_reference':return 'clean_reference',0.0
    if not truth(row.effective_injection):return 'no_effective_injection',None
    if not truth(row.structure_valid):return 'structural_parsing',None
    if not truth(row.geometry_valid):return 'geometry_related',None
    a=np.asarray(json.loads(row.faulted_trajectory),float)
    b=np.asarray(json.loads(row.clean_model_trajectory),float)
    if a.shape!=(6,2) or b.shape!=(6,2) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Valid trajectories must contain six finite coordinate pairs: '+str(row.execution_id))
    ade=float(np.linalg.norm(a-b,axis=1).mean())
    return ('ade_only' if ade>delta else 'benign_fault'),ade

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--delta-ade',type=float,default=1.)
    args=ap.parse_args();table=pd.read_csv(args.input,low_memory=False,float_precision='round_trip')
    required=['execution_id','source_role','effective_injection','structure_valid','geometry_valid',
              'faulted_trajectory','clean_model_trajectory']
    assert all(c in table for c in required)
    results=[]
    for row in table.itertuples(index=False):
        if hasattr(row,'category') and row.category=='other_unknown':
            label,ade='other_unknown',None
        else:label,ade=category(row,args.delta_ade)
        results.append(dict(execution_id=row.execution_id,category=label,
                            ade_fault_to_matched_clean=ade,delta_ADE_m=args.delta_ade))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(args.output,index=False,float_format='%.17g')
    print(json.dumps(dict(rows=len(results),categories=pd.Series([r['category'] for r in results]).value_counts().to_dict())))

if __name__=='__main__':main()
