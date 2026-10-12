"""CPU unit checks for output-score collection; does not load a language model."""
from pathlib import Path
import importlib.util
import json
import math
import torch

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('output_score_monitor',root/'src/generation/output_score_monitor.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
contract=json.loads((root/'configs/output_feature_contract.json').read_text())
names=contract['ordered_features'];steps=contract['monitoring_positions']

class ToyTokenizer:
    def get_vocab(self):return {'0':0,'1':1,'word':2,'+':3,'-':4,'.':5,'[':6,']':7}

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.lm_head=torch.nn.Identity()

tokens=module.build_token_id_sets(ToyTokenizer())
assert tokens['numeric']=={0,1,3,4,5}
collector=module.LogitsSentinelCollector(ToyModel(),ToyTokenizer(),planned_step=8,token_sets=tokens,decode_steps=steps)
z=torch.arange(32,dtype=torch.float32)/8-2
actual=collector._stats(z);prob=torch.softmax(torch.clamp(z,-80,80),dim=0)
ids=torch.topk(z,k=20).indices
expected=dict(rms=torch.sqrt((z*z).mean()).item(),linf=z.abs().max().item(),top1_logit=z.max().item(),
    top2_logit=torch.topk(z,k=2).values[1].item(),top1_top2_margin=.125,
    top20_entropy_proxy=(-prob[ids]*torch.log(prob[ids]+1e-12)).sum().item(),
    numeric_prob_mass=prob[torch.tensor(sorted(tokens['numeric']))].sum().item(),
    numeric_max_logit=z[torch.tensor(sorted(tokens['numeric']))].max().item())
assert len(expected)==8
for key,value in expected.items():assert abs(actual[key]-value)<1e-6,(key,actual[key],value)
collector.register();collector.active=True
for _ in range(max(steps)):collector.model.lm_head(z.reshape(1,1,-1))
row=collector.row();collector.remove()
assert all(name in row and row[name] is not None for name in names)
assert len(names)==31*8==248
collector.reset();collector.register();collector.active=True
collector.model.lm_head(z.reshape(1,1,-1));partial=collector.row();collector.remove()
assert sum(partial[name] is not None for name in names)==8
assert collector._stats(torch.tensor([1.]))['top2_logit']!=collector._stats(torch.tensor([1.]))['top2_logit']
assert collector._stats(torch.tensor([math.nan]))['rms']==0.
print(json.dumps(dict(status='PASS',final_feature_count=248,monitoring_positions=31,statistic_definitions_checked=8,
                     early_termination_missing_slots_checked=True,qwen_loaded=False)))
