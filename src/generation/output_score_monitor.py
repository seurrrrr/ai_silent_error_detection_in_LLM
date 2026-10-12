"""Output-score monitor retained from the fault-execution runner."""
from __future__ import annotations
import math, re
from typing import Any, Dict, List, Optional, Set
import torch

def build_token_id_sets(tokenizer) -> Dict[str, Set[int]]:
    numeric: Set[int] = set()
    structural: Set[int] = set()
    try:
        vocab = tokenizer.get_vocab()
        for (tok, idx) in vocab.items():
            st = str(tok)
            if re.search('[0-9]', st) or any((x in st for x in ['.', '-', '+'])):
                numeric.add(int(idx))
            if any((x in st for x in ['[', ']', '{', '}', ',', ':', '"'])):
                structural.add(int(idx))
    except Exception:
        for text in list('0123456789.-+'):
            try:
                numeric.update((int(x) for x in tokenizer.encode(text, add_special_tokens=False)))
            except Exception:
                pass
        for text in ['[', ']', '{', '}', ',', ':', '"']:
            try:
                structural.update((int(x) for x in tokenizer.encode(text, add_special_tokens=False)))
            except Exception:
                pass
    return {'numeric': numeric, 'structural': structural}

class LogitsSentinelCollector:

    def __init__(self, model, tokenizer, planned_step: int, token_sets: Dict[str, Set[int]], decode_steps: Optional[List[int]]=None):
        self.model = model
        self.tokenizer = tokenizer
        self.planned_step = int(planned_step)
        self.token_sets = token_sets
        self.decode_steps = sorted({int(x) for x in decode_steps or [1, 2, 4, 8, 16] if int(x) > 0})
        self.module_map = dict(model.named_modules())
        self.lm_head_path = self._find_lm_head_path()
        self.handle = None
        self.active = False
        self.reset()

    def _find_lm_head_path(self) -> Optional[str]:
        candidates = [p for p in self.module_map if p.endswith('lm_head') or p == 'lm_head']
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x.count('.'), len(x)))
        return candidates[0]

    def reset(self):
        self.decode_count = 0
        self.prefill_hits = 0
        self.decode_hits = 0
        self.features: Dict[str, Dict[str, Any]] = {}

    def register(self):
        if self.lm_head_path is None:
            return
        self.handle = self.module_map[self.lm_head_path].register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            try:
                self.handle.remove()
            except Exception:
                pass
            self.handle = None

    def _slots_for_step(self, phase: str, step: int) -> List[str]:
        slots: List[str] = []
        if phase == 'prefill':
            slots.append('prefill')
            return slots
        if step in self.decode_steps:
            slots.append(f'decode_{step}')
        if step == self.planned_step:
            slots.append('inject_step')
        if step == self.planned_step + 1:
            slots.append('after_inject_1')
        if step == self.planned_step + 2:
            slots.append('after_inject_2')
        return slots

    def _stats(self, logits_tensor: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            x = logits_tensor.detach().float().reshape(-1)
            finite = torch.isfinite(x)
            nan_count = int(torch.isnan(x).sum().item())
            inf_count = int(torch.isinf(x).sum().item())
            if finite.any():
                xf = x[finite]
            else:
                xf = torch.zeros(1, device=x.device)
            absx = xf.abs()
            k = min(20, int(xf.numel()))
            (vals, ids) = torch.topk(xf, k=k)
            top1 = float(vals[0].item()) if k >= 1 else math.nan
            top2 = float(vals[1].item()) if k >= 2 else math.nan
            top1_id = int(ids[0].item()) if k >= 1 else -1
            top2_id = int(ids[1].item()) if k >= 2 else -1
            margin = top1 - top2 if k >= 2 else math.nan
            probs = torch.softmax(torch.clamp(x, -80, 80), dim=0)
            entropy_top = float((-probs[ids] * torch.log(probs[ids] + 1e-12)).sum().item()) if k > 0 else math.nan
            numeric_ids = [i for i in self.token_sets.get('numeric', set()) if 0 <= i < x.numel()]
            structural_ids = [i for i in self.token_sets.get('structural', set()) if 0 <= i < x.numel()]
            if numeric_ids:
                ni = torch.tensor(numeric_ids, device=x.device, dtype=torch.long)
                numeric_mass = float(probs[ni].sum().item())
                numeric_max_logit = float(x[ni].max().item())
            else:
                numeric_mass = math.nan
                numeric_max_logit = math.nan
            if structural_ids:
                si = torch.tensor(structural_ids, device=x.device, dtype=torch.long)
                structural_mass = float(probs[si].sum().item())
                structural_max_logit = float(x[si].max().item())
            else:
                structural_mass = math.nan
                structural_max_logit = math.nan
            return {'nan_count': nan_count, 'inf_count': inf_count, 'rms': float(torch.sqrt(torch.mean(xf * xf)).item()), 'linf': float(absx.max().item()), 'top1_logit': top1, 'top2_logit': top2, 'top1_top2_margin': float(margin), 'top1_token_id': top1_id, 'top2_token_id': top2_id, 'top20_entropy_proxy': entropy_top, 'numeric_prob_mass': numeric_mass, 'numeric_max_logit': numeric_max_logit, 'structural_prob_mass': structural_mass, 'structural_max_logit': structural_max_logit}

    def _hook(self, module, inputs, output):
        if not self.active:
            return None
        t = output if isinstance(output, torch.Tensor) else None
        if t is None:
            return None
        try:
            seq_len = int(t.shape[-2]) if t.ndim >= 3 else 1
            if seq_len > 1:
                self.prefill_hits += 1
                phase = 'prefill'
                step = -1
                last_logits = t[0, -1, :]
            else:
                self.decode_count += 1
                self.decode_hits += 1
                phase = 'decode'
                step = self.decode_count
                last_logits = t.reshape(-1, t.shape[-1])[0]
            slots = self._slots_for_step(phase, step)
            if slots:
                stats = self._stats(last_logits)
                for slot in slots:
                    if slot not in self.features:
                        self.features[slot] = dict(stats)
        except Exception:
            return None
        return None

    def columns(self) -> List[str]:
        decode_slots = [f'decode_{s}' for s in self.decode_steps]
        slots = ['prefill'] + decode_slots + ['inject_step', 'after_inject_1', 'after_inject_2']
        stats = ['nan_count', 'inf_count', 'rms', 'linf', 'top1_logit', 'top2_logit', 'top1_top2_margin', 'top1_token_id', 'top2_token_id', 'top20_entropy_proxy', 'numeric_prob_mass', 'numeric_max_logit', 'structural_prob_mass', 'structural_max_logit']
        return [f's__lm_head__{slot}__{st}' for slot in slots for st in stats]

    def row(self) -> Dict[str, Any]:
        out = {}
        for col in self.columns():
            (_, _, slot, st) = col.split('__', 3)
            out[col] = self.features.get(slot, {}).get(st)
        out['logits_sentinel_lm_head_path'] = self.lm_head_path
        out['logits_sentinel_prefill_hits'] = self.prefill_hits
        out['logits_sentinel_decode_hits'] = self.decode_hits
        out['logits_sentinel_slots_hit'] = ','.join(sorted(self.features.keys()))
        return out
