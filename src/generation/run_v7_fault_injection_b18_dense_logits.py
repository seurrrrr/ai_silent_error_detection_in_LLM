"""Dense output-score monitoring for bit-level fault campaigns.
Packaging adaptation: portable file references; numerical operations retained.
"""
from __future__ import annotations
import argparse
import csv
import gc
import importlib.util
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd
import torch
from transformers import StoppingCriteriaList

def import_b0fixed(path: Path):
    spec = importlib.util.spec_from_file_location('b0fixed', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import b0fixed from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    required = ['load_jsonl', 'load_txt_lines', 'load_tokenizer', 'load_model', 'V7MonitorCollector', 'FaultInjector', 'build_prompt_text', 'StopAfterTrajectoryArray', 'parse_trajectory_prefix_array', 'get_reference_trajectory', 'ade', 'fde', 'approx_collision_eval', 'self_intersection', 'get_user_prompt', 'row_to_deviation_features', 'deviation_cols_from_profile', 'CORE_STATS']
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f'b0fixed missing required symbols: {missing}')
    return mod

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')

def write_header_if_needed(path: Path, cols: List[str], resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume and path.exists() and (path.stat().st_size > 0):
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()

def append_csv_row(path: Path, cols: List[str], row: dict) -> None:
    with path.open('a', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writerow(row)

def read_completed_campaign_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                cid = r.get('campaign_id')
                if cid:
                    done.add(cid)
    except Exception:
        pass
    return done

def safe_col_name(s: str) -> str:
    s = re.sub('[^A-Za-z0-9]+', '_', str(s))
    return re.sub('_+', '_', s).strip('_')

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

class B18FaultInjector:
    """Wrapper around b0fixed.FaultInjector with logit-aware selector support."""

    def __init__(self, b0, model, campaign: dict, token_sets: Dict[str, Set[int]]):
        self.b0 = b0
        self.inner = b0.FaultInjector(model, campaign)
        self.c = campaign
        self.token_sets = token_sets
        self.target = campaign['target_module']

    def register(self):
        self.inner.register()

    def remove(self):
        self.inner.remove()

    def reset(self):
        self.inner.reset()

    @property
    def active(self):
        return self.inner.active

    @active.setter
    def active(self, v):
        self.inner.active = v

    def summary(self):
        return self.inner.summary()

    def patch_selector(self):
        original = self.inner._select_transient_flat_index
        target_lower = str(self.target).lower()
        selector = str(self.c.get('selector_policy', ''))
        token_sets = self.token_sets
        rng = self.inner.rng

        def select(t: torch.Tensor):
            if 'lm_head' not in target_lower and (not selector.startswith('logit_')):
                return original(t)
            if t.ndim < 2:
                return original(t)
            try:
                h = int(t.shape[-1])
                if t.ndim >= 3:
                    row_offset = (0 * int(t.shape[-2]) + (int(t.shape[-2]) - 1)) * h
                    vec = t[0, -1, :].detach().float()
                else:
                    row_offset = 0
                    vec = t.reshape(-1, h)[0].detach().float()
                finite = torch.isfinite(vec)
                if not finite.any():
                    return original(t)
                if selector == 'logit_numeric_topk':
                    numeric_ids = [i for i in token_sets.get('numeric', set()) if 0 <= i < h and bool(finite[i].item())]
                    if numeric_ids:
                        ids_tensor = torch.tensor(numeric_ids, device=vec.device, dtype=torch.long)
                        vals = vec[ids_tensor]
                        k = min(16, int(vals.numel()))
                        top_pos = torch.topk(vals, k=k).indices
                        chosen = int(ids_tensor[top_pos[int(rng.randrange(k))]].item())
                        return (int(row_offset + chosen), f'logit_numeric_topk_k{k}')
                (vals, ids) = torch.topk(vec.masked_fill(~finite, -float('inf')), k=min(8, h))
                k = int(ids.numel())
                if k <= 0:
                    return original(t)
                weights = [0.45, 0.3, 0.1, 0.05, 0.03, 0.03, 0.02, 0.02][:k]
                s = sum(weights)
                r = rng.random() * s
                acc = 0.0
                pos = 0
                for (j, w) in enumerate(weights):
                    acc += w
                    if r <= acc:
                        pos = j
                        break
                chosen = int(ids[pos].item())
                return (int(row_offset + chosen), f'logit_top_margin_rank{pos + 1}_k{k}')
            except Exception:
                return original(t)
        self.inner._select_transient_flat_index = select

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path', required=True)
    ap.add_argument('--adapter_dir', default='')
    ap.add_argument('--clean_bank_jsonl', required=True, type=Path)
    ap.add_argument('--campaign_jsonl', required=True, type=Path)
    ap.add_argument('--monitor_targets', required=True, type=Path)
    ap.add_argument('--profile_stats_csv', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--b0_fixed_py', type=Path, default=Path(__file__).resolve().with_name('run_v7_fault_injection_b0_fixed.py'))
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--decode_steps', default='1,2,4,8,16')
    ap.add_argument('--load_in_4bit', action='store_true')
    ap.add_argument('--local_files_only', action='store_true')
    ap.add_argument('--stop_after_trajectory_array', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    b0 = import_b0fixed(args.b0_fixed_py)
    campaign = b0.load_jsonl(args.campaign_jsonl)
    clean_rows = b0.load_jsonl(args.clean_bank_jsonl)
    sample_by_id = {str(r.get('sample_id')): r for r in clean_rows}
    monitor_paths = b0.load_txt_lines(args.monitor_targets)
    decode_steps = [int(x.strip()) for x in args.decode_steps.split(',') if x.strip()]
    timepoints = ['prefill'] + [f'decode_{s}' for s in decode_steps]
    profile_stats = pd.read_csv(args.profile_stats_csv)
    m_cols = profile_stats['feature'].astype(str).tolist()
    d_cols = b0.deviation_cols_from_profile(profile_stats)
    raw_csv = args.out_dir / 'v7_fault_b18_raw_monitor_features.csv'
    compact_csv = args.out_dir / 'v7_fault_b18_compact_deviation_features.csv'
    events_jsonl = args.out_dir / 'v7_fault_b18_injection_events.jsonl'
    summary_json = args.out_dir / 'v7_fault_b18_summary.json'
    meta_cols = ['campaign_id', 'sample_id', 'sample_split', 'sample_mission_goal', 'sample_stratum_key', 'target_module', 'target_layer_idx', 'target_layer_bin', 'target_family', 'fault_mode', 'planned_phase', 'planned_decode_step', 'decode_step_source', 'bit_policy', 'bit_semantics', 'selector_policy', 'actual_injection_count', 'target_total_hits', 'target_prefill_hits', 'target_decode_hits', 'generated_token_count', 'parse_error', 'predicted_trajectory_json', 'structure_fail', 'ade', 'fde', 'ade_fail', 'proxy_collision_risk', 'self_intersection', 'failure_label', 'ade_only_failure', 'benign_fault', 'monitor_hit_any_frac', 'monitor_prefill_any_frac', 'monitor_decode_any_frac', 'logits_sentinel_lm_head_path', 'logits_sentinel_prefill_hits', 'logits_sentinel_decode_hits', 'logits_sentinel_slots_hit', 'runner_error']
    print('[INFO] loading tokenizer/model...')
    tokenizer = b0.load_tokenizer(args.model_name_or_path, args.adapter_dir, args.local_files_only)
    token_sets = build_token_id_sets(tokenizer)
    model = b0.load_model(args.model_name_or_path, args.adapter_dir, args.load_in_4bit, args.local_files_only)
    dummy_sentinel = LogitsSentinelCollector(model, tokenizer, planned_step=1, token_sets=token_sets, decode_steps=decode_steps)
    s_cols = dummy_sentinel.columns()
    raw_cols = meta_cols + m_cols + s_cols
    compact_cols = meta_cols + d_cols + s_cols
    completed = read_completed_campaign_ids(compact_csv) if args.resume else set()
    write_header_if_needed(raw_csv, raw_cols, args.resume)
    write_header_if_needed(compact_csv, compact_cols, args.resume)
    collector = b0.V7MonitorCollector(model, monitor_paths, decode_steps)
    collector.register()
    stats = {'total_campaign_rows': len(campaign), 'processed_now': 0, 'errors_now': 0, 'missing_target_now': 0, 'no_injection_now': 0, 'failure_now': 0, 'ade_only_failure_now': 0, 'benign_now': 0, 'parse_fail_now': 0, 'logits_sentinel_missing_lm_head': 0, 'start_time': time.time()}
    try:
        for (idx, c) in enumerate(campaign, start=1):
            cid = c['campaign_id']
            if cid in completed:
                continue
            print(f"[{idx}/{len(campaign)}] {cid} {c['target_family']} {c['fault_mode']} step={c.get('planned_decode_step')}")
            sample = sample_by_id.get(str(c.get('sample_id')))
            meta = {k: c.get(k) for k in ['campaign_id', 'sample_id', 'sample_split', 'sample_mission_goal', 'sample_stratum_key', 'target_module', 'target_layer_idx', 'target_layer_bin', 'target_family', 'fault_mode', 'planned_phase', 'planned_decode_step', 'decode_step_source', 'bit_policy', 'bit_semantics', 'selector_policy']}
            meta['runner_error'] = ''
            if sample is None:
                meta['runner_error'] = 'sample_not_found'
                stats['errors_now'] += 1
                append_csv_row(raw_csv, raw_cols, meta)
                append_csv_row(compact_csv, compact_cols, meta)
                continue
            injector = None
            sentinel = None
            try:
                injector = B18FaultInjector(b0, model, c, token_sets)
                injector.patch_selector()
                injector.register()
                sentinel = LogitsSentinelCollector(model, tokenizer, planned_step=int(c.get('planned_decode_step', 1)), token_sets=token_sets, decode_steps=decode_steps)
                sentinel.register()
                if sentinel.lm_head_path is None:
                    stats['logits_sentinel_missing_lm_head'] += 1
                prompt_text = b0.build_prompt_text(tokenizer, sample)
                inputs = tokenizer(prompt_text, return_tensors='pt')
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for (k, v) in inputs.items()}
                prompt_len = int(inputs['input_ids'].shape[1])
                stopping = None
                if args.stop_after_trajectory_array:
                    stopping = StoppingCriteriaList([b0.StopAfterTrajectoryArray(tokenizer, prompt_len)])
                collector.reset()
                injector.reset()
                sentinel.reset()
                collector.active = True
                injector.active = True
                sentinel.active = True
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None, top_p=None, num_beams=1, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True, stopping_criteria=stopping)
                injector.active = False
                collector.active = False
                sentinel.active = False
                gen_ids = outputs[0][prompt_len:]
                suffix = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                generated_text = b0.TRAJ_PREFIX + suffix if hasattr(b0, 'TRAJ_PREFIX') else '{"trajectory": [[' + suffix
                (pred, parse_error) = b0.parse_trajectory_prefix_array(generated_text)
                ref = b0.get_reference_trajectory(sample)
                a = b0.ade(pred, ref)
                f = b0.fde(pred, ref)
                (min_dist, coll) = b0.approx_collision_eval(pred, b0.get_user_prompt(sample))
                si = b0.self_intersection(pred)
                structure_fail = pred is None
                ade_fail = a is None or a > 1.0
                failure = bool(structure_fail or coll is True or si is True or ade_fail)
                ade_only = bool(not structure_fail and coll is not True and (si is not True) and (a is not None) and (a > 1.0))
                benign = not failure
                inj_sum = injector.summary()
                mon_sum = collector.hit_summary()
                raw_feats = collector.row_features(timepoints, b0.CORE_STATS)
                d_feats = b0.row_to_deviation_features(raw_feats, profile_stats, m_cols)
                s_feats = sentinel.row()
                meta.update({'actual_injection_count': inj_sum.get('actual_injection_count'), 'target_total_hits': inj_sum.get('total_hits'), 'target_prefill_hits': inj_sum.get('prefill_hits'), 'target_decode_hits': inj_sum.get('decode_hits'), 'generated_token_count': int(gen_ids.shape[0]), 'parse_error': parse_error, 'predicted_trajectory_json': json.dumps(pred, ensure_ascii=False, default=str) if pred is not None else '', 'structure_fail': structure_fail, 'ade': a, 'fde': f, 'ade_fail': ade_fail, 'proxy_collision_risk': coll, 'self_intersection': si, 'failure_label': failure, 'ade_only_failure': ade_only, 'benign_fault': benign, 'monitor_hit_any_frac': mon_sum.get('hit_any_frac'), 'monitor_prefill_any_frac': mon_sum.get('prefill_any_frac'), 'monitor_decode_any_frac': mon_sum.get('decode_any_frac')})
                meta.update({k: s_feats.get(k) for k in ['logits_sentinel_lm_head_path', 'logits_sentinel_prefill_hits', 'logits_sentinel_decode_hits', 'logits_sentinel_slots_hit']})
                if int(inj_sum.get('actual_injection_count', 0) or 0) <= 0:
                    stats['no_injection_now'] += 1
                if failure:
                    stats['failure_now'] += 1
                if ade_only:
                    stats['ade_only_failure_now'] += 1
                if benign:
                    stats['benign_now'] += 1
                if parse_error:
                    stats['parse_fail_now'] += 1
                raw_row = dict(meta)
                raw_row.update(raw_feats)
                raw_row.update(s_feats)
                compact_row = dict(meta)
                compact_row.update(d_feats)
                compact_row.update(s_feats)
                append_csv_row(raw_csv, raw_cols, raw_row)
                append_csv_row(compact_csv, compact_cols, compact_row)
                append_jsonl(events_jsonl, {'campaign_id': cid, 'sample_id': c.get('sample_id'), 'target_module': c.get('target_module'), 'injector_summary': inj_sum, 'logits_sentinel_summary': {'lm_head_path': sentinel.lm_head_path, 'slots_hit': sorted(sentinel.features.keys()), 'prefill_hits': sentinel.prefill_hits, 'decode_hits': sentinel.decode_hits}, 'predicted_trajectory': pred, 'generated_text_head': generated_text[:1000]})
                stats['processed_now'] += 1
            except Exception as e:
                stats['errors_now'] += 1
                meta['runner_error'] = f'{type(e).__name__}: {e}'
                append_csv_row(raw_csv, raw_cols, meta)
                append_csv_row(compact_csv, compact_cols, meta)
            finally:
                if sentinel is not None:
                    sentinel.active = False
                    sentinel.remove()
                if injector is not None:
                    injector.active = False
                    injector.remove()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        collector.active = False
        collector.remove()
    stats['elapsed_seconds'] = time.time() - stats['start_time']
    stats['outputs'] = {'raw_monitor_features_csv': str(raw_csv), 'compact_deviation_features_csv': str(compact_csv), 'events_jsonl': str(events_jsonl), 'summary_json': str(summary_json)}
    save_json(summary_json, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
if __name__ == '__main__':
    main()
