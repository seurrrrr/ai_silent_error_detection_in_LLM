"""Bit-level fault injection and execution monitoring.
Packaging adaptation: portable file references; numerical operations retained.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
import random
import re
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
TRAJ_PREFIX = '{"trajectory": [['
CORE_STATS = ['nan_count', 'inf_count', 'abs_mean', 'rms', 'linf', 'large_abs_ratio', 'top1_abs', 'top16_abs_mean', 'channel_norm_mean', 'channel_norm_max']
DEFAULT_DECODE_STEPS = [1, 2, 4, 8, 16]
EGO_PROXY_RADIUS = 1.2
AGENT_RADIUS = {'adult': 0.8, 'child': 0.7, 'pedestrian': 0.8, 'bicycle': 1.0, 'motorcycle': 1.0, 'animal': 0.8, 'car': 2.0, 'truck': 2.7, 'bus': 3.0, 'trailer': 3.0, 'construction': 2.5, 'construction_vehicle': 2.8, 'barrier': 0.8, 'trafficcone': 0.4, 'traffic_cone': 0.4, 'rigid': 2.3}

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def load_txt_lines(path: Path) -> List[str]:
    with path.open('r', encoding='utf-8') as f:
        return [x.strip() for x in f if x.strip()]

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

def get_system_prompt(row: dict) -> str:
    if row.get('system_prompt'):
        return row['system_prompt']
    return ''

def get_user_prompt(row: dict) -> str:
    if row.get('user_prompt'):
        return row['user_prompt']
    return ''

def get_reference_trajectory(row: dict):
    if row.get('reference_trajectory') is not None:
        return row.get('reference_trajectory')
    if row.get('reference_trajectory_clean') is not None:
        return row.get('reference_trajectory_clean')
    return None

def normalize_trajectory(obj: Any) -> Optional[List[List[float]]]:
    if isinstance(obj, dict) and 'trajectory' in obj:
        obj = obj['trajectory']
    if not isinstance(obj, list) or len(obj) != 6:
        return None
    out = []
    for p in obj:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        try:
            x = float(p[0])
            y = float(p[1])
        except Exception:
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        out.append([x, y])
    return out

def euclid(a, b) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)

def ade(pred, ref) -> Optional[float]:
    pred = normalize_trajectory(pred)
    ref = normalize_trajectory(ref)
    if pred is None or ref is None:
        return None
    return sum((euclid(p, r) for (p, r) in zip(pred, ref))) / len(pred)

def fde(pred, ref) -> Optional[float]:
    pred = normalize_trajectory(pred)
    ref = normalize_trajectory(ref)
    if pred is None or ref is None:
        return None
    return euclid(pred[-1], ref[-1])

def extract_trajectory_array_after_key(text: str) -> Optional[str]:
    if not text:
        return None
    key_pos = text.find('"trajectory"')
    if key_pos < 0:
        key_pos = text.find("'trajectory'")
    if key_pos < 0:
        return None
    start = text.find('[', key_pos)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def parse_trajectory_prefix_array(text: str) -> Tuple[Optional[List[List[float]]], str]:
    arr_text = extract_trajectory_array_after_key(text)
    if arr_text is None:
        return (None, 'no_complete_trajectory_array')
    try:
        arr = json.loads(arr_text)
    except Exception as e:
        return (None, f'trajectory_array_json_error:{type(e).__name__}')
    traj = normalize_trajectory(arr)
    if traj is None:
        return (None, 'trajectory_invalid')
    return (traj, '')

class StopAfterTrajectoryArray(StoppingCriteria):

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0, self.prompt_len:]
        if gen_ids.numel() <= 0:
            return False
        try:
            suffix = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            text = TRAJ_PREFIX + suffix
            (traj, err) = parse_trajectory_prefix_array(text)
            return traj is not None
        except Exception:
            return False

def ccw(A, B, C) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D) -> bool:
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def self_intersection(traj) -> Optional[bool]:
    traj = normalize_trajectory(traj)
    if traj is None:
        return None
    segs = [(traj[i], traj[i + 1]) for i in range(len(traj) - 1)]
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if abs(i - j) <= 1:
                continue
            if segments_intersect(segs[i][0], segs[i][1], segs[j][0], segs[j][1]):
                return True
    return False

def parse_agents_from_user_prompt(user_prompt: str) -> List[dict]:
    agents = []
    pattern = re.compile('-\\s*([A-Za-z_]+)\\s+at\\s*\\(\\s*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)\\s*\\)\\s*,\\s*moving to\\s*\\(\\s*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)\\s*\\)', re.IGNORECASE)
    for m in pattern.finditer(user_prompt or ''):
        name = m.group(1).lower()
        (x1, y1, x2, y2) = map(float, m.groups()[1:])
        agents.append({'name': name, 'current': [x1, y1], 'future': [x2, y2]})
    return agents

def interpolate_agent(agent: dict, step_idx: int, total_steps: int=6) -> List[float]:
    (x1, y1) = agent['current']
    (x2, y2) = agent['future']
    alpha = step_idx / total_steps
    return [x1 + alpha * (x2 - x1), y1 + alpha * (y2 - y1)]

def agent_collision_radius(name: str) -> float:
    return AGENT_RADIUS.get(name.lower(), 1.2)

def approx_collision_eval(traj, user_prompt: str) -> Tuple[Optional[float], Optional[bool]]:
    traj = normalize_trajectory(traj)
    if traj is None:
        return (None, None)
    agents = parse_agents_from_user_prompt(user_prompt)
    if not agents:
        return (None, None)
    min_dist = float('inf')
    collision = False
    for (i, ego_xy) in enumerate(traj, start=1):
        (ex, ey) = ego_xy
        for ag in agents:
            (ax, ay) = interpolate_agent(ag, i, total_steps=6)
            d = math.sqrt((ex - ax) ** 2 + (ey - ay) ** 2)
            min_dist = min(min_dist, d)
            if d < agent_collision_radius(ag['name']) + EGO_PROXY_RADIUS:
                collision = True
    return (None if min_dist == float('inf') else min_dist, collision)

def build_quant_config() -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)

def load_tokenizer(model_name_or_path: str, adapter_dir: str, local_files_only: bool):
    source = adapter_dir if adapter_dir and Path(adapter_dir).exists() else model_name_or_path
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=False, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_model(model_name_or_path: str, adapter_dir: str, load_in_4bit: bool, local_files_only: bool):
    kwargs = dict(trust_remote_code=True, device_map='auto', local_files_only=local_files_only)
    if load_in_4bit:
        kwargs['quantization_config'] = build_quant_config()
    else:
        kwargs['torch_dtype'] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=local_files_only)
    model.eval()
    return model

def build_prompt_text(tokenizer, row: dict) -> str:
    messages = [{'role': 'system', 'content': get_system_prompt(row)}, {'role': 'user', 'content': get_user_prompt(row)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + TRAJ_PREFIX

def safe_col_name(module_path: str) -> str:
    s = re.sub('[^A-Za-z0-9]+', '_', module_path)
    s = re.sub('_+', '_', s).strip('_')
    return s

def first_tensor_from_output(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            t = first_tensor_from_output(item)
            if t is not None:
                return t
    if isinstance(output, dict):
        for v in output.values():
            t = first_tensor_from_output(v)
            if t is not None:
                return t
    return None

def replace_first_tensor(output: Any, new_tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return new_tensor
    if isinstance(output, tuple):
        items = list(output)
        for (i, item) in enumerate(items):
            if first_tensor_from_output(item) is not None:
                items[i] = replace_first_tensor(item, new_tensor)
                return tuple(items)
        return output
    if isinstance(output, list):
        items = list(output)
        for (i, item) in enumerate(items):
            if first_tensor_from_output(item) is not None:
                items[i] = replace_first_tensor(item, new_tensor)
                return items
        return output
    if isinstance(output, dict):
        out = dict(output)
        for (k, v) in output.items():
            if first_tensor_from_output(v) is not None:
                out[k] = replace_first_tensor(v, new_tensor)
                return out
        return output
    return output

def infer_seq_len(t: torch.Tensor) -> int:
    if t.ndim >= 3:
        return int(t.shape[-2])
    if t.ndim == 2:
        return 1
    return 1

def compute_lite_stats(t: torch.Tensor, large_abs_threshold: float=1000.0) -> Dict[str, float]:
    with torch.no_grad():
        x = t.detach()
        if not torch.is_floating_point(x):
            x = x.float()
        else:
            x = x.float()
        nan_mask = torch.isnan(x)
        inf_mask = torch.isinf(x)
        finite_mask = torch.isfinite(x)
        nan_count = float(nan_mask.sum().item())
        inf_count = float(inf_mask.sum().item())
        if finite_mask.any():
            xf = x[finite_mask]
            absx = xf.abs()
            abs_mean = float(absx.mean().item())
            rms = float(torch.sqrt(torch.mean(xf * xf)).item())
            linf = float(absx.max().item())
            large_abs_ratio = float((absx > large_abs_threshold).float().mean().item())
            top1_abs = linf
            k = min(16, int(absx.numel()))
            top16_abs_mean = float(torch.topk(absx.reshape(-1), k=k).values.mean().item()) if k > 0 else math.nan
        else:
            abs_mean = math.nan
            rms = math.nan
            linf = math.nan
            large_abs_ratio = math.nan
            top1_abs = math.nan
            top16_abs_mean = math.nan
        try:
            x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if x0.ndim >= 2:
                h = int(x0.shape[-1])
                mat = x0.reshape(-1, h)
                cn = torch.sqrt(torch.mean(mat * mat, dim=0))
            else:
                cn = x0.abs().reshape(-1)
            channel_norm_mean = float(cn.mean().item()) if cn.numel() else math.nan
            channel_norm_max = float(cn.max().item()) if cn.numel() else math.nan
        except Exception:
            channel_norm_mean = math.nan
            channel_norm_max = math.nan
        return {'nan_count': nan_count, 'inf_count': inf_count, 'abs_mean': abs_mean, 'rms': rms, 'linf': linf, 'large_abs_ratio': large_abs_ratio, 'top1_abs': top1_abs, 'top16_abs_mean': top16_abs_mean, 'channel_norm_mean': channel_norm_mean, 'channel_norm_max': channel_norm_max}

class V7MonitorCollector:

    def __init__(self, model, monitor_paths: List[str], decode_steps: List[int], large_abs_threshold: float=1000.0):
        self.model = model
        self.monitor_paths = monitor_paths
        self.decode_steps = set((int(x) for x in decode_steps))
        self.large_abs_threshold = large_abs_threshold
        self.module_map = dict(model.named_modules())
        self.handles = []
        self.safe_names = {p: safe_col_name(p) for p in monitor_paths}
        missing = [p for p in monitor_paths if p not in self.module_map]
        if missing:
            raise RuntimeError(f'Missing monitor modules: {missing[:10]} total={len(missing)}')
        self.active = False
        self.reset()

    def reset(self):
        self.features = {}
        self.decode_counts = {p: 0 for p in self.monitor_paths}
        self.hit_counts = {p: 0 for p in self.monitor_paths}
        self.prefill_hits = {p: 0 for p in self.monitor_paths}
        self.decode_hits = {p: 0 for p in self.monitor_paths}

    def register(self):
        for path in self.monitor_paths:
            self.handles.append(self.module_map[path].register_forward_hook(self._make_hook(path)))

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _make_hook(self, path: str):

        def hook(module, inputs, output):
            if not self.active:
                return None
            t = first_tensor_from_output(output)
            if t is None:
                return None
            self.hit_counts[path] += 1
            seq_len = infer_seq_len(t)
            if seq_len > 1:
                self.prefill_hits[path] += 1
                if (path, 'prefill') not in self.features:
                    self.features[path, 'prefill'] = compute_lite_stats(t, self.large_abs_threshold)
            else:
                self.decode_counts[path] += 1
                step = self.decode_counts[path]
                self.decode_hits[path] += 1
                if step in self.decode_steps:
                    self.features[path, f'decode_{step}'] = compute_lite_stats(t, self.large_abs_threshold)
            return None
        return hook

    def feature_columns(self, timepoints: List[str], stats: List[str]) -> List[str]:
        out = []
        for path in self.monitor_paths:
            safe = self.safe_names[path]
            for tp in timepoints:
                for st in stats:
                    out.append(f'm__{safe}__{tp}__{st}')
        return out

    def row_features(self, timepoints: List[str], stats: List[str]) -> Dict[str, Any]:
        out = {}
        for path in self.monitor_paths:
            safe = self.safe_names[path]
            for tp in timepoints:
                d = self.features.get((path, tp))
                for st in stats:
                    out[f'm__{safe}__{tp}__{st}'] = None if d is None else d.get(st)
        return out

    def hit_summary(self) -> dict:
        n = len(self.monitor_paths)
        return {'hit_any_frac': sum((1 for p in self.monitor_paths if self.hit_counts[p] > 0)) / max(1, n), 'prefill_any_frac': sum((1 for p in self.monitor_paths if self.prefill_hits[p] > 0)) / max(1, n), 'decode_any_frac': sum((1 for p in self.monitor_paths if self.decode_hits[p] > 0)) / max(1, n)}

def float_to_fp32_bits(x: float) -> int:
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]

def fp32_bits_to_float(u: int) -> float:
    return struct.unpack('<f', struct.pack('<I', int(u) & 4294967295))[0]

def flip_fp32_bit(x: float, bit: int) -> float:
    u = float_to_fp32_bits(x)
    u2 = u ^ 1 << bit
    return fp32_bits_to_float(u2)

def flip_fp16_bit(x: float, bit: int) -> float:
    a = np.array([np.float16(x)], dtype=np.float16)
    u = a.view(np.uint16)[0].item()
    u2 = u ^ 1 << bit
    b = np.array([u2], dtype=np.uint16).view(np.float16)[0]
    return float(b)

def flip_bf16_bit(x: float, bit: int) -> float:
    u32 = float_to_fp32_bits(x)
    bf = u32 >> 16 & 65535
    bf2 = bf ^ 1 << bit
    u32_2 = (bf2 & 65535) << 16
    return fp32_bits_to_float(u32_2)

def choose_native_bit(dtype: torch.dtype, policy: str, rng: random.Random) -> Tuple[str, int, str]:
    if dtype == torch.bfloat16:
        fmt = 'bf16_native'
        if policy == 'native_uniform':
            bit = rng.randint(0, 15)
            return (fmt, bit, 'uniform_0_15')
        bucket = rng.random()
        if bucket < 0.5:
            choices = [15, 14, 13, 12, 11]
            cls = 'sign_or_high_exponent'
        elif bucket < 0.65:
            choices = [10, 9, 8, 7]
            cls = 'low_exponent'
        elif bucket < 0.85:
            choices = [6, 5, 4]
            cls = 'mantissa_high'
        else:
            choices = [3, 2, 1, 0]
            cls = 'mantissa_mid_low'
        return (fmt, rng.choice(choices), cls)
    if dtype == torch.float16:
        fmt = 'fp16_native'
        if policy == 'native_uniform':
            bit = rng.randint(0, 15)
            return (fmt, bit, 'uniform_0_15')
        bucket = rng.random()
        if bucket < 0.5:
            choices = [15, 14, 13, 12]
            cls = 'sign_or_high_exponent'
        elif bucket < 0.65:
            choices = [11, 10]
            cls = 'low_exponent'
        elif bucket < 0.85:
            choices = [9, 8, 7, 6, 5]
            cls = 'mantissa_high'
        else:
            choices = [4, 3, 2, 1, 0]
            cls = 'mantissa_mid_low'
        return (fmt, rng.choice(choices), cls)
    fmt = 'fp32_native'
    if policy == 'native_uniform':
        return (fmt, rng.randint(0, 31), 'uniform_0_31')
    bucket = rng.random()
    if bucket < 0.5:
        choices = [31, 30, 29, 28, 27]
        cls = 'sign_or_high_exponent'
    elif bucket < 0.65:
        choices = [26, 25, 24, 23]
        cls = 'low_exponent'
    elif bucket < 0.85:
        choices = list(range(22, 11, -1))
        cls = 'mantissa_high'
    else:
        choices = list(range(11, -1, -1))
        cls = 'mantissa_mid_low'
    return (fmt, rng.choice(choices), cls)

def flip_value(old_value: float, dtype: torch.dtype, bit_policy: str, rng: random.Random) -> Tuple[float, str, int, str]:
    if bit_policy == 'fp32_cast_bit30':
        return (flip_fp32_bit(old_value, 30), 'fp32_cast', 30, 'fp32_exp_bit30_stress')
    (fmt, bit, bit_class) = choose_native_bit(dtype, bit_policy, rng)
    if fmt == 'bf16_native':
        return (flip_bf16_bit(old_value, bit), fmt, bit, bit_class)
    if fmt == 'fp16_native':
        return (flip_fp16_bit(old_value, bit), fmt, bit, bit_class)
    return (flip_fp32_bit(old_value, bit), fmt, bit, bit_class)

def _fp32_bits(x: float) -> int:
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]

def _fp32_from_bits(u: int) -> float:
    return struct.unpack('<f', struct.pack('<I', int(u) & 4294967295))[0]

def _apply_bit_action_to_uint(u: int, bit: int, action: str):
    mask = 1 << int(bit)
    old_bit = 1 if int(u) & mask else 0
    if action in {'flip', 'fixed_flip'}:
        u2 = int(u) ^ mask
        stuck_value = -1
    elif action == 'stuck_at_0':
        u2 = int(u) & ~mask
        stuck_value = 0
    elif action == 'stuck_at_1':
        u2 = int(u) | mask
        stuck_value = 1
    else:
        raise ValueError(f'Unknown bit action: {action}')
    new_bit = 1 if int(u2) & mask else 0
    changed = bool(int(u2) != int(u))
    return (int(u2), old_bit, new_bit, stuck_value, changed)

def _bits_of_value_by_format(x: float, bit_format: str) -> int:
    if bit_format == 'bf16_native':
        return _fp32_bits(x) >> 16 & 65535
    if bit_format == 'fp16_native':
        arr = np.array([np.float16(x)], dtype=np.float16)
        return int(arr.view(np.uint16)[0].item())
    if bit_format in {'fp32_native', 'fp32_cast'}:
        return int(_fp32_bits(x))
    raise ValueError(f'Unsupported bit_format: {bit_format}')

def _value_from_bits_by_format(u: int, bit_format: str) -> float:
    if bit_format == 'bf16_native':
        return _fp32_from_bits((int(u) & 65535) << 16)
    if bit_format == 'fp16_native':
        arr = np.array([int(u) & 65535], dtype=np.uint16).view(np.float16)
        return float(arr[0])
    if bit_format in {'fp32_native', 'fp32_cast'}:
        return _fp32_from_bits(int(u) & 4294967295)
    raise ValueError(f'Unsupported bit_format: {bit_format}')

def _bit_action_value(old_value: float, bit_format: str, bit_position: int, bit_action: str):
    u = _bits_of_value_by_format(old_value, bit_format)
    (u2, old_bit, new_bit, stuck_value, changed) = _apply_bit_action_to_uint(u, bit_position, bit_action)
    new_value = _value_from_bits_by_format(u2, bit_format)
    return (new_value, old_bit, new_bit, stuck_value, changed)

class FaultInjector:
    """
    Corrected persistent semantics:

    transient:
      - one injection at planned prefill/decode point.

    persistent_whole_run:
      - first eligible hit chooses one fixed logical site + one fixed bit.
      - later eligible hits reuse the same logical site / hidden channel and same bit action.
      - start can be from_prefill or from_decode.
      - bit semantics can be fixed_flip or stuck_at_*.

    This avoids the previous too-strong behavior:
      every forward hit randomly selects a new scalar and flips it.
    """

    def __init__(self, model, campaign: dict):
        self.model = model
        self.c = campaign
        self.module_map = dict(model.named_modules())
        self.target = campaign['target_module']
        if self.target not in self.module_map:
            raise RuntimeError(f'Target module not found: {self.target}')
        self.rng = random.Random(int(campaign.get('seed', 0)))
        self.handle = None
        self.active = False
        self.reset()

    def reset(self):
        self.prefill_hits = 0
        self.decode_hits = 0
        self.total_hits = 0
        self.injection_count = 0
        self.events = []
        self.persistent_site = None
        self.persistent_active = False

    def register(self):
        self.handle = self.module_map[self.target].register_forward_hook(self._hook)

    def remove(self):
        if self.handle is not None:
            try:
                self.handle.remove()
            except Exception:
                pass
            self.handle = None

    def _phase_and_step(self, t: torch.Tensor):
        seq_len = infer_seq_len(t)
        if seq_len > 1:
            self.prefill_hits += 1
            return ('prefill', -1, seq_len)
        self.decode_hits += 1
        return ('decode', self.decode_hits, seq_len)

    def should_inject(self, phase: str, decode_step: int):
        mode = self.c['fault_mode']
        if mode == 'transient_prefill':
            return phase == 'prefill' and self.injection_count == 0
        if mode in {'transient_decode', 'reserve_late_decode', 'fp32_cast_bit30_stress'}:
            planned = int(self.c.get('planned_decode_step', -1))
            return phase == 'decode' and decode_step == planned and (self.injection_count == 0)
        if mode == 'persistent_whole_run':
            start_policy = self.c.get('persistent_start_policy', 'from_prefill')
            if start_policy == 'from_prefill':
                return phase in {'prefill', 'decode'}
            if start_policy == 'from_decode':
                start_step = int(self.c.get('persistent_start_decode_step', 1))
                if phase == 'decode' and decode_step >= start_step:
                    return True
                return False
            return phase in {'prefill', 'decode'}
        return False

    def _choose_bit_plan(self, dtype: torch.dtype):
        bit_policy = self.c.get('bit_policy', 'native_uniform')
        if bit_policy == 'fp32_cast_bit30':
            return ('fp32_cast', 30, 'fp32_exp_bit30_stress')
        (bit_format, bit_position, bit_class) = choose_native_bit(dtype, bit_policy, self.rng)
        return (bit_format, bit_position, bit_class)

    def _bit_action_for_current_injection(self):
        sem = self.c.get('bit_semantics', 'flip')
        if sem == 'stuck_at_random':
            return 'stuck_at_1' if self.rng.random() < 0.5 else 'stuck_at_0'
        if sem in {'stuck_at_0', 'stuck_at_1'}:
            return sem
        if sem in {'fixed_flip', 'flip'}:
            return 'fixed_flip'
        return 'fixed_flip'

    def _select_transient_flat_index(self, t: torch.Tensor):
        flat = t.reshape(-1)
        with torch.no_grad():
            finite = torch.isfinite(flat)
            idxs = torch.nonzero(finite, as_tuple=False).reshape(-1)
            if idxs.numel() == 0:
                return (None, 'no_finite_scalar')
            selector = self.c.get('selector_policy', 'random_finite_scalar')
            if selector == 'topk16_high_activation':
                vals = flat[idxs].detach().float().abs()
                k = min(16, int(vals.numel()))
                top_pos = torch.topk(vals, k=k).indices
                chosen_pos = int(self.rng.randrange(k))
                return (int(idxs[top_pos[chosen_pos]].item()), f'topk16_k{k}')
            chosen_pos = int(self.rng.randrange(int(idxs.numel())))
            return (int(idxs[chosen_pos].item()), 'random_finite')

    def _flat_idx_from_hidden_idx(self, t: torch.Tensor, hidden_idx: int):
        if t.ndim >= 3:
            h = int(t.shape[-1])
            if h <= 0:
                return None
            hi = int(hidden_idx) % h
            return int((0 * int(t.shape[-2]) + (int(t.shape[-2]) - 1)) * h + hi)
        if t.ndim == 2:
            h = int(t.shape[-1])
            if h <= 0:
                return None
            return int(int(hidden_idx) % h)
        numel = int(t.numel())
        if numel <= 0:
            return None
        return int(int(hidden_idx) % numel)

    def _select_persistent_site_first_time(self, t: torch.Tensor):
        """
        Choose a fixed logical site.

        For 3D activation [B, T, H], persistent site is represented by hidden_idx.
        Later prefill/decode hits use the same hidden channel at the last token.
        This is much more stable than reusing a raw flat index across shape changes.
        """
        selector = self.c.get('selector_policy', 'random_finite_scalar')
        with torch.no_grad():
            if t.ndim >= 2:
                h = int(t.shape[-1])
                if h <= 0:
                    return (None, 'invalid_hidden_dim')
                if t.ndim >= 3:
                    vec = t[0, -1, :]
                else:
                    vec = t[0, :]
                finite = torch.isfinite(vec)
                idxs = torch.nonzero(finite, as_tuple=False).reshape(-1)
                if idxs.numel() == 0:
                    return (None, 'no_finite_hidden_scalar')
                if selector == 'topk16_high_activation':
                    vals = vec[idxs].detach().float().abs()
                    k = min(16, int(vals.numel()))
                    top_pos = torch.topk(vals, k=k).indices
                    chosen_pos = int(self.rng.randrange(k))
                    hidden_idx = int(idxs[top_pos[chosen_pos]].item())
                    detail = f'persistent_hidden_topk16_k{k}'
                else:
                    chosen_pos = int(self.rng.randrange(int(idxs.numel())))
                    hidden_idx = int(idxs[chosen_pos].item())
                    detail = 'persistent_hidden_random_finite'
                (bit_format, bit_position, bit_class) = self._choose_bit_plan(t.dtype)
                bit_action = self._bit_action_for_current_injection()
                return ({'site_kind': 'hidden_channel', 'hidden_idx': hidden_idx, 'bit_format': bit_format, 'bit_position': int(bit_position), 'bit_class': bit_class, 'bit_action': bit_action, 'selector_detail': detail}, detail)
            (flat_idx, detail) = self._select_transient_flat_index(t)
            if flat_idx is None:
                return (None, detail)
            (bit_format, bit_position, bit_class) = self._choose_bit_plan(t.dtype)
            bit_action = self._bit_action_for_current_injection()
            return ({'site_kind': 'flat_idx', 'flat_idx': int(flat_idx), 'bit_format': bit_format, 'bit_position': int(bit_position), 'bit_class': bit_class, 'bit_action': bit_action, 'selector_detail': detail}, detail)

    def _get_persistent_flat_idx(self, t: torch.Tensor):
        if self.persistent_site is None:
            return (None, 'persistent_site_not_initialized')
        if self.persistent_site['site_kind'] == 'hidden_channel':
            idx = self._flat_idx_from_hidden_idx(t, self.persistent_site['hidden_idx'])
            if idx is None or idx >= int(t.numel()):
                return (None, 'persistent_hidden_site_shape_mismatch')
            return (idx, 'persistent_hidden_channel_reused')
        idx = int(self.persistent_site['flat_idx'])
        if idx >= int(t.numel()):
            return (None, 'persistent_flat_site_shape_mismatch')
        return (idx, 'persistent_flat_idx_reused')

    def _apply_fault_at_flat_idx(self, t: torch.Tensor, flat_idx: int, bit_format: str, bit_position: int, bit_class: str, bit_action: str, phase: str, decode_step: int, selector_detail: str):
        t2 = t.clone()
        flat = t2.reshape(-1)
        old_value = float(flat[flat_idx].detach().float().cpu().item())
        try:
            (new_value, old_bit, new_bit, stuck_value, changed) = _bit_action_value(old_value=old_value, bit_format=bit_format, bit_position=int(bit_position), bit_action=bit_action)
            flat[flat_idx] = torch.tensor(new_value, dtype=t.dtype, device=t.device)
        except Exception as e:
            new_value = float('nan')
            old_bit = -1
            new_bit = -1
            stuck_value = -1
            changed = True
            flat[flat_idx] = torch.tensor(float('nan'), dtype=t.dtype, device=t.device)
        self.injection_count += 1
        self.events.append({'event': 'injected', 'injection_index': self.injection_count, 'target_module': self.target, 'fault_mode': self.c.get('fault_mode'), 'persistent_start_policy': self.c.get('persistent_start_policy'), 'persistent_start_decode_step': self.c.get('persistent_start_decode_step'), 'bit_semantics': self.c.get('bit_semantics', 'flip'), 'phase': phase, 'decode_step': decode_step, 'tensor_dtype': str(t.dtype), 'tensor_shape': list(t.shape), 'flat_idx': int(flat_idx), 'selector_policy': self.c.get('selector_policy'), 'selector_detail': selector_detail, 'bit_policy': self.c.get('bit_policy'), 'bit_format': bit_format, 'bit_position': int(bit_position), 'bit_class': bit_class, 'bit_action': bit_action, 'old_bit': int(old_bit), 'new_bit': int(new_bit), 'stuck_value': int(stuck_value), 'changed': bool(changed), 'old_value': old_value, 'new_value': new_value, 'old_isfinite': math.isfinite(old_value), 'new_isfinite': math.isfinite(float(new_value)) if isinstance(new_value, (float, int)) else False})
        return t2

    def inject_tensor(self, t: torch.Tensor, phase: str, decode_step: int):
        mode = self.c['fault_mode']
        if mode == 'persistent_whole_run':
            if self.persistent_site is None:
                (site, detail) = self._select_persistent_site_first_time(t)
                if site is None:
                    self.events.append({'event': 'injection_skipped', 'reason': detail, 'phase': phase, 'decode_step': decode_step})
                    return t
                self.persistent_site = site
            (flat_idx, detail) = self._get_persistent_flat_idx(t)
            if flat_idx is None:
                self.events.append({'event': 'injection_skipped', 'reason': detail, 'phase': phase, 'decode_step': decode_step, 'persistent_site': self.persistent_site})
                return t
            return self._apply_fault_at_flat_idx(t=t, flat_idx=flat_idx, bit_format=self.persistent_site['bit_format'], bit_position=self.persistent_site['bit_position'], bit_class=self.persistent_site['bit_class'], bit_action=self.persistent_site['bit_action'], phase=phase, decode_step=decode_step, selector_detail=self.persistent_site.get('selector_detail', detail))
        (flat_idx, selector_detail) = self._select_transient_flat_index(t)
        if flat_idx is None:
            self.events.append({'event': 'injection_skipped', 'reason': selector_detail, 'phase': phase, 'decode_step': decode_step})
            return t
        if self.c.get('bit_policy') == 'fp32_cast_bit30':
            (bit_format, bit_position, bit_class) = ('fp32_cast', 30, 'fp32_exp_bit30_stress')
        else:
            (bit_format, bit_position, bit_class) = self._choose_bit_plan(t.dtype)
        return self._apply_fault_at_flat_idx(t=t, flat_idx=flat_idx, bit_format=bit_format, bit_position=bit_position, bit_class=bit_class, bit_action='fixed_flip', phase=phase, decode_step=decode_step, selector_detail=selector_detail)

    def _hook(self, module, inputs, output):
        if not self.active:
            return None
        t = first_tensor_from_output(output)
        if t is None:
            return None
        self.total_hits += 1
        (phase, decode_step, seq_len) = self._phase_and_step(t)
        if not self.should_inject(phase, decode_step):
            return None
        t2 = self.inject_tensor(t, phase=phase, decode_step=decode_step)
        return replace_first_tensor(output, t2)

    def summary(self) -> dict:
        return {'target_module': self.target, 'total_hits': self.total_hits, 'prefill_hits': self.prefill_hits, 'decode_hits': self.decode_hits, 'actual_injection_count': self.injection_count, 'persistent_site': self.persistent_site, 'first_events': self.events[:20]}

def safe_label(s: str) -> str:
    s = re.sub('[^A-Za-z0-9]+', '_', str(s))
    s = re.sub('_+', '_', s).strip('_')
    return s

def build_group_index(profile_stats: pd.DataFrame) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[int]] = {'global__all': []}
    for (i, r) in profile_stats.iterrows():
        groups['global__all'].append(i)
        tp = safe_label(r['timepoint'])
        stat = safe_label(r['stat'])
        family = safe_label(r['family'])
        layer = int(r['layer_idx'])
        keys = [f'timepoint__{tp}', f'stat__{stat}', f'family__{family}', f'timepoint_stat__{tp}__{stat}', f'timepoint_family__{tp}__{family}']
        if layer >= 0:
            keys.extend([f'layer__{layer:02d}', f'layer_timepoint__{layer:02d}__{tp}', f'layer_family__{layer:02d}__{family}'])
        for k in keys:
            groups.setdefault(k, []).append(i)
    return {k: np.array(v, dtype=np.int64) for (k, v) in groups.items() if len(v) > 0}

def deviation_cols_from_profile(profile_stats: pd.DataFrame) -> List[str]:
    cols = ['d__global__top1_abs_z', 'd__global__top5_mean_abs_z', 'd__global__top20_mean_abs_z', 'd__global__top100_mean_abs_z']
    groups = build_group_index(profile_stats)
    for gname in groups:
        sg = safe_label(gname)
        for metric in ['max_abs_z', 'mean_abs_z', 'p95_abs_z', 'frac_gt3', 'frac_gt5', 'frac_gt8']:
            cols.append(f'd__{sg}__{metric}')
    return cols

def row_to_deviation_features(raw_feats: Dict[str, Any], profile_stats: pd.DataFrame, m_cols: List[str]) -> Dict[str, Any]:
    x = np.array([float(raw_feats.get(c, 0.0)) if raw_feats.get(c) is not None else 0.0 for c in m_cols], dtype=np.float32)
    med = profile_stats['median'].to_numpy(dtype=np.float32)
    scale = profile_stats['scale'].to_numpy(dtype=np.float32)
    scale[~np.isfinite(scale)] = 1.0
    scale[scale == 0] = 1.0
    z = np.abs((x - med) / scale)
    z[~np.isfinite(z)] = 0.0
    z = np.clip(z, 0.0, 1000000.0)
    out = {}
    z_sorted = np.sort(z)
    out['d__global__top1_abs_z'] = float(z_sorted[-1])
    out['d__global__top5_mean_abs_z'] = float(z_sorted[-min(5, len(z)):].mean())
    out['d__global__top20_mean_abs_z'] = float(z_sorted[-min(20, len(z)):].mean())
    out['d__global__top100_mean_abs_z'] = float(z_sorted[-min(100, len(z)):].mean())
    groups = build_group_index(profile_stats)
    for (gname, idx) in groups.items():
        sub = z[idx]
        sg = safe_label(gname)
        out[f'd__{sg}__max_abs_z'] = float(np.nanmax(sub))
        out[f'd__{sg}__mean_abs_z'] = float(np.nanmean(sub))
        out[f'd__{sg}__p95_abs_z'] = float(np.nanquantile(sub, 0.95))
        out[f'd__{sg}__frac_gt3'] = float(np.nanmean(sub > 3.0))
        out[f'd__{sg}__frac_gt5'] = float(np.nanmean(sub > 5.0))
        out[f'd__{sg}__frac_gt8'] = float(np.nanmean(sub > 8.0))
    return out

def generate_one(model, tokenizer, sample: dict, collector: V7MonitorCollector, injector: FaultInjector, max_new_tokens: int, stop_after_trajectory_array: bool):
    prompt_text = build_prompt_text(tokenizer, sample)
    inputs = tokenizer(prompt_text, return_tensors='pt')
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for (k, v) in inputs.items()}
    prompt_len = int(inputs['input_ids'].shape[1])
    stopping = None
    if stop_after_trajectory_array:
        stopping = StoppingCriteriaList([StopAfterTrajectoryArray(tokenizer, prompt_len)])
    collector.reset()
    injector.reset()
    collector.active = True
    injector.active = True
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, num_beams=1, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True, stopping_criteria=stopping)
    injector.active = False
    collector.active = False
    gen_ids = outputs[0][prompt_len:]
    suffix = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    text = TRAJ_PREFIX + suffix
    (traj, parse_error) = parse_trajectory_prefix_array(text)
    return {'generated_text': text, 'generated_token_count': int(gen_ids.shape[0]), 'predicted_trajectory': traj, 'parse_error': parse_error, 'injector_summary': injector.summary(), 'monitor_summary': collector.hit_summary()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path', required=True)
    ap.add_argument('--adapter_dir', default='')
    ap.add_argument('--clean_bank_jsonl', required=True, type=Path)
    ap.add_argument('--campaign_jsonl', required=True, type=Path)
    ap.add_argument('--monitor_targets', required=True, type=Path)
    ap.add_argument('--profile_stats_csv', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--decode_steps', default='1,2,4,8,16')
    ap.add_argument('--load_in_4bit', action='store_true')
    ap.add_argument('--local_files_only', action='store_true')
    ap.add_argument('--stop_after_trajectory_array', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    campaign = load_jsonl(args.campaign_jsonl)
    clean_rows = load_jsonl(args.clean_bank_jsonl)
    sample_by_id = {str(r.get('sample_id')): r for r in clean_rows}
    monitor_paths = load_txt_lines(args.monitor_targets)
    decode_steps = [int(x.strip()) for x in args.decode_steps.split(',') if x.strip()]
    timepoints = ['prefill'] + [f'decode_{s}' for s in decode_steps]
    profile_stats = pd.read_csv(args.profile_stats_csv)
    m_cols = profile_stats['feature'].astype(str).tolist()
    d_cols = deviation_cols_from_profile(profile_stats)
    raw_csv = args.out_dir / 'v7_fault_b0_raw_monitor_features.csv'
    compact_csv = args.out_dir / 'v7_fault_b0_compact_deviation_features.csv'
    events_jsonl = args.out_dir / 'v7_fault_b0_injection_events.jsonl'
    summary_json = args.out_dir / 'v7_fault_b0_summary.json'
    meta_cols = ['campaign_id', 'sample_id', 'sample_split', 'sample_mission_goal', 'sample_stratum_key', 'target_module', 'target_layer_idx', 'target_layer_bin', 'target_family', 'fault_mode', 'planned_phase', 'planned_decode_step', 'decode_step_source', 'bit_policy', 'selector_policy', 'actual_injection_count', 'target_total_hits', 'target_prefill_hits', 'target_decode_hits', 'generated_token_count', 'parse_error', 'structure_fail', 'ade', 'fde', 'ade_fail', 'proxy_collision_risk', 'self_intersection', 'failure_label', 'ade_only_failure', 'benign_fault', 'monitor_hit_any_frac', 'monitor_prefill_any_frac', 'monitor_decode_any_frac', 'runner_error']
    raw_cols = meta_cols + m_cols
    compact_cols = meta_cols + d_cols
    completed = read_completed_campaign_ids(compact_csv) if args.resume else set()
    write_header_if_needed(raw_csv, raw_cols, args.resume)
    write_header_if_needed(compact_csv, compact_cols, args.resume)
    print('[INFO] loading tokenizer/model...')
    tokenizer = load_tokenizer(args.model_name_or_path, args.adapter_dir, args.local_files_only)
    model = load_model(args.model_name_or_path, args.adapter_dir, args.load_in_4bit, args.local_files_only)
    collector = V7MonitorCollector(model, monitor_paths, decode_steps)
    collector.register()
    module_map = dict(model.named_modules())
    stats = {'total_campaign_rows': len(campaign), 'processed_now': 0, 'errors_now': 0, 'missing_target_now': 0, 'no_injection_now': 0, 'failure_now': 0, 'ade_only_failure_now': 0, 'benign_now': 0, 'parse_fail_now': 0, 'start_time': time.time()}
    try:
        for (idx, c) in enumerate(campaign, start=1):
            cid = c['campaign_id']
            if cid in completed:
                continue
            print(f"[{idx}/{len(campaign)}] {cid} {c['fault_mode']} {c['target_module']}")
            sample = sample_by_id.get(str(c['sample_id']))
            meta = {k: c.get(k) for k in ['campaign_id', 'sample_id', 'sample_split', 'sample_mission_goal', 'sample_stratum_key', 'target_module', 'target_layer_idx', 'target_layer_bin', 'target_family', 'fault_mode', 'planned_phase', 'planned_decode_step', 'decode_step_source', 'bit_policy', 'selector_policy']}
            meta['runner_error'] = ''
            if sample is None:
                meta['runner_error'] = 'sample_not_found'
                stats['errors_now'] += 1
                append_csv_row(raw_csv, raw_cols, meta)
                append_csv_row(compact_csv, compact_cols, meta)
                continue
            if c['target_module'] not in module_map:
                meta['runner_error'] = 'target_module_not_found'
                stats['errors_now'] += 1
                stats['missing_target_now'] += 1
                append_csv_row(raw_csv, raw_cols, meta)
                append_csv_row(compact_csv, compact_cols, meta)
                continue
            injector = None
            try:
                injector = FaultInjector(model, c)
                injector.register()
                res = generate_one(model=model, tokenizer=tokenizer, sample=sample, collector=collector, injector=injector, max_new_tokens=args.max_new_tokens, stop_after_trajectory_array=args.stop_after_trajectory_array)
                raw_feats = collector.row_features(timepoints, CORE_STATS)
                ref = get_reference_trajectory(sample)
                pred = res['predicted_trajectory']
                a = ade(pred, ref)
                f = fde(pred, ref)
                (min_dist, coll) = approx_collision_eval(pred, get_user_prompt(sample))
                si = self_intersection(pred)
                structure_fail = pred is None
                ade_fail = a is None or a > 1.0
                failure = bool(structure_fail or coll is True or si is True or ade_fail)
                ade_only = bool(not structure_fail and coll is not True and (si is not True) and (a is not None) and (a > 1.0))
                benign = not failure
                inj_sum = res['injector_summary']
                mon_sum = res['monitor_summary']
                meta.update({'actual_injection_count': inj_sum['actual_injection_count'], 'target_total_hits': inj_sum['total_hits'], 'target_prefill_hits': inj_sum['prefill_hits'], 'target_decode_hits': inj_sum['decode_hits'], 'generated_token_count': res['generated_token_count'], 'parse_error': res['parse_error'], 'structure_fail': structure_fail, 'ade': a, 'fde': f, 'ade_fail': ade_fail, 'proxy_collision_risk': coll, 'self_intersection': si, 'failure_label': failure, 'ade_only_failure': ade_only, 'benign_fault': benign, 'monitor_hit_any_frac': mon_sum['hit_any_frac'], 'monitor_prefill_any_frac': mon_sum['prefill_any_frac'], 'monitor_decode_any_frac': mon_sum['decode_any_frac']})
                if inj_sum['actual_injection_count'] <= 0:
                    stats['no_injection_now'] += 1
                if failure:
                    stats['failure_now'] += 1
                if ade_only:
                    stats['ade_only_failure_now'] += 1
                if benign:
                    stats['benign_now'] += 1
                if res['parse_error']:
                    stats['parse_fail_now'] += 1
                raw_row = dict(meta)
                raw_row.update(raw_feats)
                d_feats = row_to_deviation_features(raw_feats, profile_stats, m_cols)
                compact_row = dict(meta)
                compact_row.update(d_feats)
                append_csv_row(raw_csv, raw_cols, raw_row)
                append_csv_row(compact_csv, compact_cols, compact_row)
                append_jsonl(events_jsonl, {'campaign_id': cid, 'sample_id': c['sample_id'], 'target_module': c['target_module'], 'injector_summary': inj_sum, 'generated_text_head': res['generated_text'][:500]})
                stats['processed_now'] += 1
            except Exception as e:
                stats['errors_now'] += 1
                meta['runner_error'] = f'{type(e).__name__}: {e}'
                append_csv_row(raw_csv, raw_cols, meta)
                append_csv_row(compact_csv, compact_cols, meta)
            finally:
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
