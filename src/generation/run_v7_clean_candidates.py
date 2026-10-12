"""Generate and qualify clean-reference trajectories.
Packaging adaptation: portable file references; numerical operations retained.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
TRAJ_PREFIX = '{"trajectory": [['
EGO_PROXY_RADIUS = 1.2
AGENT_RADIUS = {'adult': 0.8, 'child': 0.7, 'pedestrian': 0.8, 'bicycle': 1.0, 'motorcycle': 1.0, 'animal': 0.8, 'car': 2.0, 'truck': 2.7, 'bus': 3.0, 'trailer': 3.0, 'construction': 2.5, 'construction_vehicle': 2.8, 'barrier': 0.8, 'trafficcone': 0.4, 'traffic_cone': 0.4, 'rigid': 2.3}

def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + '\n')

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')

def get_system_prompt(row: dict) -> str:
    if row.get('system_prompt'):
        return row['system_prompt']
    for m in row.get('messages', []) or []:
        if m.get('role') == 'system':
            return m.get('content', '')
    return ''

def get_user_prompt(row: dict) -> str:
    if row.get('user_prompt'):
        return row['user_prompt']
    for m in row.get('messages', []) or []:
        if m.get('role') == 'user':
            return m.get('content', '')
    return ''

def get_reference_answer(row: dict) -> str:
    if row.get('reference_answer'):
        return row['reference_answer']
    for m in row.get('messages', []) or []:
        if m.get('role') == 'assistant':
            return m.get('content', '')
    return ''

def get_reference_trajectory(row: dict) -> Optional[list]:
    if row.get('reference_trajectory') is not None:
        return row.get('reference_trajectory')
    if row.get('reference_trajectory_clean') is not None:
        return row.get('reference_trajectory_clean')
    return None

def normalize_trajectory(obj: Any) -> Optional[List[List[float]]]:
    """
    只接受：
      [[x1,y1], ..., [x6,y6]]
    或：
      {"trajectory": [[...]]}

    要求：
      - 正好 6 个点
      - 每个点至少两个数字
      - x/y 有限
    """
    if isinstance(obj, dict) and 'trajectory' in obj:
        obj = obj['trajectory']
    if not isinstance(obj, list) or len(obj) != 6:
        return None
    out: List[List[float]] = []
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

def is_valid_traj(traj: Any) -> bool:
    return normalize_trajectory(traj) is not None

def euclid(p: List[float], q: List[float]) -> float:
    return math.sqrt((float(p[0]) - float(q[0])) ** 2 + (float(p[1]) - float(q[1])) ** 2)

def ade(pred: Any, ref: Any) -> Optional[float]:
    pred_n = normalize_trajectory(pred)
    ref_n = normalize_trajectory(ref)
    if pred_n is None or ref_n is None:
        return None
    return sum((euclid(p, r) for (p, r) in zip(pred_n, ref_n))) / len(pred_n)

def fde(pred: Any, ref: Any) -> Optional[float]:
    pred_n = normalize_trajectory(pred)
    ref_n = normalize_trajectory(ref)
    if pred_n is None or ref_n is None:
        return None
    return euclid(pred_n[-1], ref_n[-1])

def max_step_distance(traj: Any) -> Optional[float]:
    traj_n = normalize_trajectory(traj)
    if traj_n is None:
        return None
    mx = 0.0
    for i in range(1, len(traj_n)):
        mx = max(mx, euclid(traj_n[i], traj_n[i - 1]))
    return mx

def max_step_accel_like(traj: Any) -> Optional[float]:
    traj_n = normalize_trajectory(traj)
    if traj_n is None:
        return None
    steps = [euclid(traj_n[i], traj_n[i - 1]) for i in range(1, len(traj_n))]
    if len(steps) < 2:
        return 0.0
    return max((abs(steps[i] - steps[i - 1]) for i in range(1, len(steps))))

def ccw(A: List[float], B: List[float], C: List[float]) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A: List[float], B: List[float], C: List[float], D: List[float]) -> bool:
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def self_intersection(traj: Any) -> Optional[bool]:
    traj_n = normalize_trajectory(traj)
    if traj_n is None:
        return None
    segs = [(traj_n[i], traj_n[i + 1]) for i in range(len(traj_n) - 1)]
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if abs(i - j) <= 1:
                continue
            if segments_intersect(segs[i][0], segs[i][1], segs[j][0], segs[j][1]):
                return True
    return False

def parse_agents_from_user_prompt(user_prompt: str) -> List[dict]:
    """
    解析形如：
      - car at (x1,y1), moving to (x2,y2).
    """
    agents: List[dict] = []
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

def approx_collision_eval(traj: Any, user_prompt: str) -> Tuple[Optional[float], Optional[bool]]:
    traj_n = normalize_trajectory(traj)
    if traj_n is None:
        return (None, None)
    agents = parse_agents_from_user_prompt(user_prompt)
    if not agents:
        return (None, None)
    min_dist = float('inf')
    collision = False
    for (i, ego_xy) in enumerate(traj_n, start=1):
        (ex, ey) = ego_xy
        for ag in agents:
            (ax, ay) = interpolate_agent(ag, i, total_steps=6)
            d = math.sqrt((ex - ax) ** 2 + (ey - ay) ** 2)
            min_dist = min(min_dist, d)
            if d < agent_collision_radius(ag['name']) + EGO_PROXY_RADIUS:
                collision = True
    if min_dist == float('inf'):
        return (None, collision)
    return (min_dist, collision)

def extract_first_json_object(text: str) -> Optional[str]:
    """
    尝试抽取第一个完整 JSON object。
    这个函数保留作为 debug，不再作为唯一 strict 条件。
    """
    start = text.find('{')
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
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def extract_trajectory_array_after_key(text: str) -> Optional[str]:
    """
    v7 核心解析逻辑：

    只解析 trajectory 字段后面的第一个完整 JSON array。
    这不是任意文本抽数字，而是要求输出里明确出现 "trajectory"，
    并且 trajectory 后面必须有一个完整数组。

    这样可以接受：
      {"trajectory": [[...6点...]], "thoughts": ...  后面没闭合

    因为 trajectory 本身已经完整，后面的 thoughts 不影响轨迹解析。
    """
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

def parse_trajectory_strict(generated_text: str) -> Tuple[Optional[dict], Optional[List[List[float]]], str, str]:
    """
    v7 strict parser：

    - 不再要求整个 JSON object 完整闭合；
    - 但必须存在 trajectory 字段；
    - 只接受 trajectory 字段后面的第一个完整 JSON array；
    - trajectory 必须正好 6 个 [x,y] 有限数字；
    - 不使用“任意文本抽前 12 个数字”的 fallback 作为 strict pass。
    """
    arr_text = extract_trajectory_array_after_key(generated_text or '')
    if arr_text is None:
        return (None, None, 'failed', 'no_complete_trajectory_array')
    try:
        arr = json.loads(arr_text)
    except Exception as e:
        return (None, None, 'failed', f'trajectory_array_json_error:{type(e).__name__}')
    traj = normalize_trajectory(arr)
    if traj is None:
        return ({'trajectory_raw': arr}, None, 'failed', 'trajectory_invalid')
    obj = {'trajectory': traj}
    return (obj, traj, 'trajectory_prefix_array', '')

def parse_trajectory_fallback_for_debug(generated_text: str) -> Optional[List[List[float]]]:
    """
    仅用于 debug，不作为 strict_clean_pass。
    这个函数可以帮助我们判断：模型其实有没有输出点，只是 strict parser 没收。
    """
    bracket_pairs = re.findall('\\[\\s*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)\\s*\\]', generated_text or '')
    if len(bracket_pairs) >= 6:
        return [[float(x), float(y)] for (x, y) in bracket_pairs[:6]]
    paren_pairs = re.findall('\\(\\s*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)\\s*\\)', generated_text or '')
    if len(paren_pairs) >= 6:
        return [[float(x), float(y)] for (x, y) in paren_pairs[:6]]
    return None

def build_quant_config() -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)

def load_tokenizer(model_name_or_path: str, adapter_dir: str, local_files_only: bool):
    source = adapter_dir if adapter_dir and Path(adapter_dir).exists() else model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=False, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

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

def build_prompt_text(tokenizer, row: dict, enable_thinking_false: bool=False) -> str:
    messages = [{'role': 'system', 'content': get_system_prompt(row)}, {'role': 'user', 'content': get_user_prompt(row)}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking_false:
        kwargs['enable_thinking'] = False
    try:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop('enable_thinking', None)
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return text + TRAJ_PREFIX

def infer_one(model, tokenizer, row: dict, max_new_tokens: int, enable_thinking_false: bool=False) -> Dict[str, Any]:
    prompt_text = build_prompt_text(tokenizer, row, enable_thinking_false=enable_thinking_false)
    inputs = tokenizer(prompt_text, return_tensors='pt')
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for (k, v) in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, num_beams=1, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_suffix = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    generated_text = TRAJ_PREFIX + generated_suffix
    return {'prompt_token_count': int(inputs['input_ids'].shape[1]), 'generated_token_count': int(generated_ids.shape[0]), 'generated_text': generated_text, 'generated_suffix': generated_suffix, 'output_token_ids': [int(x) for x in generated_ids.detach().cpu().tolist()], 'output_sha1': hashlib.sha1(generated_text.encode('utf-8')).hexdigest()}

def evaluate_output(row: dict, generated_text: str, ade_threshold: float) -> dict:
    ref = get_reference_trajectory(row)
    user_prompt = get_user_prompt(row)
    (parsed_json, traj, parse_source, parse_error) = parse_trajectory_strict(generated_text)
    fallback_traj = parse_trajectory_fallback_for_debug(generated_text)
    a = ade(traj, ref)
    f = fde(traj, ref)
    (min_agent_distance, collision) = approx_collision_eval(traj, user_prompt)
    si = self_intersection(traj)
    msd = max_step_distance(traj)
    acc = max_step_accel_like(traj)
    strict_structure_ok = traj is not None
    structure_fail = not strict_structure_ok
    ade_fail = a is None or a >= ade_threshold
    strict_failure = bool(structure_fail or collision is True or si is True or ade_fail)
    return {'predicted_json': parsed_json, 'predicted_trajectory': traj, 'parse_source': parse_source, 'parse_error': parse_error, 'fallback_trajectory_debug': fallback_traj, 'fallback_available_debug': fallback_traj is not None, 'strict_structure_ok': bool(strict_structure_ok), 'structure_fail': bool(structure_fail), 'trajectory_length_ok': bool(traj is not None and len(traj) == 6), 'trajectory_finite_ok': bool(traj is not None), 'ade': a, 'fde': f, 'ade_threshold': ade_threshold, 'ade_fail': bool(ade_fail), 'proxy_collision_risk': collision, 'min_agent_distance': min_agent_distance, 'self_intersection': si, 'max_step_distance': msd, 'max_step_accel_like': acc, 'strict_failure_ade1p0': strict_failure, 'strict_clean_pass_ade1p0': not strict_failure}

def write_audit_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['sample_id', 'strict_clean_pass_ade1p0', 'strict_failure_ade1p0', 'structure_fail', 'strict_structure_ok', 'parse_source', 'parse_error', 'fallback_available_debug', 'ade', 'fde', 'ade_fail', 'proxy_collision_risk', 'self_intersection', 'min_agent_distance', 'max_step_distance', 'max_step_accel_like', 'prompt_token_count', 'generated_token_count', 'output_sha1', 'runner_error']
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path', required=True)
    ap.add_argument('--adapter_dir', default='')
    ap.add_argument('--input_jsonl', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--max_samples', type=int, default=0, help='<=0 means all')
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--ade_threshold', type=float, default=1.0)
    ap.add_argument('--load_in_4bit', action='store_true')
    ap.add_argument('--local_files_only', action='store_true')
    ap.add_argument('--enable_thinking_false', action='store_true', help='默认不要开；仅用于 A/B 测试。')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.out_dir / 'v7_clean_candidates_all.jsonl'
    strict_path = args.out_dir / 'v7_clean_reference_strict_ade1p0.jsonl'
    audit_csv = args.out_dir / 'v7_clean_candidates_audit.csv'
    summary_json = args.out_dir / 'v7_clean_candidates_summary.json'
    rows = load_jsonl(args.input_jsonl)
    if args.max_samples and args.max_samples > 0:
        rows = rows[:args.max_samples]
    done_ids = set()
    audit_rows: List[dict] = []
    if args.resume and all_path.exists():
        previous = load_jsonl(all_path)
        audit_rows.extend(previous)
        for r in previous:
            sid = r.get('sample_id')
            if sid is not None:
                done_ids.add(str(sid))
    print(f'[INFO] input rows={len(rows)}, done={len(done_ids)}')
    print('[INFO] loading tokenizer/model...')
    tokenizer = load_tokenizer(args.model_name_or_path, args.adapter_dir, local_files_only=args.local_files_only)
    model = load_model(args.model_name_or_path, args.adapter_dir, load_in_4bit=args.load_in_4bit, local_files_only=args.local_files_only)
    t0 = time.time()
    processed = 0
    for (idx, row) in enumerate(rows, start=1):
        sid = str(row.get('sample_id', f'sample_{idx:06d}'))
        if sid in done_ids:
            continue
        print(f'[{idx}/{len(rows)}] {sid}')
        try:
            gen = infer_one(model, tokenizer, row, max_new_tokens=args.max_new_tokens, enable_thinking_false=args.enable_thinking_false)
            ev = evaluate_output(row, gen['generated_text'], ade_threshold=args.ade_threshold)
            out = {'sample_id': sid, 'system_prompt': get_system_prompt(row), 'user_prompt': get_user_prompt(row), 'reference_answer': get_reference_answer(row), 'reference_trajectory': get_reference_trajectory(row), 'predicted_raw_response': gen['generated_text'], **gen, **ev}
        except Exception as e:
            out = {'sample_id': sid, 'system_prompt': get_system_prompt(row), 'user_prompt': get_user_prompt(row), 'reference_answer': get_reference_answer(row), 'reference_trajectory': get_reference_trajectory(row), 'runner_error': f'{type(e).__name__}: {e}', 'strict_clean_pass_ade1p0': False, 'strict_failure_ade1p0': True, 'structure_fail': True, 'strict_structure_ok': False, 'parse_source': 'failed', 'parse_error': 'runner_exception'}
        append_jsonl(all_path, out)
        audit_rows.append(out)
        if out.get('strict_clean_pass_ade1p0') is True:
            append_jsonl(strict_path, out)
        processed += 1
        if processed % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_audit_csv(audit_csv, audit_rows)
    n = len(audit_rows)
    pass_rows = [r for r in audit_rows if r.get('strict_clean_pass_ade1p0') is True]
    summary = {'input_jsonl': str(args.input_jsonl), 'out_dir': str(args.out_dir), 'max_samples': args.max_samples, 'max_new_tokens': args.max_new_tokens, 'ade_threshold': args.ade_threshold, 'load_in_4bit': bool(args.load_in_4bit), 'local_files_only': bool(args.local_files_only), 'enable_thinking_false': bool(args.enable_thinking_false), 'rows_total_considered': len(rows), 'rows_written_all': n, 'strict_clean_pass_count': len(pass_rows), 'strict_clean_pass_rate': len(pass_rows) / max(1, n), 'structure_fail_count': sum((1 for r in audit_rows if r.get('structure_fail') is True)), 'proxy_collision_count': sum((1 for r in audit_rows if r.get('proxy_collision_risk') is True)), 'self_intersection_count': sum((1 for r in audit_rows if r.get('self_intersection') is True)), 'ade_fail_count': sum((1 for r in audit_rows if r.get('ade_fail') is True)), 'parse_source_counts': {}, 'parse_error_counts': {}, 'elapsed_seconds': time.time() - t0, 'outputs': {'all_jsonl': str(all_path), 'strict_jsonl': str(strict_path), 'audit_csv': str(audit_csv), 'summary_json': str(summary_json)}}
    for r in audit_rows:
        ps = str(r.get('parse_source', ''))
        pe = str(r.get('parse_error', ''))
        summary['parse_source_counts'][ps] = summary['parse_source_counts'].get(ps, 0) + 1
        summary['parse_error_counts'][pe] = summary['parse_error_counts'].get(pe, 0) + 1
    save_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
