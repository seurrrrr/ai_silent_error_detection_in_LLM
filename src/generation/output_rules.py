"""Trajectory parser and geometry predicates retained from the execution runner."""
from __future__ import annotations
import json, math, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

TRAJ_PREFIX = '{"trajectory": [['

EGO_PROXY_RADIUS = 1.2

AGENT_RADIUS = {'adult': 0.8, 'child': 0.7, 'pedestrian': 0.8, 'bicycle': 1.0, 'motorcycle': 1.0, 'animal': 0.8, 'car': 2.0, 'truck': 2.7, 'bus': 3.0, 'trailer': 3.0, 'construction': 2.5, 'construction_vehicle': 2.8, 'barrier': 0.8, 'trafficcone': 0.4, 'traffic_cone': 0.4, 'rigid': 2.3}

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
