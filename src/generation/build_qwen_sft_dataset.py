"""Structured trajectory data preparation.
Packaging adaptation: portable file references; numerical operations retained.
"""
import json
import pickle
import argparse
from pathlib import Path
import numpy as np

def jsonl_to_dict(filename):
    data_dict = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            json_obj = json.loads(line)
            token = json_obj['token']
            data_dict[token] = json_obj['detections']
    return data_dict

def infer_mission_goal(data_dict):
    cmd_vec = data_dict['gt_ego_fut_cmd']
    (right, left, forward) = cmd_vec
    if right > 0:
        return 'RIGHT'
    elif left > 0:
        return 'LEFT'
    else:
        return 'FORWARD'

def build_user_message(data, token):
    """
    尽量对齐作者 prompt_message.py 的 user_message 逻辑，
    但这里直接在本地脚本里重写，避免 import 路径折腾。
    """
    data_dict = data[token]
    user_message = ''
    user_message += 'Perception and Prediction:\n'
    object_boxes = data_dict['gt_boxes']
    object_names = data_dict['gt_names']
    if len(object_boxes) > 0:
        object_rel_fut_trajs = data_dict['gt_agent_fut_trajs'].reshape(-1, 6, 2)
        object_fut_trajs = np.cumsum(object_rel_fut_trajs, axis=1) + object_boxes[:, None, :2]
        object_fut_trajs = np.concatenate([object_boxes[:, None, :2], object_fut_trajs], axis=1)
        object_fut_mask = data_dict['gt_agent_fut_masks']
    else:
        object_fut_trajs = np.zeros((0, 7, 2))
        object_fut_mask = np.zeros((0, 6))
    perception_range = 20.0
    for i in range(len(object_boxes)):
        cur_xy = object_boxes[i, :2]
        if (np.abs(cur_xy) > perception_range).any():
            continue
        name = str(object_names[i])
        fut_xy = cur_xy.copy()
        if i < len(object_fut_trajs):
            valid_steps = []
            if i < len(object_fut_mask):
                for t in range(6):
                    if object_fut_mask[i, t] > 0:
                        valid_steps.append(t + 1)
            if len(valid_steps) > 0:
                fut_xy = object_fut_trajs[i, valid_steps[-1], :]
            else:
                fut_xy = object_fut_trajs[i, -1, :]
        user_message += f' - {name} at ({cur_xy[0]:.2f},{cur_xy[1]:.2f}), moving to ({fut_xy[0]:.2f},{fut_xy[1]:.2f}).\n'
    vx = data_dict['gt_ego_lcf_feat'][0] * 0.5
    vy = data_dict['gt_ego_lcf_feat'][1] * 0.5
    v_yaw = data_dict['gt_ego_lcf_feat'][4]
    ax = data_dict['gt_ego_his_diff'][-1, 0] - data_dict['gt_ego_his_diff'][-2, 0]
    ay = data_dict['gt_ego_his_diff'][-1, 1] - data_dict['gt_ego_his_diff'][-2, 1]
    cx = data_dict['gt_ego_lcf_feat'][2]
    cy = data_dict['gt_ego_lcf_feat'][3]
    vhead = data_dict['gt_ego_lcf_feat'][7] * 0.5
    steering = data_dict['gt_ego_lcf_feat'][8]
    user_message += 'Ego-States:\n'
    user_message += f' - Velocity (vx,vy): ({vx:.2f},{vy:.2f})\n'
    user_message += f' - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})\n'
    user_message += f' - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})\n'
    user_message += f' - Can Bus: ({cx:.2f},{cy:.2f})\n'
    user_message += f' - Heading Speed: ({vhead:.2f})\n'
    user_message += f' - Steering: ({steering:.2f})\n'
    ego_his = data_dict['gt_ego_his_trajs']
    (xh1, yh1) = (ego_his[0][0], ego_his[0][1])
    (xh2, yh2) = (ego_his[1][0], ego_his[1][1])
    (xh3, yh3) = (ego_his[2][0], ego_his[2][1])
    (xh4, yh4) = (ego_his[3][0], ego_his[3][1])
    user_message += 'Historical Trajectory (last 2 seconds):'
    user_message += f' [({xh1:.2f},{yh1:.2f}), ({xh2:.2f},{yh2:.2f}), ({xh3:.2f},{yh3:.2f}), ({xh4:.2f},{yh4:.2f})]\n'
    mission_goal = infer_mission_goal(data_dict)
    user_message += f'Mission Goal: {mission_goal}\n'
    return user_message

def build_system_message():
    return 'You are an autonomous driving planning module.\n\nTask:\nGiven perception and prediction of nearby agents, ego states, historical trajectory, and mission goal, predict a safe and feasible 3-second future trajectory.\n\nOutput requirements:\nReturn ONLY one valid JSON object.\nDo not output chain-of-thought.\nDo not output markdown.\nDo not add any extra explanation.\n\nRequired JSON schema:\n{\n  "trajectory": [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]]\n}\n'

def build_assistant_json_from_gt(data, token):
    data_dict = data[token]
    traj = data_dict['gt_ego_fut_trajs']
    out = []
    for i in range(1, 7):
        x = round(float(traj[i][0]), 2)
        y = round(float(traj[i][1]), 2)
        out.append([x, y])
    return {'trajectory': out}

def inject_uniad_perception(data, uniad_perceptions, token):
    """
    对齐 create_data_uniad.py 的逻辑：
    用 detection_motion_result_trainval.jsonl 里的 detections 替换/构造感知输入，
    但监督目标仍然保留 GT ego future trajectory。
    """
    uniad_perception = uniad_perceptions[token]
    (uniad_boxes, uniad_names, uniad_trajs) = ([], [], [])
    for obj in uniad_perception:
        uniad_names.append(obj['name'])
        uniad_boxes.append(obj['box'])
        box_center = np.array(obj['box'][:2], dtype=np.float32)
        full_traj = np.array(obj['traj'][:6], dtype=np.float32)
        rel_traj = full_traj - box_center[None, :]
        rel_traj = np.concatenate([np.zeros((1, 2), dtype=np.float32), rel_traj], axis=0)
        rel_diff_traj = rel_traj[1:] - rel_traj[:-1]
        uniad_trajs.append(rel_diff_traj)
    if len(uniad_trajs) == 0:
        data[token]['gt_agent_fut_trajs'] = np.zeros((0, 6, 2), dtype=np.float32)
        data[token]['gt_boxes'] = np.zeros((0, 9), dtype=np.float32)
        data[token]['gt_names'] = np.array([])
    else:
        data[token]['gt_boxes'] = np.array(uniad_boxes, dtype=np.float32)
        data[token]['gt_names'] = np.array(uniad_names)
        data[token]['gt_agent_fut_trajs'] = np.stack(uniad_trajs, axis=0)
    data[token]['gt_agent_fut_masks'] = np.ones((len(uniad_boxes), 6), dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached_pkl', type=str, required=True)
    parser.add_argument('--split_json', type=str, required=True)
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument('--perception_source', type=str, default='gt', choices=['gt', 'uniad'], help='gt: 用 cached_nuscenes_info.pkl 中已有感知；uniad: 用 detection_motion_result_trainval.jsonl 替换感知输入')
    parser.add_argument('--uniad_jsonl', type=str, default='')
    parser.add_argument('--split_name', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--max_samples', type=int, default=-1)
    args = parser.parse_args()
    cached_pkl = Path(args.cached_pkl)
    split_json = Path(args.split_json)
    output_jsonl = Path(args.output_jsonl)
    print(f'[INFO] Loading cached data: {cached_pkl}')
    with open(cached_pkl, 'rb') as f:
        data = pickle.load(f)
    print(f'[INFO] Loading split: {split_json}')
    with open(split_json, 'r', encoding='utf-8') as f:
        split = json.load(f)
    tokens = split[args.split_name]
    print(f'[INFO] split={args.split_name}, total tokens={len(tokens)}')
    if args.max_samples > 0:
        tokens = tokens[:args.max_samples]
        print(f'[INFO] using first {len(tokens)} samples')
    if args.perception_source == 'uniad':
        if not args.uniad_jsonl:
            raise ValueError('perception_source=uniad 时必须提供 --uniad_jsonl')
        print(f'[INFO] Loading UniAD detections: {args.uniad_jsonl}')
        uniad_perceptions = jsonl_to_dict(args.uniad_jsonl)
    else:
        uniad_perceptions = None
    system_message = build_system_message()
    count = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for (idx, token) in enumerate(tokens, start=1):
            if args.perception_source == 'uniad':
                inject_uniad_perception(data, uniad_perceptions, token)
            user_message = build_user_message(data, token)
            assistant_obj = build_assistant_json_from_gt(data, token)
            assistant_text = json.dumps(assistant_obj, ensure_ascii=False)
            row = {'sample_id': f'{args.split_name}_{idx:06d}', 'token': token, 'perception_source': args.perception_source, 'messages': [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': user_message}, {'role': 'assistant', 'content': assistant_text}], 'reference_trajectory': assistant_obj['trajectory']}
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
            if idx <= 3:
                print(f'\n[Preview {idx}] token={token}')
                print('[System]')
                print(system_message[:300] + '...')
                print('[User]')
                print(user_message[:800] + '...')
                print('[Assistant]')
                print(assistant_text)
    print(f'\n[DONE] Wrote {count} samples to: {output_jsonl}')
if __name__ == '__main__':
    main()
