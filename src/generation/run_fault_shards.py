"""Resumable fault-campaign shard execution.
Packaging adaptation: portable file references; numerical operations retained.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Optional

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8-sig') as f:
        for (line_no, line) in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                raise RuntimeError(f'JSONL parse failed: {path} line={line_no}: {e}') from e
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def read_completed_ids(out_dir: Path) -> Set[str]:
    """
    从 events_jsonl 和 compact csv 中读取已完成 campaign_id。
    正常 runner 每处理一条都会写 event；compact 作为兜底。
    """
    done: Set[str] = set()
    events = out_dir / 'v7_fault_b18_injection_events.jsonl'
    if events.exists():
        with events.open('r', encoding='utf-8-sig') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    cid = obj.get('campaign_id')
                    if cid:
                        done.add(str(cid))
                except Exception:
                    continue
    compact = out_dir / 'v7_fault_b18_compact_deviation_features.csv'
    if compact.exists():
        try:
            with compact.open('r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and 'campaign_id' in reader.fieldnames:
                    for row in reader:
                        cid = row.get('campaign_id')
                        if cid:
                            done.add(str(cid))
        except Exception:
            pass
    return done

def read_summary(out_dir: Path) -> Optional[Dict[str, Any]]:
    p = out_dir / 'v7_fault_b18_summary.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception:
        return None

def shard_original_campaign(shards_dir: Path, prefix: str, sid: str) -> Path:
    return shards_dir / f'{prefix}_campaign_shard_{sid}.jsonl'

def build_autoskip_campaign(original: Path, skip_ids: Set[str], out_path: Path) -> int:
    rows = load_jsonl(original)
    kept = [r for r in rows if str(r.get('campaign_id', '')) not in skip_ids]
    write_jsonl(out_path, kept)
    return len(kept)

def first_pending_id(campaign_path: Path, done: Set[str], skip_ids: Set[str]) -> Optional[str]:
    for r in load_jsonl(campaign_path):
        cid = str(r.get('campaign_id', ''))
        if not cid:
            continue
        if cid in skip_ids:
            continue
        if cid not in done:
            return cid
    return None

def append_autoskip_log(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

def load_skip_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(data, list):
            return {str(x) for x in data}
        if isinstance(data, dict) and 'skip_ids' in data:
            return {str(x) for x in data['skip_ids']}
    except Exception:
        pass
    return set()

def save_skip_ids(path: Path, skip_ids: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(skip_ids), ensure_ascii=False, indent=2), encoding='utf-8')

def run_runner(args: argparse.Namespace, campaign_jsonl: Path, out_dir: Path) -> int:
    cmd = [str(args.py), '-u', str(args.runner), '--model_name_or_path', str(args.model_name_or_path), '--clean_bank_jsonl', str(args.clean_bank_jsonl), '--campaign_jsonl', str(campaign_jsonl), '--monitor_targets', str(args.monitor_targets), '--profile_stats_csv', str(args.profile_stats_csv), '--out_dir', str(out_dir), '--max_new_tokens', str(args.max_new_tokens), '--decode_steps', str(args.decode_steps), '--stop_after_trajectory_array', '--local_files_only', '--resume']
    if args.adapter_dir:
        cmd.extend(['--adapter_dir', str(args.adapter_dir)])
    if args.load_in_4bit:
        cmd.append('--load_in_4bit')
    print('[RUN]', ' '.join((f'"{x}"' if ' ' in x else x for x in cmd)), flush=True)
    proc = subprocess.run(cmd)
    return int(proc.returncode)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--py', type=Path, required=True)
    ap.add_argument('--runner', type=Path, required=True)
    ap.add_argument('--model_name_or_path', required=True)
    ap.add_argument('--adapter_dir', default='')
    ap.add_argument('--clean_bank_jsonl', type=Path, required=True)
    ap.add_argument('--shards_dir', type=Path, required=True)
    ap.add_argument('--campaign_prefix', default='v9b_dense_fault6500')
    ap.add_argument('--monitor_targets', type=Path, required=True)
    ap.add_argument('--profile_stats_csv', type=Path, required=True)
    ap.add_argument('--out_root', type=Path, required=True)
    ap.add_argument('--decode_steps', required=True)
    ap.add_argument('--start_shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=13)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--load_in_4bit', action='store_true', default=True)
    ap.add_argument('--skip_after_same_pending_failures', type=int, default=2)
    ap.add_argument('--max_autoskips_per_shard', type=int, default=20)
    ap.add_argument('--sleep_after_failure_sec', type=int, default=45)
    args = ap.parse_args()
    autoskip_root = args.out_root / '_autosupervisor'
    autoskip_root.mkdir(parents=True, exist_ok=True)
    global_log = autoskip_root / 'autosupervisor_log.jsonl'
    for i in range(args.start_shard, args.num_shards):
        sid = f'{i:03d}'
        out_dir = args.out_root / f'shard_{sid}'
        out_dir.mkdir(parents=True, exist_ok=True)
        original = shard_original_campaign(args.shards_dir, args.campaign_prefix, sid)
        if not original.exists():
            print(f'[ERROR] missing shard campaign: {original}', flush=True)
            sys.exit(2)
        skip_ids_path = autoskip_root / f'shard_{sid}_skip_ids.json'
        skip_ids = load_skip_ids(skip_ids_path)
        fail_counts: Dict[str, int] = {}
        attempts = 0
        print(f'\n===== SUPERVISOR shard {sid} start skip_ids={len(skip_ids)} =====', flush=True)
        while True:
            summary = read_summary(out_dir)
            if summary is not None:
                total = int(summary.get('total_campaign_rows', 0) or 0)
                processed = int(summary.get('processed_now', 0) or 0)
                errors = int(summary.get('errors_now', 0) or 0)
                missing = int(summary.get('missing_target_now', 0) or 0)
                if total > 0 and processed + errors + missing >= total:
                    print(f'[OK] shard {sid} already complete by summary: total={total} processed={processed} errors={errors} missing={missing}', flush=True)
                    append_autoskip_log(global_log, {'event': 'shard_complete_by_summary', 'shard': sid, 'summary': summary, 'skip_ids': sorted(skip_ids)})
                    break
            attempts += 1
            if len(skip_ids) > args.max_autoskips_per_shard:
                print(f'[STOP] shard {sid} too many autoskips: {len(skip_ids)} > {args.max_autoskips_per_shard}', flush=True)
                sys.exit(3)
            current_campaign = original
            if skip_ids:
                current_campaign = autoskip_root / f'shard_{sid}_campaign_autoskip.jsonl'
                kept = build_autoskip_campaign(original, skip_ids, current_campaign)
                print(f'[INFO] shard {sid} autoskip campaign rows={kept} skipped={len(skip_ids)}', flush=True)
            done_before = read_completed_ids(out_dir)
            pending_before = first_pending_id(original, done_before, skip_ids)
            print(f'[INFO] shard {sid} attempt={attempts} done_before={len(done_before)} pending_before={pending_before} skip_ids={len(skip_ids)}', flush=True)
            ret = run_runner(args, current_campaign, out_dir)
            if ret == 0:
                print(f'[OK] shard {sid} runner exit 0', flush=True)
                append_autoskip_log(global_log, {'event': 'shard_exit0', 'shard': sid, 'attempts': attempts, 'skip_ids': sorted(skip_ids), 'summary': read_summary(out_dir)})
                break
            print(f'[WARN] shard {sid} runner failed exit={ret}', flush=True)
            done_after = read_completed_ids(out_dir)
            pending_after = first_pending_id(original, done_after, skip_ids)
            stuck_id = pending_after or pending_before
            append_autoskip_log(global_log, {'event': 'runner_failed', 'shard': sid, 'exit_code': ret, 'done_before': len(done_before), 'done_after': len(done_after), 'pending_before': pending_before, 'pending_after': pending_after, 'skip_ids': sorted(skip_ids)})
            if stuck_id is None:
                print(f'[WARN] shard {sid} failed but no pending id found. Sleep and retry.', flush=True)
                time.sleep(args.sleep_after_failure_sec)
                continue
            fail_counts[stuck_id] = fail_counts.get(stuck_id, 0) + 1
            print(f'[WARN] shard {sid} stuck_id={stuck_id} fail_count={fail_counts[stuck_id]}', flush=True)
            if fail_counts[stuck_id] >= args.skip_after_same_pending_failures:
                skip_ids.add(stuck_id)
                save_skip_ids(skip_ids_path, skip_ids)
                append_autoskip_log(global_log, {'event': 'autoskip_added', 'shard': sid, 'campaign_id': stuck_id, 'fail_count': fail_counts[stuck_id], 'skip_ids_path': str(skip_ids_path)})
                print(f'[AUTOSKIP] shard {sid} campaign_id={stuck_id}; total_skips={len(skip_ids)}', flush=True)
            print(f'[INFO] sleep {args.sleep_after_failure_sec}s before retry', flush=True)
            time.sleep(args.sleep_after_failure_sec)
    print('\n[ALL DONE] supervisor finished requested shards', flush=True)
if __name__ == '__main__':
    main()
