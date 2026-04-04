import os
import json
import math
import time
import random
import struct
import argparse
from dataclasses import dataclass, asdict

import torch

from qwen3_5_9b_4bit import load_model_and_tokenizer, FaultType


# =========================================================
# 1. 基础工具
# =========================================================

def safe_float(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def shorten(text, n=160):
    text = (text or "").replace("\n", " ")
    return text[:n]


def tensor_stats(t: torch.Tensor):
    tf = t.detach().float()
    flat = tf.reshape(-1)

    mean = safe_float(flat.mean().item(), 0.0)
    std = safe_float(flat.std().item(), 0.0) if flat.numel() > 1 else 0.0
    mn = safe_float(flat.min().item(), 0.0)
    mx = safe_float(flat.max().item(), 0.0)
    l2 = safe_float(torch.norm(flat, p=2).item(), 0.0)
    l1 = safe_float(torch.norm(flat, p=1).item(), 0.0)
    zero_ratio = safe_float((flat == 0).float().mean().item(), 0.0)
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())

    # 极端值比例（粗略估计）
    abs_flat = flat.abs()
    if flat.numel() > 1:
        sat_thr = abs_flat.mean().item() + 6.0 * abs_flat.std().item()
        sat_ratio = safe_float((abs_flat > sat_thr).float().mean().item(), 0.0) if sat_thr > 0 else 0.0
    else:
        sat_ratio = 0.0

    checksum = safe_float(flat[: min(1024, flat.numel())].sum().item(), 0.0)

    return {
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "l2": l2,
        "l1": l1,
        "zero_ratio": zero_ratio,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "sat_ratio": sat_ratio,
        "checksum": checksum,
        "numel": int(flat.numel()),
    }


def compare_stats(base, cur):
    eps = 1e-9
    return {
        "mean_rel": abs(cur["mean"] - base["mean"]) / (abs(base["mean"]) + eps),
        "std_rel": abs(cur["std"] - base["std"]) / (abs(base["std"]) + eps),
        "l2_rel": abs(cur["l2"] - base["l2"]) / (abs(base["l2"]) + eps),
        "l1_rel": abs(cur["l1"] - base["l1"]) / (abs(base["l1"]) + eps),
        "zero_ratio_abs": abs(cur["zero_ratio"] - base["zero_ratio"]),
        "sat_ratio_abs": abs(cur["sat_ratio"] - base["sat_ratio"]),
        "checksum_rel": abs(cur["checksum"] - base["checksum"]) / (abs(base["checksum"]) + eps),
        "nan_added": max(0, cur["nan_count"] - base["nan_count"]),
        "inf_added": max(0, cur["inf_count"] - base["inf_count"]),
    }


def is_layer_anomalous(diff):
    # 第一层：层输出监测
    if diff["nan_added"] > 0 or diff["inf_added"] > 0:
        return True
    if diff["l2_rel"] > 0.20:
        return True
    if diff["std_rel"] > 0.30:
        return True
    if diff["sat_ratio_abs"] > 0.05:
        return True
    return False


def is_state_anomalous(diff):
    # 第二层：状态/KV代理监测
    if diff["nan_added"] > 0 or diff["inf_added"] > 0:
        return True
    if diff["zero_ratio_abs"] > 0.10:
        return True
    if diff["checksum_rel"] > 0.30:
        return True
    if diff["l2_rel"] > 0.15:
        return True
    return False


def is_state_like_module(name: str):
    name = name.lower()
    keywords = [
        "attn",
        "linear_attn",
        "self_attn",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "in_proj",
        "conv1d",
        "norm",
    ]
    return any(k in name for k in keywords)


# =========================================================
# 2. 生成回答
# =========================================================

def generate_response(model, tokenizer, question, max_new_tokens=64):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# =========================================================
# 3. 故障注入
# =========================================================

def _flip_bits(val: float, lo: int, hi: int, n: int = 1) -> float:
    packed = struct.pack("f", float(val))
    bits = int.from_bytes(packed, "little")
    positions = random.sample(range(lo, hi + 1), min(n, hi - lo + 1))
    for p in positions:
        bits ^= (1 << p)
    result = struct.unpack("f", bits.to_bytes(4, "little"))[0]
    if math.isnan(result) or math.isinf(result) or abs(result) > 1e15:
        return val
    return result


def apply_fault_to_tensor(
    t: torch.Tensor,
    ft: FaultType,
    flip_count=1,
    num_flips=1,
    bit_flip_low_bits=3,
    bit_flip_mid_bits=(8, 15),
    bit_flip_high_bits=(23, 30),
    stuck_ratio=0.01,
    block_size_ratio=0.02,
    block_noise_std=0.1,
    kv_noise_std_ratio=0.02,
    kv_zero_ratio=0.05,
    bias_ratio=0.01,
):
    tf = t.detach().clone().float()
    flat = tf.reshape(-1)

    detail = {
        "orig_val": 0.0,
        "inj_val": 0.0,
        "inject_delta": 0.0,
        "position": (),
    }

    if flat.numel() == 0:
        return tf.to(t.dtype), detail

    # 1) bit flip 系列
    if ft == FaultType.BIT_FLIP_LOW:
        idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
        detail["orig_val"] = flat[idxs[0]].item()
        for idx in idxs:
            flat[idx] = _flip_bits(flat[idx].item(), 0, bit_flip_low_bits - 1, num_flips)
        detail["inj_val"] = flat[idxs[0]].item()
        detail["position"] = tuple(idxs[:4])

    elif ft == FaultType.BIT_FLIP_MID:
        idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
        detail["orig_val"] = flat[idxs[0]].item()
        for idx in idxs:
            flat[idx] = _flip_bits(flat[idx].item(), *bit_flip_mid_bits, num_flips)
        detail["inj_val"] = flat[idxs[0]].item()
        detail["position"] = tuple(idxs[:4])

    elif ft == FaultType.BIT_FLIP_HIGH:
        idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
        detail["orig_val"] = flat[idxs[0]].item()
        for idx in idxs:
            flat[idx] = _flip_bits(flat[idx].item(), *bit_flip_high_bits, num_flips)
        detail["inj_val"] = flat[idxs[0]].item()
        detail["position"] = tuple(idxs[:4])

    elif ft == FaultType.BIT_FLIP_SIGN:
        idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
        detail["orig_val"] = flat[idxs[0]].item()
        for idx in idxs:
            flat[idx] = _flip_bits(flat[idx].item(), 31, 31, 1)
        detail["inj_val"] = flat[idxs[0]].item()
        detail["position"] = tuple(idxs[:4])

    # 2) permanent/stuck 系列
    elif ft == FaultType.STUCK_AT_ZERO:
        if tf.dim() >= 2:
            n_ch = tf.shape[-1]
            n_stuck = max(1, int(n_ch * stuck_ratio))
            idxs = random.sample(range(n_ch), min(n_stuck, n_ch))
            detail["orig_val"] = tf[..., idxs].abs().mean().item()
            tf[..., idxs] = 0.0
            detail["inj_val"] = 0.0
            detail["position"] = tuple(idxs[:4])

    elif ft == FaultType.STUCK_AT_ONE:
        if tf.dim() >= 2:
            n_ch = tf.shape[-1]
            n_stuck = max(1, int(n_ch * stuck_ratio))
            idxs = random.sample(range(n_ch), min(n_stuck, n_ch))
            detail["orig_val"] = tf[..., idxs].abs().mean().item()
            max_val = tf.abs().max().item()
            tf[..., idxs] = max_val
            detail["inj_val"] = tf[..., idxs].abs().mean().item()
            detail["position"] = tuple(idxs[:4])

    # 3) block corruption
    elif ft == FaultType.BLOCK_CORRUPT:
        n = flat.numel()
        blen = max(1, int(n * block_size_ratio))
        start = random.randint(0, max(0, n - blen))
        std = flat.abs().mean().item() * block_noise_std
        detail["orig_val"] = flat[start:start + blen].abs().mean().item()
        flat[start:start + blen] += torch.randn(blen, device=tf.device, dtype=tf.dtype) * std
        detail["inj_val"] = flat[start:start + blen].abs().mean().item()
        detail["position"] = (start, start + blen)

    # 4) KV/state fault
    elif ft == FaultType.KV_NOISE:
        detail["orig_val"] = tf.abs().mean().item()
        tf.add_(torch.randn_like(tf) * (detail["orig_val"] * kv_noise_std_ratio))
        detail["inj_val"] = tf.abs().mean().item()

    elif ft == FaultType.KV_ZERO_REGION:
        if tf.dim() >= 2:
            if tf.dim() >= 3:
                n_seq = tf.shape[-2]
                n_zero = max(1, int(n_seq * kv_zero_ratio))
                start = random.randint(0, max(0, n_seq - n_zero))
                detail["orig_val"] = tf[..., start:start+n_zero, :].abs().mean().item()
                tf[..., start:start+n_zero, :] = 0.0
                detail["inj_val"] = 0.0
                detail["position"] = (start, start + n_zero)
            else:
                n_seq = tf.shape[0]
                n_zero = max(1, int(n_seq * kv_zero_ratio))
                start = random.randint(0, max(0, n_seq - n_zero))
                detail["orig_val"] = tf[start:start+n_zero, ...].abs().mean().item()
                tf[start:start+n_zero, ...] = 0.0
                detail["inj_val"] = 0.0
                detail["position"] = (start, start + n_zero)

    # 5) bias / drift
    elif ft == FaultType.PERSISTENT_BIAS:
        detail["orig_val"] = tf.abs().mean().item()
        tf.add_(detail["orig_val"] * bias_ratio)
        detail["inj_val"] = tf.abs().mean().item()

    detail["inject_delta"] = (tf.float() - t.detach().float()).abs().mean().item()
    return tf.to(t.dtype), detail


# =========================================================
# 4. 监测器
# =========================================================

class StatsCollector:
    def __init__(self):
        self.stats_sum = {}
        self.count = {}

    def update(self, name, tensor):
        if not hasattr(tensor, "shape"):
            return
        st = tensor_stats(tensor)
        if name not in self.stats_sum:
            self.stats_sum[name] = {k: 0.0 for k in st.keys() if k != "numel"}
            self.stats_sum[name]["numel"] = st["numel"]
            self.count[name] = 0

        for k, v in st.items():
            if k == "numel":
                self.stats_sum[name]["numel"] = st["numel"]
            else:
                self.stats_sum[name][k] += float(v)

        self.count[name] += 1

    def finalize(self):
        out = {}
        for name, sums in self.stats_sum.items():
            c = max(1, self.count[name])
            out[name] = {}
            for k, v in sums.items():
                if k == "numel":
                    out[name][k] = int(v)
                else:
                    out[name][k] = float(v) / c
        return out


def build_target_layer_names(model, layer_filter="layers"):
    names = []
    for name, module in model.named_modules():
        if layer_filter in name:
            names.append(name)
    return names


def build_baseline_stats(model, tokenizer, question, layer_filter="layers", max_new_tokens=64):
    collector = StatsCollector()
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if hasattr(t, "shape"):
                collector.update(name, t)
            return output
        return hook

    for name, module in model.named_modules():
        if layer_filter in name:
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        baseline_answer = generate_response(model, tokenizer, question, max_new_tokens=max_new_tokens)
    finally:
        for h in handles:
            h.remove()

    baseline_stats = collector.finalize()
    return baseline_answer, baseline_stats


@dataclass
class LayerwiseRunResult:
    question: str
    fault_type: str
    target_layer: str
    baseline_answer: str
    fault_answer: str
    layer_detected: bool
    state_detected: bool
    overall_detected: bool
    first_layer_alarm: str
    first_state_alarm: str
    layer_alarm_count: int
    state_alarm_count: int
    layer_alarm_modules: list
    state_alarm_modules: list
    output_changed: bool
    inject_detail: dict
    elapsed_sec: float


def run_one_layer_injection_with_full_monitor(
    model,
    tokenizer,
    question,
    fault_type: FaultType,
    target_layer: str,
    baseline_answer: str,
    baseline_stats: dict,
    layer_filter="layers",
    max_new_tokens=64,
):
    layer_alarms = []
    state_alarms = []
    first_layer_alarm = ""
    first_state_alarm = ""
    inject_detail = {}
    fault_done = [False]

    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            nonlocal first_layer_alarm, first_state_alarm, inject_detail

            t = output[0] if isinstance(output, (tuple, list)) else output
            if not hasattr(t, "shape"):
                return output

            # 在目标层第一次执行时注入故障
            if name == target_layer and not fault_done[0]:
                t_fault, detail = apply_fault_to_tensor(t, fault_type)
                fault_done[0] = True
                inject_detail = detail
                t = t_fault
                out_local = (t, *output[1:]) if isinstance(output, (tuple, list)) else t
            else:
                out_local = output

            if name in baseline_stats:
                cur = tensor_stats(t)
                diff = compare_stats(baseline_stats[name], cur)

                # 第一层：层输出监测
                if is_layer_anomalous(diff):
                    if name not in layer_alarms:
                        layer_alarms.append(name)
                    if not first_layer_alarm:
                        first_layer_alarm = name

                # 第二层：状态/KV代理监测
                if is_state_like_module(name) and is_state_anomalous(diff):
                    if name not in state_alarms:
                        state_alarms.append(name)
                    if not first_state_alarm:
                        first_state_alarm = name

            return out_local
        return hook

    for name, module in model.named_modules():
        if layer_filter in name:
            handles.append(module.register_forward_hook(make_hook(name)))

    t0 = time.time()
    try:
        fault_answer = generate_response(model, tokenizer, question, max_new_tokens=max_new_tokens)
    finally:
        for h in handles:
            h.remove()
    elapsed = time.time() - t0

    layer_detected = len(layer_alarms) > 0
    state_detected = len(state_alarms) > 0
    overall_detected = layer_detected or state_detected
    output_changed = baseline_answer.strip() != fault_answer.strip()

    return LayerwiseRunResult(
        question=question,
        fault_type=getattr(fault_type, "value", str(fault_type)),
        target_layer=target_layer,
        baseline_answer=baseline_answer,
        fault_answer=fault_answer,
        layer_detected=layer_detected,
        state_detected=state_detected,
        overall_detected=overall_detected,
        first_layer_alarm=first_layer_alarm,
        first_state_alarm=first_state_alarm,
        layer_alarm_count=len(layer_alarms),
        state_alarm_count=len(state_alarms),
        layer_alarm_modules=layer_alarms[:30],
        state_alarm_modules=state_alarms[:30],
        output_changed=output_changed,
        inject_detail=inject_detail,
        elapsed_sec=elapsed,
    )


# =========================================================
# 5. fault type 解析
# =========================================================

def parse_fault_type(name: str):
    name = name.strip().upper()
    mapping = {
        "BIT_FLIP_LOW": FaultType.BIT_FLIP_LOW,
        "BIT_FLIP_MID": FaultType.BIT_FLIP_MID,
        "BIT_FLIP_HIGH": FaultType.BIT_FLIP_HIGH,
        "BIT_FLIP_SIGN": FaultType.BIT_FLIP_SIGN,
        "STUCK_AT_ZERO": FaultType.STUCK_AT_ZERO,
        "STUCK_AT_ONE": FaultType.STUCK_AT_ONE,
        "BLOCK_CORRUPT": FaultType.BLOCK_CORRUPT,
        "KV_NOISE": FaultType.KV_NOISE,
        "KV_ZERO_REGION": FaultType.KV_ZERO_REGION,
        "PERSISTENT_BIAS": FaultType.PERSISTENT_BIAS,
    }
    if name not in mapping:
        raise ValueError(f"未知 fault type: {name}")
    return mapping[name]


# =========================================================
# 6. 主程序：逐层全扫 + 全量监测
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="逐层全注入 + 全量监测（前两层）")
    parser.add_argument("--fault_type", type=str, required=True,
                        help="如 BIT_FLIP_LOW / BLOCK_CORRUPT / KV_ZERO_REGION")
    parser.add_argument("--question", type=str,
                        default="请用简洁的语言解释什么是量子纠缠。")
    parser.add_argument("--layer_filter", type=str, default="layers")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_path", type=str, default="layerwise_full_monitor.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    fault_type = parse_fault_type(args.fault_type)

    print("=== Step 1: 加载模型和分词器 ===")
    model, tokenizer = load_model_and_tokenizer()

    print("=== Step 2: 构建 baseline ===")
    baseline_answer, baseline_stats = build_baseline_stats(
        model=model,
        tokenizer=tokenizer,
        question=args.question,
        layer_filter=args.layer_filter,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"[Baseline] answer: {shorten(baseline_answer)}")

    target_layers = build_target_layer_names(model, layer_filter=args.layer_filter)
    print(f"=== Step 3: 开始逐层扫描 ===")
    print(f"[Info] fault_type={args.fault_type}")
    print(f"[Info] target_layers={len(target_layers)}")

    if os.path.dirname(args.log_path):
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    overall_detected_count = 0
    layer_detected_count = 0
    state_detected_count = 0
    output_changed_count = 0
    missed_layers = []

    with open(args.log_path, "w", encoding="utf-8") as f:
        for idx, target_layer in enumerate(target_layers, 1):
            print(f"\n[Layer {idx}/{len(target_layers)}] target={target_layer}")

            result = run_one_layer_injection_with_full_monitor(
                model=model,
                tokenizer=tokenizer,
                question=args.question,
                fault_type=fault_type,
                target_layer=target_layer,
                baseline_answer=baseline_answer,
                baseline_stats=baseline_stats,
                layer_filter=args.layer_filter,
                max_new_tokens=args.max_new_tokens,
            )

            if result.overall_detected:
                overall_detected_count += 1
            else:
                missed_layers.append(target_layer)

            if result.layer_detected:
                layer_detected_count += 1

            if result.state_detected:
                state_detected_count += 1

            if result.output_changed:
                output_changed_count += 1

            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

            print(f"  detected={result.overall_detected} | layer_detected={result.layer_detected} | state_detected={result.state_detected}")
            print(f"  first_layer_alarm={result.first_layer_alarm}")
            print(f"  first_state_alarm={result.first_state_alarm}")
            print(f"  output_changed={result.output_changed}")
            print(f"  fault_answer={shorten(result.fault_answer)}")

    total = len(target_layers)
    print("\n=== Step 4: 扫描完成 ===")
    print(f"fault_type                = {args.fault_type}")
    print(f"total_layers              = {total}")
    print(f"overall_detected          = {overall_detected_count}/{total} = {overall_detected_count / max(1,total):.2%}")
    print(f"layer_detected            = {layer_detected_count}/{total} = {layer_detected_count / max(1,total):.2%}")
    print(f"state_detected            = {state_detected_count}/{total} = {state_detected_count / max(1,total):.2%}")
    print(f"output_changed            = {output_changed_count}/{total} = {output_changed_count / max(1,total):.2%}")
    print(f"missed_layers_count       = {len(missed_layers)}")
    print(f"log_path                  = {args.log_path}")

    if missed_layers:
        print("\n[Missed Layers Preview]")
        for name in missed_layers[:20]:
            print(" -", name)


if __name__ == "__main__":
    main()