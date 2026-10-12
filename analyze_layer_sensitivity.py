"""
analyze_layer_sensitivity.py
──────────────────────────────────────────────────────────────────────────────
对 layer_sensitivity.log 进行全面统计性分析，输出分析报告。

分析维度：
  1. 总览与异常检测（极值离群层）
  2. 按结构大类（self_attn / mlp / linear_attn / layernorm / whole_layer）
  3. 按具体组件（q_proj / v_proj / conv1d / down_proj …）
  4. 按层深度（浅层 / 中层 / 深层）
  5. 按注入故障类型
  6. 按故障大类（位翻转 / 永久故障 / KV 扰动 / 存储块 / 老化偏置）
  7. 三个子指标（Δsemantic / Δlogits / NormDrift）的相关性
  8. 多次扫描一致性（若日志包含多轮扫描）

用法：
  python analyze_layer_sensitivity.py [日志路径] [报告路径]
"""

import re
import sys
import io
import math
import statistics
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")          # 非交互后端，适合脚本直接保存图片
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

PLOT_DIR = "sensitivity_plots"     # 图片输出目录

# Windows 控制台强制使用 UTF-8 输出，避免中文/特殊字符乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LOG_PATH    = "layer_sensitivity.log"
REPORT_PATH = "layer_sensitivity_analysis.txt"

if len(sys.argv) >= 2:
    LOG_PATH = sys.argv[1]
if len(sys.argv) >= 3:
    REPORT_PATH = sys.argv[2]


# ═══════════════════════════════════════════════════════════════════════════════
#  1. 日志解析
# ═══════════════════════════════════════════════════════════════════════════════

def parse_log(path: str) -> list:
    """解析 layer_sensitivity.log，返回所有记录的 list[dict]。

    每条记录字段：
        scan_id, timestamp, repeats,
        rank, fss, delta_semantic, delta_logits, norm_drift,
        fault_type, layer_name, orig_val, inj_val, inject_delta
    """
    records = []
    scan_id   = 0
    scan_meta = {}

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()

        # ── 扫描头部 ────────────────────────────────────────────────
        if re.match(r"^层故障敏感度扫描", raw):
            scan_id += 1
            ts = raw.split("  ", 1)[-1].strip() if "  " in raw else ""
            scan_meta = {"scan_id": scan_id, "timestamp": ts}
            i += 1
            continue

        if raw.startswith("测试问题"):
            scan_meta["question"] = raw.split(":", 1)[-1].strip()
            i += 1
            continue
        if raw.startswith("FSS 权重"):
            scan_meta["fss_weights"] = raw.split(":", 1)[-1].strip()
            i += 1
            continue
        if raw.startswith("每层重复次数"):
            try:
                scan_meta["repeats"] = int(raw.split(":", 1)[-1].strip())
            except ValueError:
                scan_meta["repeats"] = 1
            i += 1
            continue

        # ── 数据行（以整数排名开头）─────────────────────────────────
        m = re.match(
            r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
            r"\s+(\S+)\s+(\S+)\s*$",
            raw,
        )
        if m:
            orig_val = inj_val = inject_delta = float("nan")

            # 紧接的第二行是注入详情
            if i + 1 < len(lines):
                detail = lines[i + 1]
                dm = re.search(
                    r"([+-][\d.e+\-]+)\s+([+-][\d.e+\-]+)\s+([\d.e+\-]+)",
                    detail,
                )
                if dm:
                    try:
                        orig_val     = float(dm.group(1))
                        inj_val      = float(dm.group(2))
                        inject_delta = float(dm.group(3))
                    except ValueError:
                        pass
                i += 1

            records.append({
                **scan_meta,
                "rank":           int(m.group(1)),
                "fss":            float(m.group(2)),
                "delta_semantic": float(m.group(3)),
                "delta_logits":   float(m.group(4)),
                "norm_drift":     float(m.group(5)),
                "fault_type":     m.group(6),
                "layer_name":     m.group(7),
                "orig_val":       orig_val,
                "inj_val":        inj_val,
                "inject_delta":   inject_delta,
            })

        i += 1

    return records


# ═══════════════════════════════════════════════════════════════════════════════
#  2. 层名分类
# ═══════════════════════════════════════════════════════════════════════════════

# 故障大类映射
FAULT_CATEGORY = {
    "bit_flip_low":    "位翻转",
    "bit_flip_mid":    "位翻转",
    "bit_flip_high":   "位翻转",
    "bit_flip_sign":   "位翻转",
    "stuck_at_zero":   "永久故障",
    "stuck_at_one":    "永久故障",
    "block_corrupt":   "存储块损坏",
    "kv_noise":        "KV Cache扰动",
    "kv_zero_region":  "KV Cache扰动",
    "persistent_bias": "老化/漂移",
}

# 结构大类中文名
STRUCT_LABEL = {
    "self_attn":    "标准注意力 (self_attn)",
    "linear_attn":  "线性注意力 (linear_attn)",
    "mlp":          "前馈网络 (mlp)",
    "layernorm":    "层归一化 (layernorm)",
    "whole_layer":  "整层 (whole_layer)",
    "other":        "其他",
}


def classify_layer(layer_name: str) -> dict:
    """提取层名中的结构信息。"""
    parts = layer_name.split(".")

    layer_num  = None
    layers_idx = None
    for j, p in enumerate(parts):
        if p == "layers" and j + 1 < len(parts):
            try:
                layer_num  = int(parts[j + 1])
                layers_idx = j
            except ValueError:
                pass
            break

    sub = parts[layers_idx + 2:] if layers_idx is not None else []

    if not sub:
        struct_type = "whole_layer"
    elif sub[0] == "self_attn":
        struct_type = "self_attn"
    elif sub[0] == "mlp":
        struct_type = "mlp"
    elif sub[0] == "linear_attn":
        struct_type = "linear_attn"
    elif sub[0] in ("input_layernorm", "post_attention_layernorm"):
        struct_type = "layernorm"
    else:
        struct_type = "other"

    component = ".".join(sub) if sub else "(整层)"

    # 深度分区
    if layer_num is None:
        depth_zone = "未知"
    elif layer_num <= 10:
        depth_zone = "浅层 (0-10)"
    elif layer_num <= 21:
        depth_zone = "中层 (11-21)"
    else:
        depth_zone = "深层 (22+)"

    return {
        "layer_num":   layer_num,
        "struct_type": struct_type,
        "component":   component,
        "depth_zone":  depth_zone,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. 统计工具
# ═══════════════════════════════════════════════════════════════════════════════

def safe_stats(values: list) -> dict:
    """计算一组数值的基础统计量（处理 nan/inf）。"""
    v = [x for x in values if math.isfinite(x)]
    if not v:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan"),
                "p25": float("nan"), "p75": float("nan")}
    v.sort()
    n   = len(v)
    mn  = sum(v) / n
    med = statistics.median(v)
    std = statistics.pstdev(v) if n > 1 else 0.0
    p25 = v[int(n * 0.25)]
    p75 = v[int(n * 0.75)]
    return {"n": n, "mean": mn, "median": med, "std": std,
            "min": v[0], "max": v[-1], "p25": p25, "p75": p75}


def iqr_outlier_threshold(values: list, k: float = 3.0) -> float:
    """返回 IQR 法离群上界（Q3 + k×IQR）。"""
    v = sorted(x for x in values if math.isfinite(x))
    if len(v) < 4:
        return float("inf")
    q1 = v[int(len(v) * 0.25)]
    q3 = v[int(len(v) * 0.75)]
    return q3 + k * (q3 - q1)


def corr_pearson(xs: list, ys: list) -> float:
    """Pearson 相关系数。"""
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    n  = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    dx  = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    dy  = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
    return num / (dx * dy + 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. 报告生成
# ═══════════════════════════════════════════════════════════════════════════════

def build_report(records: list) -> str:
    lines = []
    W = 100

    def H1(title):
        lines.append("\n" + "═" * W)
        lines.append(f"  {title}")
        lines.append("═" * W)

    def H2(title):
        lines.append(f"\n── {title} " + "─" * max(0, W - len(title) - 4))

    def row(*cols, widths=None):
        if widths:
            s = "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))
        else:
            s = "  ".join(str(c) for c in cols)
        lines.append(s)

    def sep():
        lines.append("─" * W)

    # ── 扫描汇总 ─────────────────────────────────────────────────────────────
    scan_ids = sorted({r.get("scan_id", 1) for r in records})
    H1(f"层故障敏感度日志统计分析报告  (共 {len(scan_ids)} 轮扫描，{len(records)} 条记录)")

    for sid in scan_ids:
        sr = [r for r in records if r.get("scan_id") == sid]
        if sr:
            lines.append(
                f"  扫描 #{sid}  时间={sr[0].get('timestamp','')}  "
                f"重复次数={sr[0].get('repeats',1)}  层数={len(sr)}"
            )

    # ── 汇总所有记录，为后续分析附加分类字段 ──────────────────────────────────
    for r in records:
        info = classify_layer(r["layer_name"])
        r.update(info)
        r["fault_cat"] = FAULT_CATEGORY.get(r["fault_type"], "未知")

    all_fss = [r["fss"] for r in records]
    outlier_thresh = iqr_outlier_threshold(all_fss)
    normal_records = [r for r in records if r["fss"] <= outlier_thresh]
    outlier_records = [r for r in records if r["fss"] > outlier_thresh]

    # ══════════════════════════════════════════════════════════════════════════
    H1("一、总览与异常检测")

    st = safe_stats(all_fss)
    lines.append(f"\n  全部 {st['n']} 条记录的 FSS 分布：")
    lines.append(f"    均值={st['mean']:.4f}  中位数={st['median']:.4f}  标准差={st['std']:.4f}")
    lines.append(f"    最小={st['min']:.4f}  最大={st['max']:.4e}  P25={st['p25']:.4f}  P75={st['p75']:.4f}")
    lines.append(f"\n  IQR 离群阈值（Q3 + 3×IQR）= {outlier_thresh:.4f}")
    lines.append(f"  正常层数={len(normal_records)}  离群层数={len(outlier_records)}")

    if outlier_records:
        H2("离群层列表（FSS 异常偏高，通常由 bit_flip_high 引起数值爆炸）")
        row("FSS", "故障类型", "注入点MAE", "层名", widths=[18, 22, 16, 50])
        sep()
        for r in sorted(outlier_records, key=lambda x: -x["fss"])[:20]:
            row(f"{r['fss']:.4e}", r["fault_type"],
                f"{r['inject_delta']:.4e}", r["layer_name"], widths=[18, 22, 16, 50])

    # 后续分析均使用剔除极端离群点后的正常记录
    recs = normal_records
    lines.append(f"\n  ※ 以下各维度分析均基于正常层（{len(recs)} 条，已剔除 {len(outlier_records)} 个极端离群点）。")

    # ══════════════════════════════════════════════════════════════════════════
    H1("二、按结构大类分析")

    struct_groups = defaultdict(list)
    for r in recs:
        struct_groups[r["struct_type"]].append(r)

    H2("各结构大类的 FSS 统计（按中位数降序）")
    col_w = [22, 6, 9, 9, 9, 9, 9, 9]
    row("结构大类", "N", "均值", "中位数", "最大值", "P75", "P25", "最小值", widths=col_w)
    sep()

    struct_stats = {}
    for st_type, grp in struct_groups.items():
        fss_vals = [r["fss"] for r in grp]
        s = safe_stats(fss_vals)
        struct_stats[st_type] = s
    for st_type, s in sorted(struct_stats.items(), key=lambda x: -x[1]["median"]):
        label = STRUCT_LABEL.get(st_type, st_type)
        row(label, s["n"], f"{s['mean']:.4f}", f"{s['median']:.4f}",
            f"{s['max']:.4f}", f"{s['p75']:.4f}", f"{s['p25']:.4f}", f"{s['min']:.4f}",
            widths=col_w)

    H2("各结构大类中 FSS 最高的层")
    for st_type, grp in sorted(struct_groups.items(),
                                key=lambda x: -struct_stats[x[0]]["median"]):
        top = max(grp, key=lambda r: r["fss"])
        label = STRUCT_LABEL.get(st_type, st_type)
        lines.append(
            f"  {label:<30}  FSS={top['fss']:.4f}  "
            f"fault={top['fault_type']:<22}  {top['layer_name']}"
        )

    H2("各结构大类的子指标对比（中位数）")
    col_w2 = [30, 12, 12, 12]
    row("结构大类", "Δsemantic", "Δlogits", "NormDrift", widths=col_w2)
    sep()
    for st_type in sorted(struct_stats, key=lambda x: -struct_stats[x]["median"]):
        grp = struct_groups[st_type]
        label = STRUCT_LABEL.get(st_type, st_type)
        med_sem = statistics.median([r["delta_semantic"] for r in grp if math.isfinite(r["delta_semantic"])])
        med_log = statistics.median([r["delta_logits"]   for r in grp if math.isfinite(r["delta_logits"])])
        med_nrm = statistics.median([r["norm_drift"]     for r in grp if math.isfinite(r["norm_drift"])])
        row(label, f"{med_sem:.4f}", f"{med_log:.4f}", f"{med_nrm:.4f}", widths=col_w2)

    # ══════════════════════════════════════════════════════════════════════════
    H1("三、按具体组件分析")

    comp_groups = defaultdict(list)
    for r in recs:
        comp_groups[r["component"]].append(r)

    # 只展示出现 ≥ 2 次的组件，按中位数 FSS 降序
    comp_stats = {}
    for comp, grp in comp_groups.items():
        comp_stats[comp] = safe_stats([r["fss"] for r in grp])

    H2("组件 FSS 中位数 Top-20（出现 ≥ 2 次）")
    col_w3 = [40, 6, 10, 10, 10]
    row("组件路径", "N", "中位数FSS", "最大FSS", "均值FSS", widths=col_w3)
    sep()
    top_comps = sorted(
        [(c, s) for c, s in comp_stats.items() if s["n"] >= 2],
        key=lambda x: -x[1]["median"]
    )[:20]
    for comp, s in top_comps:
        row(comp, s["n"], f"{s['median']:.4f}", f"{s['max']:.4f}", f"{s['mean']:.4f}",
            widths=col_w3)

    H2("组件 FSS 中位数 Bottom-10（影响最小，出现 ≥ 2 次）")
    row("组件路径", "N", "中位数FSS", "最大FSS", "均值FSS", widths=col_w3)
    sep()
    bot_comps = sorted(
        [(c, s) for c, s in comp_stats.items() if s["n"] >= 2],
        key=lambda x: x[1]["median"]
    )[:10]
    for comp, s in bot_comps:
        row(comp, s["n"], f"{s['median']:.4f}", f"{s['max']:.4f}", f"{s['mean']:.4f}",
            widths=col_w3)

    # ══════════════════════════════════════════════════════════════════════════
    H1("四、按层深度分析")

    depth_groups = defaultdict(list)
    for r in recs:
        depth_groups[r["depth_zone"]].append(r)

    H2("各深度区间 FSS 统计")
    col_w = [20, 6, 9, 9, 9, 9]
    row("深度区间", "N", "均值", "中位数", "最大值", "最小值", widths=col_w)
    sep()
    depth_order = ["浅层 (0-10)", "中层 (11-21)", "深层 (22+)", "未知"]
    for zone in depth_order:
        grp = depth_groups.get(zone, [])
        if not grp:
            continue
        s = safe_stats([r["fss"] for r in grp])
        row(zone, s["n"], f"{s['mean']:.4f}", f"{s['median']:.4f}",
            f"{s['max']:.4f}", f"{s['min']:.4f}", widths=col_w)

    H2("各深度区间中影响最大的结构类型（按中位数 FSS）")
    for zone in depth_order:
        grp = depth_groups.get(zone, [])
        if not grp:
            continue
        sub_struct = defaultdict(list)
        for r in grp:
            sub_struct[r["struct_type"]].append(r["fss"])
        ranked = sorted(
            [(st, statistics.median(v)) for st, v in sub_struct.items() if v],
            key=lambda x: -x[1]
        )
        lines.append(f"\n  {zone}：")
        for st_type, med in ranked:
            label = STRUCT_LABEL.get(st_type, st_type)
            lines.append(f"    {label:<30}  中位数FSS={med:.4f}")

    H2("FSS 与层序号的 Pearson 相关系数")
    for st_type in ("self_attn", "mlp", "linear_attn", "layernorm"):
        grp = struct_groups.get(st_type, [])
        valid = [(r["layer_num"], r["fss"]) for r in grp
                 if r["layer_num"] is not None and math.isfinite(r["fss"])]
        if len(valid) >= 3:
            c = corr_pearson([p[0] for p in valid], [p[1] for p in valid])
            label = STRUCT_LABEL.get(st_type, st_type)
            direction = "（深层影响更大）" if c > 0.1 else "（浅层影响更大）" if c < -0.1 else "（与深度无明显关系）"
            lines.append(f"  {label:<30}  r={c:+.4f}  {direction}")

    # ══════════════════════════════════════════════════════════════════════════
    H1("五、按注入故障类型分析")

    ft_groups = defaultdict(list)
    for r in recs:
        ft_groups[r["fault_type"]].append(r)

    H2("各故障类型 FSS 统计（按中位数降序）")
    col_w4 = [24, 6, 10, 10, 10, 10]
    row("故障类型", "N", "均值", "中位数", "最大值", "最小值", widths=col_w4)
    sep()
    ft_stats = {ft: safe_stats([r["fss"] for r in grp]) for ft, grp in ft_groups.items()}
    for ft, s in sorted(ft_stats.items(), key=lambda x: -x[1]["median"]):
        row(ft, s["n"], f"{s['mean']:.4f}", f"{s['median']:.4f}",
            f"{s['max']:.4f}", f"{s['min']:.4f}", widths=col_w4)

    H2("各故障类型偏好影响的结构大类")
    for ft, grp in sorted(ft_groups.items(), key=lambda x: -ft_stats[x[0]]["median"]):
        sub_struct = defaultdict(list)
        for r in grp:
            sub_struct[r["struct_type"]].append(r["fss"])
        top_st = max(sub_struct.items(), key=lambda x: statistics.median(x[1]))
        label = STRUCT_LABEL.get(top_st[0], top_st[0])
        lines.append(
            f"  {ft:<24}  主要影响结构：{label:<30}"
            f"  中位FSS={statistics.median(top_st[1]):.4f}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    H1("六、按故障大类分析")

    cat_groups = defaultdict(list)
    for r in recs:
        cat_groups[r["fault_cat"]].append(r)

    H2("各故障大类 FSS 统计")
    row("故障大类", "N", "均值", "中位数", "最大值", widths=[20, 6, 10, 10, 10])
    sep()
    cat_stats = {c: safe_stats([r["fss"] for r in g]) for c, g in cat_groups.items()}
    for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["median"]):
        row(cat, s["n"], f"{s['mean']:.4f}", f"{s['median']:.4f}",
            f"{s['max']:.4f}", widths=[20, 6, 10, 10, 10])

    H2("各故障大类的子指标对比（中位数）")
    row("故障大类", "Δsemantic", "Δlogits", "NormDrift", "注入点MAE", widths=[20, 12, 12, 12, 14])
    sep()
    for cat in sorted(cat_stats, key=lambda x: -cat_stats[x]["median"]):
        grp = cat_groups[cat]
        def _med(key):
            v = [r[key] for r in grp if math.isfinite(r.get(key, float("nan")))]
            return statistics.median(v) if v else float("nan")
        row(cat, f"{_med('delta_semantic'):.4f}", f"{_med('delta_logits'):.4f}",
            f"{_med('norm_drift'):.4f}", f"{_med('inject_delta'):.4e}",
            widths=[20, 12, 12, 12, 14])

    # ══════════════════════════════════════════════════════════════════════════
    H1("七、子指标相关性分析")

    valid_triples = [
        (r["delta_semantic"], r["delta_logits"], r["norm_drift"])
        for r in recs
        if all(math.isfinite(r[k]) for k in ("delta_semantic", "delta_logits", "norm_drift"))
    ]
    sem_vals = [t[0] for t in valid_triples]
    log_vals = [t[1] for t in valid_triples]
    nrm_vals = [t[2] for t in valid_triples]

    r_sl = corr_pearson(sem_vals, log_vals)
    r_sn = corr_pearson(sem_vals, nrm_vals)
    r_ln = corr_pearson(log_vals, nrm_vals)

    lines.append(f"\n  Pearson 相关系数（基于 {len(valid_triples)} 条正常记录）：")
    lines.append(f"    Δsemantic <-> Δlogits    : r = {r_sl:+.4f}")
    lines.append(f"    Δsemantic <-> NormDrift  : r = {r_sn:+.4f}")
    lines.append(f"    Δlogits   <-> NormDrift  : r = {r_ln:+.4f}")

    def _interp(r_val):
        if abs(r_val) >= 0.7:
            return "强相关"
        elif abs(r_val) >= 0.4:
            return "中等相关"
        elif abs(r_val) >= 0.2:
            return "弱相关"
        else:
            return "几乎无线性相关"

    lines.append(f"\n  解释：")
    lines.append(f"    Δsemantic <-> Δlogits    => {_interp(r_sl)}")
    lines.append(f"    Δsemantic <-> NormDrift  => {_interp(r_sn)}")
    lines.append(f"    Δlogits   <-> NormDrift  => {_interp(r_ln)}")

    H2("各结构大类子指标相关性")
    for st_type in ("self_attn", "mlp", "linear_attn", "layernorm"):
        grp = struct_groups.get(st_type, [])
        if len(grp) < 5:
            continue
        label = STRUCT_LABEL.get(st_type, st_type)
        sv = [r["delta_semantic"] for r in grp if math.isfinite(r["delta_semantic"])]
        lv = [r["delta_logits"]   for r in grp if math.isfinite(r["delta_logits"])]
        nv = [r["norm_drift"]     for r in grp if math.isfinite(r["norm_drift"])]
        min_len = min(len(sv), len(lv), len(nv))
        if min_len < 3:
            continue
        lines.append(
            f"  {label:<30}  "
            f"sem-log={corr_pearson(sv[:min_len], lv[:min_len]):+.3f}  "
            f"sem-norm={corr_pearson(sv[:min_len], nv[:min_len]):+.3f}  "
            f"log-norm={corr_pearson(lv[:min_len], nv[:min_len]):+.3f}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    H1("八、多轮扫描一致性（出现在多轮中的层）")

    if len(scan_ids) >= 2:
        layer_scan_fss = defaultdict(list)
        for r in records:
            layer_scan_fss[r["layer_name"]].append(r["fss"])

        multi_layers = {
            ln: vals for ln, vals in layer_scan_fss.items() if len(vals) >= 2
        }
        lines.append(f"\n  共 {len(multi_layers)} 个层出现在多轮扫描中。")

        if multi_layers:
            H2("多轮扫描中 FSS 最稳定的层（标准差最小，按均值降序 Top-20）")
            row("层名", "N", "均值FSS", "标准差", "CV%", widths=[55, 5, 10, 10, 8])
            sep()
            stable = sorted(
                multi_layers.items(),
                key=lambda x: (
                    -statistics.mean(v for v in x[1] if math.isfinite(v)),
                    statistics.pstdev(v for v in x[1] if math.isfinite(v))
                )
            )[:20]
            for ln, vals in stable:
                v = [x for x in vals if math.isfinite(x)]
                if not v:
                    continue
                m = sum(v) / len(v)
                s = statistics.pstdev(v)
                cv = s / (m + 1e-12) * 100
                row(ln, len(v), f"{m:.4f}", f"{s:.4f}", f"{cv:.1f}%", widths=[55, 5, 10, 10, 8])
    else:
        lines.append("\n  日志中仅含 1 轮扫描，无法进行多轮一致性分析。")
        lines.append("  建议多次运行后再对比，以验证结果的稳定性。")

    # ══════════════════════════════════════════════════════════════════════════
    H1("九、综合结论")

    # 最脆弱结构
    top_struct = max(struct_stats.items(), key=lambda x: x[1]["median"])
    top_struct_label = STRUCT_LABEL.get(top_struct[0], top_struct[0])

    # 最危险故障
    top_fault = max(ft_stats.items(), key=lambda x: x[1]["median"])

    # Top-5 层（正常记录）
    top5 = sorted(recs, key=lambda r: -r["fss"])[:5]

    lines.append(f"""
  【最脆弱结构类型】
    {top_struct_label}（中位数 FSS = {top_struct[1]['median']:.4f}）
    该类层注入故障后，对模型输出的语义和数值影响最为显著。

  【最危险故障类型】
    {top_fault[0]}（中位数 FSS = {top_fault[1]['median']:.4f}）
    该故障类型在正常层中造成的平均影响最高。

  【影响最大的 Top-5 正常层】""")

    for i, r in enumerate(top5, 1):
        label = STRUCT_LABEL.get(r["struct_type"], r["struct_type"])
        lines.append(
            f"    {i}. FSS={r['fss']:.4f}  [{label}]  "
            f"fault={r['fault_type']}  {r['layer_name']}"
        )

    lines.append(f"""
  【层深度影响规律】""")
    for zone in depth_order:
        grp = depth_groups.get(zone, [])
        if grp:
            med = statistics.median([r["fss"] for r in grp if math.isfinite(r["fss"])])
            lines.append(f"    {zone:<18}  中位数 FSS = {med:.4f}")

    lines.append("\n" + "═" * W)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. 可视化
# ═══════════════════════════════════════════════════════════════════════════════

def generate_plots(records: list, out_dir: str) -> list:
    """生成 6 张分析图，返回保存的文件路径列表。"""
    if not HAS_PLOT:
        print("[可视化] 未检测到 matplotlib，跳过绘图。")
        return []

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved = []

    # matplotlib 中文字体设置（Windows 用 SimHei；无中文字体则回退英文标签）
    try:
        plt.rcParams["font.family"] = "SimHei"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    normal_recs = [r for r in records
                   if math.isfinite(r["fss"]) and r["fss"] <= iqr_outlier_threshold(
                       [x["fss"] for x in records])]

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

    # ── 图1：结构大类 FSS 分布（箱线图）────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    struct_order = ["self_attn", "mlp", "linear_attn", "layernorm", "whole_layer"]
    struct_labels = [STRUCT_LABEL.get(s, s) for s in struct_order]
    data_boxes = []
    for st in struct_order:
        vals = [r["fss"] for r in normal_recs
                if r["struct_type"] == st and math.isfinite(r["fss"])]
        data_boxes.append(vals)
    bp = ax.boxplot(data_boxes, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(struct_labels) + 1))
    ax.set_xticklabels(struct_labels, fontsize=9)
    ax.set_ylabel("FSS")
    ax.set_title("各结构大类 FSS 分布（箱线图，已剔除极端离群点）")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    p = str(Path(out_dir) / "01_struct_type_boxplot.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # ── 图2：故障类型 FSS 中位数（水平条形图）──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ft_med = {}
    for r in normal_recs:
        ft_med.setdefault(r["fault_type"], []).append(r["fss"])
    ft_sorted = sorted(ft_med.items(),
                       key=lambda x: statistics.median(x[1]))
    labels = [x[0] for x in ft_sorted]
    medians = [statistics.median(x[1]) for x in ft_sorted]
    bars = ax.barh(labels, medians,
                   color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
                   alpha=0.8)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("FSS 中位数")
    ax.set_title("各故障类型 FSS 中位数对比")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    p = str(Path(out_dir) / "02_fault_type_bar.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # ── 图3：故障大类 × 结构大类热力图（中位数 FSS）────────────────────
    cat_order   = ["位翻转", "永久故障", "存储块损坏", "KV Cache扰动", "老化/漂移"]
    struct_hm   = ["self_attn", "mlp", "linear_attn", "layernorm", "whole_layer"]
    matrix = []
    for cat in cat_order:
        row_vals = []
        for st in struct_hm:
            vals = [r["fss"] for r in normal_recs
                    if r["fault_cat"] == cat and r["struct_type"] == st
                    and math.isfinite(r["fss"])]
            row_vals.append(statistics.median(vals) if vals else float("nan"))
        matrix.append(row_vals)

    mat_arr = [[v if not math.isnan(v) else 0 for v in row] for row in matrix]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mat_arr, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(struct_hm)))
    ax.set_xticklabels([STRUCT_LABEL.get(s, s) for s in struct_hm], fontsize=8)
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels(cat_order, fontsize=9)
    for i in range(len(cat_order)):
        for j in range(len(struct_hm)):
            v = matrix[i][j]
            text = f"{v:.4f}" if not math.isnan(v) else "N/A"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="black" if mat_arr[i][j] < 0.08 else "white")
    plt.colorbar(im, ax=ax, label="FSS 中位数")
    ax.set_title("故障大类 × 结构大类 FSS 中位数热力图")
    fig.tight_layout()
    p = str(Path(out_dir) / "03_fault_struct_heatmap.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # ── 图4：FSS 随层序号变化（按结构大类分组折线图）─────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, st in enumerate(["self_attn", "mlp", "linear_attn", "layernorm"]):
        pts = sorted(
            [(r["layer_num"], r["fss"]) for r in normal_recs
             if r["struct_type"] == st and r["layer_num"] is not None
             and math.isfinite(r["fss"])],
            key=lambda x: x[0]
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # 按层号取均值（同一层可能有多个子模块）
        layer_means = {}
        for x, y in zip(xs, ys):
            layer_means.setdefault(x, []).append(y)
        sx = sorted(layer_means.keys())
        sy = [statistics.mean(layer_means[x]) for x in sx]
        label = STRUCT_LABEL.get(st, st)
        ax.plot(sx, sy, marker="o", markersize=3, linewidth=1.5,
                label=label, color=COLORS[idx], alpha=0.85)
    ax.set_xlabel("层序号")
    ax.set_ylabel("FSS（同层均值）")
    ax.set_title("FSS 随层序号变化（按结构大类）")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    p = str(Path(out_dir) / "04_fss_by_layer_depth.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # ── 图5：Top-20 组件 FSS 中位数（水平条形图）───────────────────────
    comp_med = {}
    for r in normal_recs:
        comp_med.setdefault(r["component"], []).append(r["fss"])
    top_comps = sorted(
        [(c, vals) for c, vals in comp_med.items() if len(vals) >= 2],
        key=lambda x: -statistics.median(x[1])
    )[:20]
    fig, ax = plt.subplots(figsize=(10, 7))
    comp_labels = [x[0] for x in top_comps][::-1]
    comp_vals   = [statistics.median(x[1]) for x in top_comps][::-1]
    bars = ax.barh(comp_labels, comp_vals,
                   color=COLORS[0], alpha=0.75)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
    ax.set_xlabel("FSS 中位数")
    ax.set_title("组件 FSS 中位数 Top-20（≥2 次出现）")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    p = str(Path(out_dir) / "05_component_top20_bar.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # ── 图6：三子指标联合散点图（Δsemantic vs Δlogits，颜色=结构大类）──
    fig, ax = plt.subplots(figsize=(8, 6))
    struct_colors = {
        "self_attn":   COLORS[0],
        "mlp":         COLORS[1],
        "linear_attn": COLORS[2],
        "layernorm":   COLORS[3],
        "whole_layer": COLORS[4],
    }
    for st, color in struct_colors.items():
        pts = [(r["delta_semantic"], r["delta_logits"])
               for r in normal_recs
               if r["struct_type"] == st
               and math.isfinite(r["delta_semantic"])
               and math.isfinite(r["delta_logits"])]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=color, label=STRUCT_LABEL.get(st, st),
                   alpha=0.5, s=20, edgecolors="none")
    ax.set_xlabel("Δsemantic")
    ax.set_ylabel("Δlogits")
    ax.set_title("Δsemantic vs Δlogits 散点图（按结构大类着色）")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    p = str(Path(out_dir) / "06_semantic_vs_logits_scatter.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    return saved


# ═══════════════════════════════════════════════════════════════════════════════
#  6. 入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log_file = Path(LOG_PATH)
    if not log_file.exists():
        print(f"[错误] 找不到日志文件：{LOG_PATH}")
        sys.exit(1)

    print(f"[分析] 正在解析：{LOG_PATH} ...")
    records = parse_log(str(log_file))
    if not records:
        print("[错误] 日志中未解析到任何数据记录，请检查日志格式。")
        sys.exit(1)
    print(f"[分析] 共解析 {len(records)} 条记录，来自 "
          f"{len({r.get('scan_id',1) for r in records})} 轮扫描。")

    print("[分析] 正在生成报告...")
    report = build_report(records)

    # 写报告文件
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[分析] 报告已写入：{REPORT_PATH}\n")

    # 生成可视化图表
    saved_plots = generate_plots(records, PLOT_DIR)
    if saved_plots:
        print(f"[可视化] 共生成 {len(saved_plots)} 张图表，保存至 {PLOT_DIR}/")
        for p in saved_plots:
            print(f"         {p}")
        print()

    # 同时打印到控制台
    print(report)


if __name__ == "__main__":
    main()
