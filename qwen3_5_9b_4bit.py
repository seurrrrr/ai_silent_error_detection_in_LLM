"""
本地部署 Qwen3.5-9B，采用 4-bit 量化推理
依赖安装：
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers>=4.45.0" accelerate bitsandbytes

模型下载（国内加速）：
    export HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download Qwen/Qwen3.5-9B --local-dir /home/qiyuan/models/Qwen3.5-9B

下载完成后将 MODEL_PATH 改为本地目录路径，例如：
    MODEL_PATH = "/home/qiyuan/models/Qwen3.5-9B"
"""

import importlib.metadata
import logging
import os
import sys
import warnings

# 屏蔽 flash-linear-attention / causal-conv1d 未安装的警告（可选加速库，不影响功能）
warnings.filterwarnings("ignore", message=".*fast path.*not available.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _check_transformers_version(min_version: str = "4.45.0") -> None:
    """检查 transformers 版本，过低时给出明确提示并退出。"""
    from packaging.version import Version

    installed = importlib.metadata.version("transformers")
    if Version(installed) < Version(min_version):
        print(
            f"[错误] 当前 transformers 版本为 {installed}，"
            f"需要 >= {min_version} 才能支持 Qwen2Tokenizer。\n"
            f"请执行：pip install -U 'transformers>={min_version}'"
        )
        sys.exit(1)


# 本地模型路径（填写后优先使用本地）；留空则从 HuggingFace 下载
MODEL_PATH = "/home/qiyuan/models/Qwen3.5-9B"
# HuggingFace 模型 ID（MODEL_PATH 为空时使用）
MODEL_ID   = "Qwen/Qwen3.5-9B"


def _resolve_model_source() -> str:
    """优先使用本地路径；若未设置则使用 HF 镜像下载。"""
    if MODEL_PATH and os.path.isdir(MODEL_PATH):
        print(f"[本地模式] 从本地路径加载：{MODEL_PATH}")
        return MODEL_PATH
    # 自动启用 HF 镜像加速（国内网络）
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"[镜像加速] 已自动设置 HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    print(f"[远程模式] 将从 HuggingFace 下载：{MODEL_ID}")
    return MODEL_ID


def build_quantization_config() -> BitsAndBytesConfig:
    """构建 4-bit 量化配置（NF4 格式 + double quantization）。"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 量化类型，精度损失最小
        bnb_4bit_use_double_quant=True,      # 二次量化，进一步降低显存
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model_and_tokenizer(model_id: str = ""):
    """加载分词器与 4-bit 量化模型。"""
    model_id = model_id or _resolve_model_source()
    print(f"正在加载分词器：{model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    quant_config = build_quantization_config()

    print(f"正在加载模型（4-bit 量化）：{model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="cuda",          # 自动分配到可用 GPU；无 GPU 时回退 CPU
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    print("模型加载完毕。")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    user_input: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """给定用户输入，返回模型生成的文本。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_input},
    ]

    # 使用 chat template 构建输入
    # enable_thinking=False：关闭 Qwen3 思考模式，直接输出答案，节省 token
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只截取新生成的 token
    new_token_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    # 过滤 Qwen3 思考块（<think>...</think> 或纯文本推理段落）
    # 若模型未开启思考模式，此步无影响
    if "<think>" in response and "</think>" in response:
        response = response[response.rfind("</think>") + len("</think>"):].strip()

    return response


def interactive_chat(model, tokenizer) -> None:
    """简易命令行对话循环。输入 'exit' 或 'quit' 退出。"""
    print("\n=== Qwen3.5-9B 4-bit 量化对话（输入 exit 退出）===\n")
    while True:
        user_input = input("用户：").strip()
        if user_input.lower() in {"exit", "quit", ""}:
            print("已退出对话。")
            break
        response = generate_response(model, tokenizer, user_input)
        print(f"模型：{response}\n")


import random
import dataclasses
from enum import Enum


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  故障分类体系（依据：大语言模型中常见硬件故障—影响—模拟方法）
#
#  一、瞬态位翻转（Transient Bit Flip）
#      BIT_FLIP_LOW   — 低有效位翻转（尾数 bit0-3），最隐蔽，接近 ECC 漏检
#      BIT_FLIP_MID   — 中间位翻转（尾数 bit8-15），中等扰动
#      BIT_FLIP_HIGH  — 高有效位翻转（指数 bit23-30），较大数值跳变
#      BIT_FLIP_SIGN  — 符号位翻转（bit31），数值变号，方向完全反转
#
#  二、永久故障（Stuck-at Faults）
#      STUCK_AT_ZERO  — 通道永久粘滞为 0（stuck-at-0），模拟 SRAM 单元失效
#      STUCK_AT_ONE   — 通道永久粘滞为最大量化值（stuck-at-1）
#
#  三、存储块损坏（Block-level Memory Corruption）
#      BLOCK_CORRUPT  — 连续内存块整体扰动，模拟 HBM row/cache line 损坏
#
#  四、KV Cache 扰动（KV Cache Perturbation）
#      KV_NOISE       — 对激活值加高斯噪声，模拟 KV cache 随机污染
#      KV_ZERO_REGION — 对特定区域归零，模拟 KV cache 局部坏块
#
#  五、老化/漂移（Aging / Gradual Drift）
#      PERSISTENT_BIAS — 全层加微小固定偏置，模拟器件老化长期累积偏移
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FaultType(Enum):
    # 一、瞬态位翻转
    BIT_FLIP_LOW   = "bit_flip_low"    # 低位翻转（bit0-3），偏移 ~1e-7，最隐蔽
    BIT_FLIP_MID   = "bit_flip_mid"    # 中间位翻转（bit8-15），偏移 ~1e-3
    BIT_FLIP_HIGH  = "bit_flip_high"   # 高位翻转（bit23-30），偏移较大
    BIT_FLIP_SIGN  = "bit_flip_sign"   # 符号位翻转（bit31），数值取反
    # 二、永久故障
    STUCK_AT_ZERO  = "stuck_at_zero"   # 通道永久粘滞为 0
    STUCK_AT_ONE   = "stuck_at_one"    # 通道永久粘滞为正最大量化值
    # 三、存储块损坏
    BLOCK_CORRUPT  = "block_corrupt"   # 连续内存块扰动
    # 四、KV Cache 扰动
    KV_NOISE       = "kv_noise"        # 加高斯噪声污染激活
    KV_ZERO_REGION = "kv_zero_region"  # 局部区域归零
    # 五、老化/漂移
    PERSISTENT_BIAS = "persistent_bias"  # 全层固定小偏置累积


# 故障来源分类标签（用于日志）
FAULT_SOURCE = {
    FaultType.BIT_FLIP_LOW:    "瞬态位翻转",
    FaultType.BIT_FLIP_MID:    "瞬态位翻转",
    FaultType.BIT_FLIP_HIGH:   "瞬态位翻转",
    FaultType.BIT_FLIP_SIGN:   "瞬态位翻转",
    FaultType.STUCK_AT_ZERO:   "永久故障",
    FaultType.STUCK_AT_ONE:    "永久故障",
    FaultType.BLOCK_CORRUPT:   "存储块损坏",
    FaultType.KV_NOISE:        "KV Cache扰动",
    FaultType.KV_ZERO_REGION:  "KV Cache扰动",
    FaultType.PERSISTENT_BIAS: "老化/漂移",
}


@dataclasses.dataclass
class FaultEvent:
    layer_name:  str
    fault_type:  FaultType
    fault_source: str        # 故障大类（中文）
    position:    tuple
    original:    float
    injected:    float
    delta:       float       # 均值绝对误差


def register_fault_injection(
    model,
    fault_prob:       float = 0.20,
    fault_types:      list  = None,
    layer_filter:     str   = "layers",
    # 位翻转参数
    bit_flip_low_bits:  int   = 3,    # 低位翻转范围 [0, 2^n-1]
    bit_flip_mid_bits:  tuple = (8, 15),
    bit_flip_high_bits: tuple = (23, 30),
    # 永久故障参数
    stuck_ratio:        float = 0.01,  # 粘滞通道占比
    # 存储块参数
    block_size_ratio:   float = 0.02,  # 连续块长度占总元素比例
    block_noise_std:    float = 0.1,   # 块内扰动强度（相对标准差）
    # KV Cache 参数
    kv_noise_std_ratio: float = 0.02,  # 噪声标准差 = 均值 × ratio
    kv_zero_ratio:      float = 0.05,  # 归零区域比例
    # 老化参数
    bias_ratio:         float = 0.01,  # 固定偏置 = 均值 × ratio
    log_events:         bool  = True,
    seed:               int   = None,
) -> tuple:
    """为模型各层注册静默故障注入 hook（依据文献分类体系）。

    所有故障均不产生 NaN/Inf，模拟真实硬件静默错误（SDC）。
    故障分五大类：瞬态位翻转 / 永久故障 / 存储块损坏 / KV Cache 扰动 / 老化漂移。

    Returns:
        (handles, event_log)
    """
    import struct

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    fault_types = fault_types or list(FaultType)
    handles:   list = []
    event_log: list = []

    YELLOW = "\033[93m"
    RESET  = "\033[0m"

    # ── 位翻转工具 ────────────────────────────────────────────────────
    def _flip_bits(val: float, bit_range: tuple) -> float:
        """在 float32 的指定位范围内随机翻转若干位，保证不产生 NaN/Inf。"""
        lo, hi = bit_range
        packed = struct.pack("f", val)
        bits   = int.from_bytes(packed, "little")
        flip_bit = random.randint(lo, hi)
        bits ^= (1 << flip_bit)
        result = struct.unpack("f", bits.to_bytes(4, "little"))[0]
        if result != result or abs(result) > 1e15:   # NaN / Inf 保护
            return val
        return result

    def _apply(tf: torch.Tensor, ft: FaultType):
        """对 float32 张量施加指定故障，返回 (修改后张量, orig_val, inj_val, pos)。"""
        orig_val = inj_val = 0.0
        pos      = ()

        # ── 瞬态位翻转 ──────────────────────────────────────────────
        if ft in (FaultType.BIT_FLIP_LOW, FaultType.BIT_FLIP_MID,
                  FaultType.BIT_FLIP_HIGH, FaultType.BIT_FLIP_SIGN):
            ranges = {
                FaultType.BIT_FLIP_LOW:  (0,  bit_flip_low_bits - 1),
                FaultType.BIT_FLIP_MID:  bit_flip_mid_bits,
                FaultType.BIT_FLIP_HIGH: bit_flip_high_bits,
                FaultType.BIT_FLIP_SIGN: (31, 31),
            }
            flat     = tf.reshape(-1)
            idx      = random.randint(0, flat.numel() - 1)
            orig_val = flat[idx].item()
            inj_val  = _flip_bits(orig_val, ranges[ft])
            flat[idx] = inj_val
            pos = (idx,)

        # ── 永久故障 ─────────────────────────────────────────────────
        elif ft == FaultType.STUCK_AT_ZERO:
            if tf.dim() >= 2:
                n_ch    = tf.shape[-1]
                n_stuck = max(1, int(n_ch * stuck_ratio))
                idxs    = random.sample(range(n_ch), n_stuck)
                orig_val = tf[..., idxs].abs().mean().item()
                tf[..., idxs] = 0.0
                inj_val = 0.0
                pos     = tuple(idxs[:4])

        elif ft == FaultType.STUCK_AT_ONE:
            if tf.dim() >= 2:
                n_ch    = tf.shape[-1]
                n_stuck = max(1, int(n_ch * stuck_ratio))
                idxs    = random.sample(range(n_ch), n_stuck)
                max_val = tf.abs().max().item()
                orig_val = tf[..., idxs].abs().mean().item()
                tf[..., idxs] = max_val
                inj_val  = max_val
                pos      = tuple(idxs[:4])

        # ── 存储块损坏 ───────────────────────────────────────────────
        elif ft == FaultType.BLOCK_CORRUPT:
            flat       = tf.reshape(-1)
            n          = flat.numel()
            block_len  = max(1, int(n * block_size_ratio))
            start      = random.randint(0, n - block_len)
            std        = flat.abs().mean().item() * block_noise_std
            orig_val   = flat[start:start + block_len].abs().mean().item()
            noise      = torch.randn(block_len, device=tf.device, dtype=tf.dtype) * std
            flat[start:start + block_len] = flat[start:start + block_len] + noise
            inj_val    = flat[start:start + block_len].abs().mean().item()
            pos        = (start, start + block_len)

        # ── KV Cache 扰动 ────────────────────────────────────────────
        elif ft == FaultType.KV_NOISE:
            std      = tf.abs().mean().item() * kv_noise_std_ratio
            orig_val = tf.abs().mean().item()
            tf.add_(torch.randn_like(tf) * std)
            inj_val  = tf.abs().mean().item()
            pos      = ("全张量",)

        elif ft == FaultType.KV_ZERO_REGION:
            if tf.dim() >= 2:
                n_seq    = tf.shape[-2] if tf.dim() >= 3 else tf.shape[0]
                n_zero   = max(1, int(n_seq * kv_zero_ratio))
                start    = random.randint(0, max(0, n_seq - n_zero))
                orig_val = tf[..., start:start + n_zero, :].abs().mean().item()
                tf[..., start:start + n_zero, :] = 0.0
                inj_val  = 0.0
                pos      = (start, start + n_zero)

        # ── 老化/漂移 ────────────────────────────────────────────────
        elif ft == FaultType.PERSISTENT_BIAS:
            bias     = tf.abs().mean().item() * bias_ratio
            orig_val = tf.abs().mean().item()
            tf.add_(bias)
            inj_val  = tf.abs().mean().item()
            pos      = ("全张量",)

        return tf, orig_val, inj_val, pos

    def _make_fault_hook(name: str):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not hasattr(tensor, "shape") or tensor.numel() == 0:
                return output
            if random.random() > fault_prob:
                return output

            t  = (output[0].clone() if isinstance(output, (tuple, list)) else output.clone())
            tf = t.float()
            ft = random.choice(fault_types)

            tf, orig_val, inj_val, pos = _apply(tf, ft)

            t_out = tf.to(t.dtype)
            delta = (t_out.float() - t.float()).abs().mean().item()

            ev = FaultEvent(
                layer_name=name,
                fault_type=ft,
                fault_source=FAULT_SOURCE[ft],
                position=pos,
                original=orig_val,
                injected=inj_val,
                delta=delta,
            )
            event_log.append(ev)

            if log_events:
                print(
                    f"{YELLOW}[静默故障] [{ev.fault_source}] {ft.value:18s}  "
                    f"层={name:40s}  位置={str(pos):18s}  "
                    f"{orig_val:+.6f}→{inj_val:+.6f}  Δ={delta:.2e}{RESET}"
                )

            return (t_out, *output[1:]) if isinstance(output, (tuple, list)) else t_out

        return hook

    count = 0
    for name, module in model.named_modules():
        if layer_filter in name:
            h = module.register_forward_hook(_make_fault_hook(name))
            handles.append(h)
            count += 1

    print(f"\n[故障注入] 已注册 {count} 个 hook（过滤词='{layer_filter}'，触发概率={fault_prob:.0%}）")
    print(f"[故障注入] 故障类型：{[ft.value for ft in fault_types]}\n")
    return handles, event_log


def register_health_hooks(
    model,
    nan_inf_threshold: float = 1e4,
    norm_spike_ratio:  float = 10.0,
    layer_filter:      str   = "layers",
) -> list:
    """为模型各层注册数值健康检查 hook，检测静默内存故障导致的数据异常。

    检测项：
        1. NaN / Inf：直接报警
        2. 最大绝对值超过阈值：可能是 bit flip 导致的极值
        3. 层间范数突变（当前层范数 / 上一层范数 > spike_ratio）：数值突变信号

    Args:
        model:              已加载的模型。
        nan_inf_threshold:  输出张量最大绝对值超过此值时报警（默认 1e4）。
        norm_spike_ratio:   相邻层范数之比超过此值时报警（默认 10x）。
        layer_filter:       只监控名称包含此关键词的层（默认 "layers"）。

    Returns:
        handles 列表，推理结束后调用 h.remove() 清理。
    """
    handles   = []
    prev_norm = [None]   # 用列表包装以便闭包修改

    def _make_health_hook(name: str):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not hasattr(tensor, "shape"):
                return

            # 转 float32 再检查，避免 bf16 溢出误判
            t = tensor.detach().float()

            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            max_abs = t.abs().max().item()
            norm    = t.norm().item()

            alerts = []
            if has_nan:
                alerts.append("NaN 检测到！")
            if has_inf:
                alerts.append("Inf 检测到！")
            if max_abs > nan_inf_threshold:
                alerts.append(f"极值异常 max_abs={max_abs:.2e} > {nan_inf_threshold:.0e}")
            if prev_norm[0] is not None and prev_norm[0] > 0:
                ratio = norm / prev_norm[0]
                if ratio > norm_spike_ratio or ratio < 1.0 / norm_spike_ratio:
                    alerts.append(f"范数突变 {prev_norm[0]:.2f} → {norm:.2f} (x{ratio:.1f})")

            prev_norm[0] = norm

            status = "⚠ " + " | ".join(alerts) if alerts else "✓"
            print(
                f"[健康检查] {name:55s}  "
                f"norm={norm:10.2f}  max_abs={max_abs:.2e}  {status}"
            )
        return hook

    count = 0
    for name, module in model.named_modules():
        if layer_filter in name:
            h = module.register_forward_hook(_make_health_hook(name))
            handles.append(h)
            count += 1

    print(f"[健康监控] 已注册 {count} 个检测 hook（过滤词='{layer_filter}'）")
    print(f"[健康监控] 阈值：极值>{nan_inf_threshold:.0e}  范数突变>{norm_spike_ratio}x\n")
    return handles


def print_model_structure(model, max_depth: int = 3) -> None:
    """打印模型层级结构、参数量及所在设备。

    Args:
        model: 已加载的模型。
        max_depth: 展示的最大层级深度（默认 3 层）。
    """
    def _fmt_params(n: int) -> str:
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        if n >= 1e6:
            return f"{n/1e6:.2f}M"
        if n >= 1e3:
            return f"{n/1e3:.2f}K"
        return str(n)

    def _device(module) -> str:
        try:
            return str(next(module.parameters()).device)
        except StopIteration:
            return "—"

    def _walk(module, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        children = list(module.named_children())
        for i, (name, child) in enumerate(children):
            is_last   = (i == len(children) - 1)
            connector = "└─" if is_last else "├─"
            params    = sum(p.numel() for p in child.parameters())
            dtype_set = {p.dtype for p in child.parameters()}
            dtype_str = "/".join(str(d).split(".")[-1] for d in dtype_set) if dtype_set else "—"
            print(
                f"{prefix}{connector} {name}  "
                f"[{child.__class__.__name__}]  "
                f"params={_fmt_params(params)}  "
                f"dtype={dtype_str}  "
                f"device={_device(child)}"
            )
            next_prefix = prefix + ("   " if is_last else "│  ")
            _walk(child, next_prefix, depth + 1)

    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"模型结构（最大展示深度={max_depth}）")
    print(f"总参数量：{_fmt_params(total)}")
    print(f"{'='*70}")
    print(f"{model.__class__.__name__}")
    _walk(model, "", 1)
    print(f"{'='*70}\n")


def register_hooks(model, layer_name_filter: str = "") -> list:
    """为匹配名称的层注册 forward hook，实时打印输入/输出张量形状。

    Args:
        model: 已加载的模型。
        layer_name_filter: 只对名称包含此字符串的层注册 hook；为空则注册所有层。

    Returns:
        handles 列表，调用 handle.remove() 可取消 hook。
    """
    handles = []

    def _make_hook(name):
        def hook(module, inputs, output):
            in_shapes  = [tuple(x.shape) for x in inputs if hasattr(x, "shape")]
            out_shape  = tuple(output.shape) if hasattr(output, "shape") else type(output).__name__
            print(f"[HOOK] {name:60s}  in={in_shapes}  out={out_shape}")
        return hook

    for name, module in model.named_modules():
        if layer_name_filter in name:
            h = module.register_forward_hook(_make_hook(name))
            handles.append(h)

    print(f"已注册 {len(handles)} 个 forward hook（过滤词='{layer_name_filter}'）")
    return handles


def _get_hidden_states(text: str, model, tokenizer) -> "torch.Tensor":
    """返回模型最后一层隐状态的 mean-pooling 归一化向量（用于语义相似度）。"""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[-1].mean(dim=1).squeeze(0).float()
    return hidden / (hidden.norm() + 1e-9)


def _get_logits_vector(text: str, model, tokenizer) -> "torch.Tensor":
    """返回最后一个 token 位置的 logits 归一化向量（用于 logits 余弦相似度）。"""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()   # [vocab_size]
    return logits / (logits.norm() + 1e-9)


def _topk_tokens(text: str, model, tokenizer, k: int = 10) -> list:
    """返回最后一个 token 位置 top-k 预测 token 列表。"""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    topk_ids = logits.topk(k).indices.tolist()
    return tokenizer.convert_ids_to_tokens(topk_ids)


def _rouge_l(a: str, b: str) -> float:
    """字符级 ROUGE-L（F1，基于最长公共子序列）。"""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[la][lb]
    p = lcs / lb
    r = lcs / la
    return 2 * p * r / (p + r + 1e-9)


def _compare_outputs(baseline: str, fault: str, tokenizer=None, model=None) -> dict:
    """多维度评估基准输出与故障输出的差异。

    评估维度对应文献推荐三层：
    ┌─────────────┬────────────────────────────────────────────────────┐
    │ 数值层       │ hidden_states_cosine（隐状态余弦）                  │
    │ （文献[3][4]）│ logits_cosine（logits 余弦）                       │
    │             │ topk_overlap（top-10 预测词重叠率）                  │
    ├─────────────┼────────────────────────────────────────────────────┤
    │ 语义层       │ semantic_cosine（语义向量余弦，主判定依据）           │
    │ （文献[2][3]）│ rouge_l（字符级 ROUGE-L，考虑词序）                 │
    │             │ token_jaccard（词汇集合 Jaccard，辅助参考）           │
    ├─────────────┼────────────────────────────────────────────────────┤
    │ 结构层       │ len_base / len_fault（输出长度）                     │
    │             │ first_sentence_same（首句一致性）                    │
    └─────────────┴────────────────────────────────────────────────────┘
    """
    a = baseline.strip()
    b = fault.strip()

    # ── 数值层：隐状态余弦 + logits 余弦 + top-k 重叠 ───────────────
    hidden_cosine = logits_cosine = topk_overlap = None
    if model is not None and tokenizer is not None:
        try:
            ha = _get_hidden_states(a or "空", model, tokenizer)
            hb = _get_hidden_states(b or "空", model, tokenizer)
            hidden_cosine = float((ha * hb).sum().clamp(-1, 1).item())

            la_ = _get_logits_vector(a or "空", model, tokenizer)
            lb_ = _get_logits_vector(b or "空", model, tokenizer)
            logits_cosine = float((la_ * lb_).sum().clamp(-1, 1).item())

            topk_a = set(_topk_tokens(a or "空", model, tokenizer))
            topk_b = set(_topk_tokens(b or "空", model, tokenizer))
            topk_overlap = len(topk_a & topk_b) / max(len(topk_a | topk_b), 1)
        except Exception:
            pass

    # ── 语义层：ROUGE-L + Jaccard ────────────────────────────────────
    rouge = _rouge_l(a, b)
    set_a = set(a.split())
    set_b = set(b.split())
    union = set_a | set_b
    token_jaccard = len(set_a & set_b) / len(union) if union else 1.0

    # ── 结构层：首句 ─────────────────────────────────────────────────
    def _first_sentence(text: str) -> str:
        for sep in ("。", ".", "！", "!", "？", "?", "\n"):
            idx = text.find(sep)
            if idx != -1:
                return text[: idx + 1].strip()
        return text[:80].strip()
    first_sentence_same = _first_sentence(a) == _first_sentence(b)

    # ── 综合判定（优先隐状态余弦，降级时用语义余弦，再降级用 ROUGE-L）
    score = hidden_cosine if hidden_cosine is not None else rouge
    if score >= 0.97:
        verdict = "数值/语义无差异（故障被模型完全吸收）"
    elif score >= 0.90:
        verdict = "语义高度相似（表达略有不同，含义一致）"
    elif score >= 0.75:
        verdict = "语义基本相似（部分表述偏移，核心含义保留）"
    elif score >= 0.55:
        verdict = "语义中等偏离（故障影响了部分语义内容）"
    else:
        verdict = "语义显著偏离（故障明显改变了输出含义）"

    return {
        # 数值层
        "hidden_cosine":       hidden_cosine,
        "logits_cosine":       logits_cosine,
        "topk_overlap":        topk_overlap,
        # 语义层
        "rouge_l":             rouge,
        "token_jaccard":       token_jaccard,
        # 结构层
        "len_base":            len(a),
        "len_fault":           len(b),
        "first_sentence_same": first_sentence_same,
        # 综合
        "verdict":             verdict,
    }


def run_single_fault_experiment(
    model,
    tokenizer,
    question:           str   = "请用简洁的语言解释什么是量子纠缠。",
    fault_types:        list  = None,
    layer_filter:       str   = "layers",
    log_path:           str   = "fault_experiment.log",
    max_new_tokens:     int   = 512,
    # 位翻转参数
    bit_flip_low_bits:  int   = 3,
    bit_flip_mid_bits:  tuple = (8, 15),
    bit_flip_high_bits: tuple = (23, 30),
    num_flips:          int   = 1,     # 每次翻转的 bit 数量（>1 可显著放大影响）
    flip_count:         int   = 1,     # 同一张量中受影响的元素数量
    # 永久故障参数
    stuck_ratio:        float = 0.01,
    # 存储块参数
    block_size_ratio:   float = 0.02,
    block_noise_std:    float = 0.10,
    # KV Cache 参数
    kv_noise_std_ratio: float = 0.02,
    kv_zero_ratio:      float = 0.05,
    # 老化参数
    bias_ratio:         float = 0.01,
    seed:               int   = None,
) -> None:
    """单点故障传播实验（依据文献分类体系）：
      1. 先跑一次无故障推理，记录每层基准范数和最终输出；
      2. 随机选定一层，注入一个静默故障（五大类之一），再跑一次推理，
         记录故障层之后每层的范数偏差及最终输出变化；
      3. 按文献三层评估体系（数值层/语义层/结构层）写入日志。
    """
    import struct
    import datetime

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    fault_types = fault_types or list(FaultType)

    # ── 共享状态 ────────────────────────────────────────────────────────
    baseline_norms: dict = {}
    probed_norms:   dict = {}
    fault_event:    list = [None]
    fault_done:     list = [False]
    fault_layer:    list = [""]

    # ── 辅助：位翻转（支持多位同时翻转）────────────────────────────────
    def _flip_bits(val: float, lo: int, hi: int, n: int = 1) -> float:
        """翻转 float32 的 [lo, hi] 范围内随机 n 个 bit，防止产生 NaN/Inf。"""
        packed = struct.pack("f", val)
        bits   = int.from_bytes(packed, "little")
        positions = random.sample(range(lo, hi + 1), min(n, hi - lo + 1))
        for p in positions:
            bits ^= (1 << p)
        result = struct.unpack("f", bits.to_bytes(4, "little"))[0]
        return val if (result != result or abs(result) > 1e15) else result

    # ── 辅助：对张量施加故障（与 register_fault_injection 保持一致）──────
    def _apply_fault(t: torch.Tensor, ft: FaultType):
        tf       = t.float()
        flat     = tf.reshape(-1)
        orig_val = inj_val = 0.0
        pos      = ()

        # 一、瞬态位翻转（flip_count 个元素，每个翻 num_flips 个 bit）
        if ft == FaultType.BIT_FLIP_LOW:
            idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
            orig_val = flat[idxs[0]].item()
            for idx in idxs:
                flat[idx] = _flip_bits(flat[idx].item(), 0, bit_flip_low_bits - 1, num_flips)
            inj_val = flat[idxs[0]].item()
            pos = tuple(idxs[:4])

        elif ft == FaultType.BIT_FLIP_MID:
            idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
            orig_val = flat[idxs[0]].item()
            for idx in idxs:
                flat[idx] = _flip_bits(flat[idx].item(), *bit_flip_mid_bits, num_flips)
            inj_val = flat[idxs[0]].item()
            pos = tuple(idxs[:4])

        elif ft == FaultType.BIT_FLIP_HIGH:
            idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
            orig_val = flat[idxs[0]].item()
            for idx in idxs:
                flat[idx] = _flip_bits(flat[idx].item(), *bit_flip_high_bits, num_flips)
            inj_val = flat[idxs[0]].item()
            pos = tuple(idxs[:4])

        elif ft == FaultType.BIT_FLIP_SIGN:
            idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
            orig_val = flat[idxs[0]].item()
            for idx in idxs:
                flat[idx] = _flip_bits(flat[idx].item(), 31, 31, 1)
            inj_val = flat[idxs[0]].item()
            pos = tuple(idxs[:4])

        # 二、永久故障
        elif ft == FaultType.STUCK_AT_ZERO:
            if tf.dim() >= 2:
                n_ch    = tf.shape[-1]
                n_stuck = max(1, int(n_ch * stuck_ratio))
                idxs    = random.sample(range(n_ch), n_stuck)
                orig_val = tf[..., idxs].abs().mean().item()
                tf[..., idxs] = 0.0
                inj_val = 0.0
                pos     = tuple(idxs[:4])

        elif ft == FaultType.STUCK_AT_ONE:
            if tf.dim() >= 2:
                n_ch    = tf.shape[-1]
                n_stuck = max(1, int(n_ch * stuck_ratio))
                idxs    = random.sample(range(n_ch), n_stuck)
                max_val = tf.abs().max().item()
                orig_val = tf[..., idxs].abs().mean().item()
                tf[..., idxs] = max_val
                inj_val  = max_val
                pos      = tuple(idxs[:4])

        # 三、存储块损坏
        elif ft == FaultType.BLOCK_CORRUPT:
            flat      = tf.reshape(-1)
            n         = flat.numel()
            blen      = max(1, int(n * block_size_ratio))
            start     = random.randint(0, n - blen)
            std       = flat.abs().mean().item() * block_noise_std
            orig_val  = flat[start:start + blen].abs().mean().item()
            flat[start:start + blen] += torch.randn(blen, device=tf.device, dtype=tf.dtype) * std
            inj_val   = flat[start:start + blen].abs().mean().item()
            pos       = (start, start + blen)

        # 四、KV Cache 扰动
        elif ft == FaultType.KV_NOISE:
            std      = tf.abs().mean().item() * kv_noise_std_ratio
            orig_val = tf.abs().mean().item()
            tf.add_(torch.randn_like(tf) * std)
            inj_val  = tf.abs().mean().item()
            pos      = ("全张量",)

        elif ft == FaultType.KV_ZERO_REGION:
            if tf.dim() >= 2:
                n_seq  = tf.shape[-2] if tf.dim() >= 3 else tf.shape[0]
                n_zero = max(1, int(n_seq * kv_zero_ratio))
                start  = random.randint(0, max(0, n_seq - n_zero))
                orig_val = tf[..., start:start + n_zero, :].abs().mean().item()
                tf[..., start:start + n_zero, :] = 0.0
                inj_val  = 0.0
                pos      = (start, start + n_zero)

        # 五、老化/漂移
        elif ft == FaultType.PERSISTENT_BIAS:
            bias     = tf.abs().mean().item() * bias_ratio
            orig_val = tf.abs().mean().item()
            tf.add_(bias)
            inj_val  = tf.abs().mean().item()
            pos      = ("全张量",)

        t_out = tf.to(t.dtype)
        delta = (t_out.float() - t.float()).abs().mean().item()
        return t_out, FaultEvent(
            layer_name="",
            fault_type=ft,
            fault_source=FAULT_SOURCE.get(ft, "未知"),
            position=pos,
            original=orig_val,
            injected=inj_val,
            delta=delta,
        )

    # ── 第一阶段：基准推理（记录每层范数）───────────────────────────────
    print("\n[实验] 第一阶段：基准推理（无故障）...")
    handles_base = []
    all_layer_names = [
        name for name, _ in model.named_modules() if layer_filter in name
    ]

    def _make_baseline_hook(name):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if hasattr(t, "shape"):
                baseline_norms[name] = t.detach().float().norm().item()
        return hook

    for name, module in model.named_modules():
        if layer_filter in name:
            handles_base.append(module.register_forward_hook(_make_baseline_hook(name)))

    baseline_answer = generate_response(model, tokenizer, question, max_new_tokens=max_new_tokens)
    for h in handles_base:
        h.remove()
    print(f"[实验] 基准输出：{baseline_answer}")

    # ── 随机选定注入层和故障类型 ─────────────────────────────────────────
    chosen_layer = random.choice(all_layer_names)
    chosen_fault = random.choice(fault_types)
    fault_layer[0] = chosen_layer
    print(f"\n[实验] 第二阶段：在层 [{chosen_layer}] 注入故障 [{chosen_fault.value}]")

    # ── 第二阶段：故障推理 ───────────────────────────────────────────────
    handles_probe = []

    def _make_inject_hook(name):
        def hook(module, inputs, output):
            nonlocal fault_done
            t = output[0] if isinstance(output, (tuple, list)) else output
            if not hasattr(t, "shape"):
                return output

            if name == fault_layer[0] and not fault_done[0]:
                t_clone = t.clone()
                t_out, ev = _apply_fault(t_clone, chosen_fault)
                ev.layer_name = name
                fault_event[0] = ev
                fault_done[0]  = True
                probed_norms[name] = t_out.detach().float().norm().item()
                return (t_out, *output[1:]) if isinstance(output, (tuple, list)) else t_out

            # 故障注入后的层，只记录范数
            if fault_done[0]:
                probed_norms[name] = t.detach().float().norm().item()
            return output
        return hook

    for name, module in model.named_modules():
        if layer_filter in name:
            handles_probe.append(module.register_forward_hook(_make_inject_hook(name)))

    fault_answer = generate_response(model, tokenizer, question, max_new_tokens=max_new_tokens)
    for h in handles_probe:
        h.remove()
    print(f"[实验] 故障后输出：{fault_answer}")

    # ── 写入日志 ─────────────────────────────────────────────────────────
    ev = fault_event[0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 找到注入层在所有层中的索引，只统计其后的层
    inject_idx = all_layer_names.index(chosen_layer) if chosen_layer in all_layer_names else 0
    layers_after = all_layer_names[inject_idx:]

    with open(log_path, "a", encoding="utf-8") as f:
        sep = "=" * 80
        f.write(f"\n{sep}\n")
        f.write(f"实验时间      : {timestamp}\n")
        f.write(f"测试问题      : {question}\n")
        f.write(f"注入层        : {chosen_layer}\n")
        f.write(f"故障大类      : {ev.fault_source if ev else '未触发'}\n")
        f.write(f"故障类型      : {ev.fault_type.value if ev else '未触发'}\n")
        if ev:
            f.write(f"故障位置      : {ev.position}\n")
            f.write(f"原始值        : {ev.original:+.8f}\n")
            f.write(f"注入值        : {ev.injected:+.8f}\n")
            f.write(f"注入点均值偏差: {ev.delta:.4e}\n")
        f.write(f"\n{'─'*80}\n")
        f.write(f"{'层名':<50} {'基准范数':>12} {'故障后范数':>12} {'偏差率':>10} {'状态':>6}\n")
        f.write(f"{'─'*80}\n")

        found_inject = False
        for lname in all_layer_names:
            base_n  = baseline_norms.get(lname, float("nan"))
            probe_n = probed_norms.get(lname, float("nan"))
            if base_n > 0 and probe_n == probe_n:
                ratio = abs(probe_n - base_n) / (base_n + 1e-9)
            else:
                ratio = float("nan")

            is_inject = (lname == chosen_layer)
            if is_inject:
                found_inject = True

            if found_inject:
                flag = "★注入" if is_inject else ("⚠ 异常" if (ratio == ratio and ratio > 0.01) else "  正常")
                f.write(
                    f"{lname:<50} {base_n:>12.4f} {probe_n:>12.4f} "
                    f"{ratio:>9.2%}  {flag}\n"
                )

        # ── 多维度语义差异分析（写入前计算，含语义向量）────────────────
        diff = _compare_outputs(baseline_answer, fault_answer, tokenizer, model)

        def _fmt(v): return f"{v:.4f} ({v:.2%})" if v is not None else "不可用"

        f.write(f"基准输出  : {baseline_answer}\n")
        f.write(f"故障后输出: {fault_answer}\n")
        f.write(f"\n{'─'*80}\n")
        f.write(f"输出差异分析（依据文献评估三层体系）：\n")
        f.write(f"\n  ▶ 数值层（文献[3][4]）\n")
        f.write(f"    隐状态余弦相似度  : {_fmt(diff['hidden_cosine'])}\n")
        f.write(f"    Logits 余弦相似度 : {_fmt(diff['logits_cosine'])}\n")
        f.write(f"    Top-10 词重叠率   : {_fmt(diff['topk_overlap'])}\n")
        f.write(f"\n  ▶ 语义层（文献[2][3]）\n")
        f.write(f"    ROUGE-L 相似度    : {_fmt(diff['rouge_l'])}\n")
        f.write(f"    词级 Jaccard      : {_fmt(diff['token_jaccard'])}\n")
        f.write(f"\n  ▶ 结构层\n")
        f.write(f"    输出长度变化      : 基准={diff['len_base']} 字符 / 故障后={diff['len_fault']} 字符\n")
        f.write(f"    首句是否一致      : {'是' if diff['first_sentence_same'] else '否'}\n")
        f.write(f"\n  ★ 综合判定          : {diff['verdict']}\n")
        f.write(f"{sep}\n")

    # 控制台摘要
    diff = _compare_outputs(baseline_answer, fault_answer, tokenizer, model)
    def _fmt(v): return f"{v:.4f} ({v:.2%})" if v is not None else "不可用"
    print(f"\n{'─'*65}")
    print(f"[实验] 输出差异分析：")
    print(f"  ▶ 数值层  隐状态余弦={_fmt(diff['hidden_cosine'])}  "
          f"Logits余弦={_fmt(diff['logits_cosine'])}  "
          f"Top-10重叠={_fmt(diff['topk_overlap'])}")
    print(f"  ▶ 语义层  ROUGE-L={_fmt(diff['rouge_l'])}  "
          f"Jaccard={_fmt(diff['token_jaccard'])}")
    print(f"  ▶ 结构层  首句一致={'是' if diff['first_sentence_same'] else '否'}  "
          f"长度 {diff['len_base']}→{diff['len_fault']}")
    print(f"  ★ 判定结论: {diff['verdict']}")
    print(f"[实验] 日志已写入：{log_path}\n")


def run_layer_sensitivity_scan(
    model,
    tokenizer,
    question:           str   = "请用简洁的语言解释什么是量子纠缠。",
    fault_types:        list  = None,
    layer_filter:       str   = "layers",
    log_path:           str   = "layer_sensitivity.log",
    max_new_tokens:     int   = 128,
    repeats_per_layer:  int   = 3,
    # FSS 权重（三维度加权求和）
    w_semantic:         float = 0.50,
    w_logits:           float = 0.20,
    w_norm:             float = 0.30,
    # 故障参数（同 run_single_fault_experiment）
    num_flips:          int   = 4,
    flip_count:         int   = 8,
    bit_flip_low_bits:  int   = 8,
    bit_flip_mid_bits:  tuple = (8, 22),
    bit_flip_high_bits: tuple = (23, 30),
    stuck_ratio:        float = 0.10,
    block_size_ratio:   float = 0.10,
    block_noise_std:    float = 0.50,
    kv_noise_std_ratio: float = 0.20,
    kv_zero_ratio:      float = 0.20,
    bias_ratio:         float = 0.10,
    seed:               int   = None,
    top_k:              int   = 20,
) -> list:
    """逐层故障敏感度扫描（Layer Fault Sensitivity Scan）。

    对模型中所有匹配 layer_filter 的层依次注入故障，计算每层的
    故障敏感度分数（FSS），并按敏感度从高到低排序输出。

    FSS 指标设计（三维度加权）：
    ─────────────────────────────────────────────────────────────────
      FSS(l) = w_semantic × ΔSemantic(l)
             + w_logits   × ΔLogits(l)
             + w_norm     × NormDrift(l)

      ΔSemantic(l) = 1 − hidden_cosine(baseline, fault_output)
                     反映最终输出的语义偏移，是用户可感知的核心指标。

      ΔLogits(l)   = 1 − logits_cosine(baseline, fault_output)
                     反映最后一个 token 的 logits 分布偏移，
                     量化模型内部决策的变化程度。

      NormDrift(l) = mean_{l'≥l} |‖h_{l'}^fault‖ − ‖h_{l'}^base‖|
                                  ─────────────────────────────────
                                          ‖h_{l'}^base‖
                     注入点及其所有下游层的范数相对漂移均值，
                     反映故障的向后传播强度与衰减特性。
    ─────────────────────────────────────────────────────────────────

    Args:
        repeats_per_layer: 每层重复注入次数，多次取均值以降低随机方差。
        top_k:             控制台显示的最敏感层数量；日志记录全部结果。

    Returns:
        list of dict，按 FSS 降序排列，每项包含层名及各子指标。
    """
    import struct
    import datetime

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    fault_types = fault_types or list(FaultType)

    # ── 0. 收集待扫描层 ──────────────────────────────────────────────
    all_layer_names = [
        name for name, _ in model.named_modules() if layer_filter in name
    ]
    if not all_layer_names:
        print(f"[扫描] 未找到包含 '{layer_filter}' 的层，请检查 layer_filter。")
        return []

    total_runs = len(all_layer_names) * repeats_per_layer
    print(f"\n[扫描] 共 {len(all_layer_names)} 个目标层，"
          f"每层重复 {repeats_per_layer} 次，"
          f"预计推理次数：1（基准）+ {total_runs}（故障）= {total_runs + 1} 次")
    print(f"[扫描] FSS 权重：w_semantic={w_semantic}  "
          f"w_logits={w_logits}  w_norm={w_norm}\n")

    # ── 1. 基准推理（记录每层范数和最终输出）────────────────────────
    print("[扫描] 执行基准推理...")
    baseline_norms: dict = {}
    handles_base = []

    def _make_baseline_hook(name):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if hasattr(t, "shape"):
                baseline_norms[name] = t.detach().float().norm().item()
        return hook

    for name, module in model.named_modules():
        if layer_filter in name:
            handles_base.append(module.register_forward_hook(_make_baseline_hook(name)))

    baseline_answer = generate_response(
        model, tokenizer, question, max_new_tokens=max_new_tokens
    )
    for h in handles_base:
        h.remove()
    print(f"[扫描] 基准输出（前 80 字）：{baseline_answer[:80]}...\n")

    # ── 辅助：对张量施加指定故障 ──────────────────────────────────────
    def _flip_bits_local(val: float, lo: int, hi: int, n: int = 1) -> float:
        packed = struct.pack("f", val)
        bits   = int.from_bytes(packed, "little")
        for p in random.sample(range(lo, hi + 1), min(n, hi - lo + 1)):
            bits ^= (1 << p)
        result = struct.unpack("f", bits.to_bytes(4, "little"))[0]
        return val if (result != result or abs(result) > 1e15) else result

    def _apply_fault_local(t: torch.Tensor, ft: FaultType):
        """返回 (注入后张量, orig_val, inj_val)。
        orig_val / inj_val 均为受影响区域的绝对值均值，语义与 FaultEvent 一致。
        """
        tf        = t.float()
        flat      = tf.reshape(-1)
        orig_val  = inj_val = 0.0

        if ft == FaultType.BIT_FLIP_LOW:
            idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
            orig_val = flat[idxs[0]].item()
            for idx in idxs:
                flat[idx] = _flip_bits_local(flat[idx].item(), 0, bit_flip_low_bits - 1, num_flips)
            inj_val = flat[idxs[0]].item()
        elif ft == FaultType.BIT_FLIP_MID:
            idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
            orig_val = flat[idxs[0]].item()
            for idx in idxs:
                flat[idx] = _flip_bits_local(flat[idx].item(), *bit_flip_mid_bits, num_flips)
            inj_val = flat[idxs[0]].item()
        elif ft == FaultType.BIT_FLIP_HIGH:
            idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
            orig_val = flat[idxs[0]].item()
            for idx in idxs:
                flat[idx] = _flip_bits_local(flat[idx].item(), *bit_flip_high_bits, num_flips)
            inj_val = flat[idxs[0]].item()
        elif ft == FaultType.BIT_FLIP_SIGN:
            idxs = random.sample(range(flat.numel()), min(flip_count, flat.numel()))
            orig_val = flat[idxs[0]].item()
            for idx in idxs:
                flat[idx] = _flip_bits_local(flat[idx].item(), 31, 31, 1)
            inj_val = flat[idxs[0]].item()
        elif ft == FaultType.STUCK_AT_ZERO:
            if tf.dim() >= 2:
                idxs     = random.sample(range(tf.shape[-1]), max(1, int(tf.shape[-1] * stuck_ratio)))
                orig_val = tf[..., idxs].abs().mean().item()
                tf[..., idxs] = 0.0
                inj_val  = 0.0
        elif ft == FaultType.STUCK_AT_ONE:
            if tf.dim() >= 2:
                idxs     = random.sample(range(tf.shape[-1]), max(1, int(tf.shape[-1] * stuck_ratio)))
                orig_val = tf[..., idxs].abs().mean().item()
                tf[..., idxs] = tf.abs().max().item()
                inj_val  = tf[..., idxs].abs().mean().item()
        elif ft == FaultType.BLOCK_CORRUPT:
            n        = flat.numel()
            blen     = max(1, int(n * block_size_ratio))
            start    = random.randint(0, n - blen)
            std      = flat.abs().mean().item() * block_noise_std
            orig_val = flat[start:start + blen].abs().mean().item()
            flat[start:start + blen] += (
                torch.randn(blen, device=tf.device, dtype=tf.dtype) * std
            )
            inj_val  = flat[start:start + blen].abs().mean().item()
        elif ft == FaultType.KV_NOISE:
            orig_val = tf.abs().mean().item()
            tf.add_(torch.randn_like(tf) * (orig_val * kv_noise_std_ratio))
            inj_val  = tf.abs().mean().item()
        elif ft == FaultType.KV_ZERO_REGION:
            if tf.dim() >= 2:
                n_seq    = tf.shape[-2] if tf.dim() >= 3 else tf.shape[0]
                n_zero   = max(1, int(n_seq * kv_zero_ratio))
                start    = random.randint(0, max(0, n_seq - n_zero))
                orig_val = tf[..., start:start + n_zero, :].abs().mean().item()
                tf[..., start:start + n_zero, :] = 0.0
                inj_val  = 0.0
        elif ft == FaultType.PERSISTENT_BIAS:
            orig_val = tf.abs().mean().item()
            tf.add_(orig_val * bias_ratio)
            inj_val  = tf.abs().mean().item()

        return tf.to(t.dtype), orig_val, inj_val

    # ── 2. 逐层扫描 ──────────────────────────────────────────────────
    results = []

    for layer_idx, target_layer in enumerate(all_layer_names):
        acc = {"delta_semantic": 0.0, "delta_logits": 0.0,
               "norm_drift": 0.0, "rouge_l_dist": 0.0,
               "orig_val": 0.0, "inj_val": 0.0, "inject_delta": 0.0}
        fault_types_used: list = []   # 记录每次 repeat 实际注入的故障类型
        valid_runs = 0

        for _ in range(repeats_per_layer):
            chosen_ft     = random.choice(fault_types)
            fault_types_used.append(chosen_ft.value)
            probed_norms  = {}
            fault_done    = [False]
            inject_detail = [0.0, 0.0, 0.0]   # [orig_val, inj_val, inject_delta]

            def _make_scan_hook(name, tgt, done_flag, pnorms, idetail, _ft=chosen_ft):
                def hook(module, inputs, output):
                    t = output[0] if isinstance(output, (tuple, list)) else output
                    if not hasattr(t, "shape"):
                        return output
                    if name == tgt and not done_flag[0]:
                        t_clone = t.clone()
                        t_out, orig_val, inj_val = _apply_fault_local(t_clone, _ft)
                        idetail[0]     = orig_val
                        idetail[1]     = inj_val
                        idetail[2]     = (t_out.float() - t.float()).abs().mean().item()
                        done_flag[0]   = True
                        pnorms[name]   = t_out.detach().float().norm().item()
                        return (t_out, *output[1:]) if isinstance(output, (tuple, list)) else t_out
                    if done_flag[0]:
                        pnorms[name] = t.detach().float().norm().item()
                    return output
                return hook

            handles_scan = []
            for name, module in model.named_modules():
                if layer_filter in name:
                    handles_scan.append(
                        module.register_forward_hook(
                            _make_scan_hook(name, target_layer, fault_done,
                                            probed_norms, inject_detail)
                        )
                    )
            try:
                fault_answer = generate_response(
                    model, tokenizer, question, max_new_tokens=max_new_tokens
                )
            except Exception:
                fault_answer = ""
            finally:
                for h in handles_scan:
                    h.remove()

            # ── 语义差异（三层评估）──────────────────────────────────
            diff = _compare_outputs(baseline_answer, fault_answer, tokenizer, model)
            hc = diff["hidden_cosine"]
            lc = diff["logits_cosine"]
            rl = diff["rouge_l"]

            # 优先使用隐状态余弦，无 model 时降级到 ROUGE-L
            # cosine ∈ [-1,1]，故 (1-cosine)/2 ∈ [0,1]，保证量纲统一
            d_sem    = ((1.0 - hc) / 2.0) if hc is not None else (1.0 - rl)
            d_logits = ((1.0 - lc) / 2.0) if lc is not None else 0.0

            # ── 下游层范数漂移 ────────────────────────────────────────
            inject_pos = all_layer_names.index(target_layer)
            drifts = []
            for ln in all_layer_names[inject_pos:]:
                bn = baseline_norms.get(ln, 0.0)
                pn = probed_norms.get(ln, float("nan"))
                if bn > 0 and pn == pn:
                    drifts.append(abs(pn - bn) / (bn + 1e-9))
            norm_drift = sum(drifts) / len(drifts) if drifts else 0.0

            acc["delta_semantic"] += d_sem
            acc["delta_logits"]   += d_logits
            acc["norm_drift"]     += norm_drift
            acc["rouge_l_dist"]   += (1.0 - rl)
            acc["orig_val"]       += inject_detail[0]
            acc["inj_val"]        += inject_detail[1]
            acc["inject_delta"]   += inject_detail[2]
            valid_runs += 1

        if valid_runs == 0:
            continue
        for k in acc:
            acc[k] /= valid_runs

        fss = (
            w_semantic * acc["delta_semantic"]
            + w_logits * acc["delta_logits"]
            + w_norm   * acc["norm_drift"]
        )
        results.append({
            "layer_name":       target_layer,
            "layer_idx":        layer_idx,
            "fss":              fss,
            "delta_semantic":   acc["delta_semantic"],
            "delta_logits":     acc["delta_logits"],
            "norm_drift":       acc["norm_drift"],
            "rouge_l_dist":     acc["rouge_l_dist"],
            "fault_types_used": fault_types_used,
            "avg_orig_val":     acc["orig_val"],
            "avg_inj_val":      acc["inj_val"],
            "avg_inject_delta": acc["inject_delta"],
        })

        pct = (layer_idx + 1) / len(all_layer_names) * 100
        print(
            f"[扫描] [{pct:5.1f}%] 层 {layer_idx+1:3d}/{len(all_layer_names)} "
            f"{target_layer:50s}  FSS={fss:.4f}  "
            f"Δsem={acc['delta_semantic']:.4f}  drift={acc['norm_drift']:.4f}"
        )

    # ── 3. 按 FSS 排序 ────────────────────────────────────────────────
    results.sort(key=lambda x: x["fss"], reverse=True)

    # ── 4. 控制台输出：Top-K 敏感层 + Bottom-K 低影响层 ──────────────
    w_line = 90
    show = min(top_k, len(results))
    header = (
        f"{'排名':>4}  {'FSS':>7}  {'Δsemantic':>10}  "
        f"{'Δlogits':>8}  {'NormDrift':>10}  层名"
    )
    row_fmt = (
        "{marker}{rank:3d}  {fss:7.4f}  {ds:10.4f}  "
        "{dl:8.4f}  {nd:10.4f}  {name}"
    )
    sep = "─" * w_line

    print(f"\n{'='*w_line}")
    print(f"  层故障敏感度排名（影响最大的 Top-{show} 层）")
    print(f"  FSS = {w_semantic}×Δsemantic + {w_logits}×Δlogits + {w_norm}×NormDrift")
    print(f"{'='*w_line}")
    print(header)
    print(sep)
    for rank, r in enumerate(results[:show], 1):
        marker = "▶" if rank <= 3 else " "
        print(row_fmt.format(
            marker=marker, rank=rank, fss=r["fss"],
            ds=r["delta_semantic"], dl=r["delta_logits"],
            nd=r["norm_drift"], name=r["layer_name"]
        ))
    print(sep)

    if len(results) > show:
        tail_n = min(show, len(results) - show)
        print(f"\n  影响最小的 Bottom-{tail_n} 层：")
        print(sep)
        for rank, r in enumerate(results[-tail_n:], len(results) - tail_n + 1):
            print(row_fmt.format(
                marker=" ", rank=rank, fss=r["fss"],
                ds=r["delta_semantic"], dl=r["delta_logits"],
                nd=r["norm_drift"], name=r["layer_name"]
            ))
        print(sep)

    # ── 5. 写入完整日志 ───────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_w = 120
    log_sep = "─" * log_w
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*log_w}\n")
        f.write(f"层故障敏感度扫描  {timestamp}\n")
        f.write(f"测试问题     : {question}\n")
        f.write(f"FSS 权重     : w_semantic={w_semantic}  "
                f"w_logits={w_logits}  w_norm={w_norm}\n")
        f.write(f"每层重复次数 : {repeats_per_layer}\n")
        f.write(f"{log_sep}\n")
        f.write(
            f"{'排名':>4}  {'FSS':>8}  {'Δsemantic':>10}  "
            f"{'Δlogits':>9}  {'NormDrift':>11}  "
            f"{'注入故障类型':<36}  层名\n"
        )
        f.write(f"{'':>4}  {'':>8}  {'':>10}  {'':>9}  {'':>11}  "
                f"{'原始值均值':>12}  {'注入值均值':>12}  {'注入点MAE':>12}\n")
        f.write(f"{log_sep}\n")
        for rank, r in enumerate(results, 1):
            seen: list = []
            for ft_val in r["fault_types_used"]:
                if ft_val not in seen:
                    seen.append(ft_val)
            faults_str = " / ".join(seen)
            f.write(
                f"{rank:4d}  {r['fss']:8.4f}  {r['delta_semantic']:10.4f}  "
                f"{r['delta_logits']:9.4f}  {r['norm_drift']:11.4f}  "
                f"{faults_str:<36}  {r['layer_name']}\n"
            )
            f.write(
                f"{'':>4}  {'':>8}  {'':>10}  {'':>9}  {'':>11}  "
                f"{r['avg_orig_val']:>+12.6f}  "
                f"{r['avg_inj_val']:>+12.6f}  "
                f"{r['avg_inject_delta']:>12.4e}\n"
            )
        f.write(f"{'='*log_w}\n")

    print(f"\n[扫描] 全部 {len(results)} 层结果已写入：{log_path}")
    return results


if __name__ == "__main__":
    _check_transformers_version()
    model, tokenizer = load_model_and_tokenizer()

    # 打印模型层级结构（调整 max_depth 控制展示深度）
    print_model_structure(model, max_depth=3)

    # ------------------------------------------------------------------ #
    #  单点故障传播实验（五大类静默故障，依据文献分类体系）
    #
    #  fault_types : None = 随机选一种；也可指定子集，如：
    #    [FaultType.BIT_FLIP_LOW]            # 只测低位翻转
    #    [FaultType.STUCK_AT_ZERO]           # 只测永久粘滞
    #    [FaultType.BLOCK_CORRUPT]           # 只测存储块损坏
    #    [FaultType.KV_NOISE, FaultType.KV_ZERO_REGION]  # 只测 KV Cache
    #    [FaultType.PERSISTENT_BIAS]         # 只测老化漂移
    #
    #  seed        : 固定后可复现同一次故障（便于对比实验）
    #  max_new_tokens: 控制输出长度
    # ------------------------------------------------------------------ #
    run_single_fault_experiment(
        model, tokenizer,
        question            = "请用简洁的语言解释什么是量子纠缠。",
        fault_types         = None,
        layer_filter        = "layers",
        log_path            = "fault_experiment.log",
        max_new_tokens      = 512,

        # ── 位翻转（放大：多元素 + 多 bit）──────────────────────────────
        # num_flips=4  : 每个元素同时翻 4 个 bit，指数位被波及概率大幅提升
        # flip_count=8 : 同一层同时损坏 8 个元素（BER ≈ 8/tensor_size）
        num_flips           = 4,
        flip_count          = 8,
        bit_flip_low_bits   = 8,        # 低位范围扩展到 bit0-7
        bit_flip_mid_bits   = (8, 22),  # 中位扩展覆盖全部尾数
        bit_flip_high_bits  = (23, 30), # 高位（指数域）保持不变

        # ── 永久故障（放大：10% 通道粘滞）──────────────────────────────
        stuck_ratio         = 0.10,     # 原 1% → 10%

        # ── 存储块损坏（放大：块更大、噪声更强）─────────────────────────
        block_size_ratio    = 0.10,     # 原 2% → 10%
        block_noise_std     = 0.50,     # 原 0.10 → 0.50（噪声为均值的 50%）

        # ── KV Cache 扰动（放大：噪声 ×10、归零区域 ×4）────────────────
        kv_noise_std_ratio  = 0.20,     # 原 2% → 20%
        kv_zero_ratio       = 0.20,     # 原 5% → 20%

        # ── 老化/漂移（放大：偏置 ×10）──────────────────────────────────
        bias_ratio          = 0.10,     # 原 1% → 10%

        seed                = None,
    )

    # ------------------------------------------------------------------ #
    #  层故障敏感度扫描
    #
    #  对所有层依次注入故障，计算每层的 FSS 并排名。
    #  FSS(l) = 0.5×ΔSemantic + 0.2×ΔLogits + 0.3×NormDrift
    #
    #  参数调整建议：
    #    repeats_per_layer=1  : 快速粗扫（节省时间）
    #    repeats_per_layer=3+ : 均值更稳定，推荐正式实验使用
    #    max_new_tokens=64    : 越短越快，敏感度对比仍有效
    #    fault_types=[FaultType.BIT_FLIP_HIGH, FaultType.STUCK_AT_ZERO]
    #                         : 只扫特定故障类型
    # ------------------------------------------------------------------ #
    run_layer_sensitivity_scan(
        model, tokenizer,
        question            = "请用简洁的语言解释什么是量子纠缠。",
        fault_types         = None,           # None = 全部故障类型
        layer_filter        = "layers",
        log_path            = "layer_sensitivity.log",
        max_new_tokens      = 64,
        repeats_per_layer   = 1,              # 快速扫：每层只注入 1 次
        w_semantic          = 0.50,
        w_logits            = 0.20,
        w_norm              = 0.30,
        top_k               = 20,
        seed                = 42,
    )

    # 进入交互式对话
    interactive_chat(model, tokenizer)
