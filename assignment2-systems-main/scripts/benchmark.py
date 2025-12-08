import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
cs336_dir = os.path.join(parent_dir, 'cs336-basics')
sys.path.insert(0, cs336_dir)

import cs336_basics
# 确保包内的子模块被加载，这样可以通过 `cs336_basics.model` 和 `cs336_basics.nn_utils` 访问
import cs336_basics.model
import cs336_basics.nn_utils
import argparse
import time
import timeit
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from torch import Tensor
from jaxtyping import Float, Bool
from einops import rearrange, einsum
import math
import torch.cuda.nvtx as nvtx

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = cs336_basics.nn_utils.softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        result = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    
    return result

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def benchmark_model(
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        warmup_steps: int,
        measurement_steps: int,
        device: torch.device,
        do_backward: bool = True,
        dtype: torch.dtype = torch.float32
) -> Tuple[float, float, float]:
    """
    基准测试模型性能
    
    Args:
        model: 要测试的模型
        batch_size: 批量大小
        seq_len: 序列长度
        vocab_size: 词表大小
        warmup_steps: 预热步数
        measurement_steps: 测量步数
        device: 设备
        do_backward: 是否进行反向传播
        dtype: 数据类型
    
    Returns:
        Tuple[平均前向时间(ms), 平均反向时间(ms), 峰值内存(MB)]
    """
    model.to(device)
    model.train()

    if dtype == torch.float16:
        model.half()

    # 预热阶段
    print(f"开始预热 ({warmup_steps}步)...")
    for i in range(warmup_steps):
        # 随机输入数据
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        if dtype == torch.float16:
            input_ids = input_ids.half()
        
        # 前向传播
        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        logits = model(input_ids)

        torch.cuda.synchronize()
        end_time = timeit.default_timer()

        # 反向传播
        if do_backward:
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            loss = cs336_basics.nn_utils.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            torch.cuda.synchronize()
            start_time_backward = timeit.default_timer()

            loss.backward()

            torch.cuda.synchronize()
            end_time_backward = timeit.default_timer()

        model.zero_grad()

        print(f"预热第{i}步完成")

    # 测量阶段
    print(f"开始测量 ({measurement_steps}步)...")
    forward_times = []
    backward_times = []

    # 清空CUDA缓存以获得更准确的内存测量
    torch.cuda.empty_cache()

    for i in range(measurement_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        if dtype == torch.float16:
            input_ids = input_ids.half()

        # 重置CUDA内存分配器统计信息
        torch.cuda.reset_peak_memory_stats(device)

        # 前向传播
        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        logits = model(input_ids)

        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        forward_time = (end_time - start_time) * 1000
        forward_times.append(forward_time)

        # 反向传播
        if do_backward:
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            loss = cs336_basics.nn_utils.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            torch.cuda.synchronize()
            start_time_backward = timeit.default_timer()

            loss.backward()

            torch.cuda.synchronize()
            end_time_backward = timeit.default_timer()
            backward_time = (end_time_backward - start_time_backward) * 1000
            backward_times.append(backward_time)

        model.zero_grad()

        # 获取峰值内存使用量
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 转换为MB

        if (i+1) % max(1, measurement_steps // 10) == 0:
            print(f"  步骤 {i+1}/{measurement_steps}: 前向={forward_time:.2f}ms",
                  f"反向={backward_time:.2f}ms" if do_backward else "")
            
    avg_forward_time = np.mean(forward_times)
    avg_backward_time = np.mean(backward_times) if do_backward else 0.0

    return avg_forward_time, avg_backward_time, peak_memory

def main():
    parser = argparse.ArgumentParser(description="Transformer模型性能基准测试")

    # 模型超参数
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--num_heads", type=int, default=16, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--ff_dim", type=int, default=1344, help="前馈网络维度")
    parser.add_argument("--max_seq_len", type=int, default=256, help="最大序列长度")
    parser.add_argument("--rope_theta", type=int, default=10000, help="RoPE theta值")
    
    # 基准测试参数
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--seq_len", type=int, default=256, help="序列长度")
    parser.add_argument("--warmup_steps", type=int, default=5, help="预热步数")
    parser.add_argument("--measurement_steps", type=int, default=20, help="测量步数")
    parser.add_argument("--no_backward", action="store_true", help="仅测试前向传播")
    parser.add_argument("--dtype", type=str, default="float32", 
                       choices=["float32", "float16"], help="数据类型")
    parser.add_argument("--device", type=str, default="cuda", 
                       choices=["cuda", "cpu"], help="运行设备")
    
    args = parser.parse_args()

    # 设置设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # 设置数据类型
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16
    }
    dtype = dtype_map[args.dtype]

    print("=" * 80)
    print("Transformer模型性能基准测试")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"数据类型: {args.dtype}")
    print(f"模型配置: d_model={args.d_model}, num_heads={args.num_heads}, num_layers={args.num_layers}")
    print(f"输入配置: batch_size={args.batch_size}, seq_len={args.seq_len}")
    print(f"测试配置: warmup_steps={args.warmup_steps}, measurement_steps={args.measurement_steps}")
    print(f"测试模式: {'前向传播' if args.no_backward else '前向+反向传播'}")
    print("-" * 80)

    # 初始化模型
    print("初始化模型...")
    model = cs336_basics.model.BasicsTransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.seq_len,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.ff_dim,
        rope_theta = args.rope_theta
    )

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} 总参数, {trainable_params:,} 可训练参数")
    
    # 运行基准测试
    print("\n开始基准测试...")
    avg_forward_time, avg_backward_time, peak_memory = benchmark_model(
        model=model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        device=device,
        do_backward=not args.no_backward,
        dtype=dtype
    )

    # 打印结果
    print("\n" + "=" * 80)
    print("基准测试结果")
    print("=" * 80)
    print(f"平均前向时间: {avg_forward_time:.2f} ms")
    
    if not args.no_backward:
        print(f"平均反向时间: {avg_backward_time:.2f} ms")
        print(f"平均总时间: {(avg_forward_time + avg_backward_time):.2f} ms")
    
    print(f"峰值GPU内存: {peak_memory:.2f} MB")

    # 计算吞吐量
    tokens_per_batch = args.batch_size * args.seq_len
    if not args.no_backward:
        total_time_ms = avg_forward_time + avg_backward_time
        tokens_per_second = (tokens_per_batch / total_time_ms) * 1000
        print(f"\n吞吐量: {tokens_per_second:.0f} tokens/秒")
    else:
        tokens_per_second = (tokens_per_batch / avg_forward_time) * 1000
        print(f"\n吞吐量: {tokens_per_second:.0f} tokens/秒 (仅前向)")
    
    print("=" * 80)

if __name__ == "__main__":
    # 设置可重现的随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    main()
        