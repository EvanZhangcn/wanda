#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_smooth_demo.py
典型层名示例 (LLaMA / 类似结构模型):
    model.layers.0.mlp.down_proj
    model.layers.0.mlp.gate_proj
    model.layers.0.mlp.up_proj
    model.layers.0.self_attn.q_proj
    model.layers.0.self_attn.k_proj
    model.layers.0.self_attn.v_proj
    model.layers.0.self_attn.o_proj

用法示例:
    python /data/zhangliwen/mc/wanda/run_smooth_demo.py \
        --model_path /data/zhangliwen/mc/Llama-2-7b-hf \
        --layer_name model.layers.10.mlp.down_proj \
        --alpha 0.5 \
        --output_dir smooth_analysis_results
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from kneed import KneeLocator

# 允许脚本在仓库根目录执行时找到 lib.prune
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from lib.prune import calculate_smoothing_scale_factor
from lib.layerwrapper import WrappedGPT
from lib.data import get_loaders

# ----------------------------- 工具函数 -----------------------------

def compute_singular_values(matrix: torch.Tensor):
    """计算给定矩阵的奇异值"""
    with torch.no_grad():
        # 使用 svdvals 可以更高效地只计算奇异值
        singular_values = torch.linalg.svdvals(matrix.float()).cpu().numpy()
    return singular_values

def calculate_effective_rank(cumulative_energy_ratio: np.ndarray, thresholds: list) -> dict:
    """根据累计能量覆盖率计算有效秩"""
    ranks = {}
    for t in thresholds:
        # 寻找第一个超过阈值的索引，其位置+1即为有效秩
        eff_rank = np.searchsorted(cumulative_energy_ratio, t, side='left') + 1
        ranks[t] = eff_rank
    return ranks

def plot_svd_analysis(axes_row, matrix, matrix_name, title_prefix):
    """
    对给定的矩阵进行SVD分析和绘图，画在一行子图上。
    :param axes_row: 一行两个子图 (e.g., axes[0])
    :param matrix: 要分析的torch矩阵 (delta_B or delta_B_smoothed)
    :param matrix_name: 矩阵的名字，用于标题 (e.g., "delta_B")
    :param title_prefix: 图表的主标题前缀 (e.g., "Layer 0 Sublayer mlp.proj")
    """
    with torch.no_grad():
        singular_values = torch.linalg.svdvals(matrix.float()).cpu().numpy()

    # 计算能量
    total_energy = np.sum(singular_values ** 2)
    cumulative_energy = np.cumsum(singular_values ** 2)
    cumulative_energy_ratio = cumulative_energy / total_energy

    # --- 左侧子图：完整谱图 ---
    ax_left = axes_row[0]
    ax_left.plot(singular_values)
    ax_left.set_yscale('log')
    ax_left.set_title(f'Complete Spectrum of {matrix_name}\n{title_prefix}')
    ax_left.set_xlabel('Singular Value Index')
    ax_left.set_ylabel('Singular Value (log scale)')
    ax_left.grid(True)

    # --- 右侧子图：局部放大和标注 ---
    ax_right = axes_row[1]
    num_values_to_show = 64
    plot_indices = np.arange(min(len(singular_values), num_values_to_show))
    safe_singular_values = singular_values[:len(plot_indices)] + 1e-10

    ax_right.plot(plot_indices, safe_singular_values, 'b-')
    ax_right.set_yscale('log')
    ax_right.set_title(f'First {num_values_to_show} Values of {matrix_name}\n{title_prefix}')
    ax_right.set_xlabel('Singular Value Index')
    ax_right.set_ylabel('Singular Value (log scale)')
    ax_right.grid(True)

    # 标注肘部点
    try:
        if len(plot_indices) > 1:
            kneedle = KneeLocator(plot_indices, np.log10(safe_singular_values), S=1.0,
                                  curve="convex", direction="decreasing")
            if kneedle.elbow is not None:
                x, y = kneedle.elbow, singular_values[kneedle.elbow]
                energy = cumulative_energy_ratio[x] * 100
                ax_right.annotate(f'Elbow (Rank {x})\nEnergy: {energy:.2f}%', xy=(x, y),
                                  xytext=(15, 40), textcoords='offset points',
                                  arrowprops=dict(facecolor='green', shrink=0.05),
                                  bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    except Exception as e:
        print(f"      - Could not find or annotate elbow point for {matrix_name}: {e}")

    # 标注秩32
    if len(plot_indices) >= 32:
        x, y = 31, singular_values[31]
        energy = cumulative_energy_ratio[x] * 100
        ax_right.annotate(f'Top 32 Ranks\nEnergy: {energy:.2f}%', xy=(x, y), xytext=(-80, -40),
                          textcoords='offset points', arrowprops=dict(facecolor='red', shrink=0.05))

    # 标注秩64
    if len(plot_indices) >= 64:
        x, y = 63, singular_values[63]
        energy = cumulative_energy_ratio[x] * 100
        ax_right.annotate(f'Top 64 Ranks\nEnergy: {energy:.2f}%', xy=(x, y), xytext=(-80, 20),
                          textcoords='offset points',
                          arrowprops=dict(facecolor='purple', shrink=0.05))


def get_llm(model_path, cache_dir="llm_weights"):
    """加载LLM模型并设置序列长度"""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    # 设置模型一次性处理的最大token数（即序列长度）
    model.seqlen = 2048
    if hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings < 2048:
        model.seqlen = model.config.max_position_embeddings
        
    return model


# ----------------------------- 主流程 -----------------------------
def analyze_layer(model, dataloader, layer_name: str, alpha: float, device: str):
    """使用真实校准数据分析单个线性层"""
    modules_dict = dict(model.named_modules())
    if layer_name not in modules_dict:
        sys.exit(f"错误: 指定的层 '{layer_name}' 未找到。")
    
    original_linear_layer = modules_dict[layer_name]
    if not isinstance(original_linear_layer, nn.Linear):
        sys.exit(f"错误: 目标层 '{layer_name}' 不是 nn.Linear, 实际类型: {type(original_linear_layer)}")

    original_weight = original_linear_layer.weight.data.clone().to(device)
    out_dim, in_dim = original_weight.shape
    print(f"层: {layer_name} 维度: (out={out_dim}, in={in_dim})")

    # ============== 使用真实校准数据收集激活尺度 (与prune.py对齐) ==============
    print("使用校准数据收集激活尺度...")
    
    # 使用 WrappedGPT 包装目标层
    wrapped_layer = WrappedGPT(original_linear_layer)
    
    # 定义一个钩子来调用包装器的 add_batch 方法
    def hook(module, inp, out):
        # wrapped_layer.add_batch 会处理输入(inp[0])和输出(out)
        wrapped_layer.add_batch(inp[0].data, out.data)

    #将钩子添加到目标层
    handle = original_linear_layer.register_forward_hook(hook)

    #在校准数据上运行模型以触发钩子
    for batch in dataloader:
        try:
            with torch.no_grad():  # 节省内存
                model(batch[0].to(device))
        except Exception as e:
            print(f"Warning: Batch processing failed: {e}")
            continue

    handle.remove() # 移除钩子
    
    # 从包装器中获取激活尺度
    activation_scales = wrapped_layer.act_scales.float().to(device)
    print("激活尺度收集完毕。")

    # ============== 计算 smoothing 因子并生成平滑矩阵 ==============
    s = calculate_smoothing_scale_factor(activation_scales, original_weight, alpha=alpha)
    weight_smoothed = original_weight * s.to(original_weight.device, dtype=original_weight.dtype)
    
    #print("以下为调试信息：")
    #print("计算 smoothing 因子 s ...")
    #print(f"原始权重形状: {original_weight.shape}")
    #print(f"s因子形状: {s.shape}")
    #print(f"s因子统计: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}")

    # ============== 重建误差计算 ==============
    print("计算重建误差...")

    # 使用随机生成的输入数据
    num_samples = 1000
    random_inputs = torch.randn(num_samples, in_dim, device=device, dtype=original_weight.dtype)

    with torch.no_grad():
        # 计算原始和smooth后的输出
        output_original = torch.matmul(random_inputs, original_weight.T)
        output_smoothed = torch.matmul(random_inputs, weight_smoothed.T)
        
        # 计算误差
        diff = output_smoothed - output_original
        l2_error = torch.mean(diff ** 2).item()
        l1_error = torch.mean(torch.abs(diff)).item()
        
        # 计算相对误差
        original_norm = torch.norm(output_original, dim=-1).mean().item()
        relative_error = torch.norm(diff, dim=-1).mean().item() / (original_norm + 1e-8)
        
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(
            output_original.flatten(), 
            output_smoothed.flatten(), 
            dim=0
        ).item()
        
        # 计算Frobenius范数误差
        frobenius_error = torch.norm(diff, 'fro').item()
        
        print("\n===== 重建误差分析 (随机输入) =====")
        print(f"随机输入样本数: {num_samples}")
        print(f"L2误差 (MSE): {l2_error:.6e}")
        print(f"L1误差 (MAE): {l1_error:.6e}")
        print(f"Frobenius范数误差: {frobenius_error:.6e}")
        print(f"相对误差: {relative_error:.6e}")
        print(f"余弦相似度: {cos_sim:.6f}")
        print(f"原始输出范数: {original_norm:.6e}")
        print(f"平滑输出范数: {torch.norm(output_smoothed, dim=-1).mean().item():.6e}")

    # ============== SVD 奇异值谱 ==============
    print("计算未平滑奇异值谱 ...")
    sv_unsmooth = compute_singular_values(original_weight)
    print("计算平滑后奇异值谱 ...")
    sv_smooth = compute_singular_values(weight_smoothed)

    # ============== 能量覆盖与有效秩 ==============
    energy_unsmooth = sv_unsmooth ** 2
    energy_smooth = sv_smooth ** 2

    #加上极小的数值以防止后续作分母时除零错误
    total_unsmooth = energy_unsmooth.sum() + 1e-12
    total_smooth = energy_smooth.sum() + 1e-12

    cer_unsmooth = np.cumsum(energy_unsmooth) / total_unsmooth
    cer_smooth = np.cumsum(energy_smooth) / total_smooth

    # 计算有效秩
    energy_thresholds = [0.9, 0.95, 0.99, 0.999]
    eff_rank_unsmooth = calculate_effective_rank(cer_unsmooth, energy_thresholds)
    eff_rank_smooth = calculate_effective_rank(cer_smooth, energy_thresholds)

    print("\n===== 能量覆盖率与有效秩对比 =====")
    print(f"{'Metric':<18} | {'Unsmooth':<12} | {'Smooth':<12} | {'Change':<12}")
    print("-" * 60)
    for k in [8, 16, 32, 64, 128, 256]:
        if k <= len(cer_unsmooth):
            eu = cer_unsmooth[k-1] * 100
            es = cer_smooth[k-1] * 100
            print(f"Energy @ top {k:<4} | {eu:8.2f}%     | {es:8.2f}%     | {es-eu:+8.2f}pp")

    print("-" * 60)
    for t in energy_thresholds:
        ru = eff_rank_unsmooth[t]
        rs = eff_rank_smooth[t]
        print(f"Eff. Rank @ {t*100}% | {ru:<12} | {rs:<12} | {rs-ru:<+12d}")

    return {
        'sv_unsmooth': sv_unsmooth,
        'sv_smooth': sv_smooth,
        'cer_unsmooth': cer_unsmooth,
        'cer_smooth': cer_smooth,
        's': s.cpu().numpy(),
        'activation_scales': activation_scales.cpu().numpy(),
        'eff_rank_unsmooth': eff_rank_unsmooth,
        'eff_rank_smooth': eff_rank_smooth,
        'energy_thresholds': energy_thresholds,
    }


def plot_and_save(results, layer_name: str, output_dir: str, alpha: float):
    os.makedirs(output_dir, exist_ok=True)
    sv_unsmooth = results['sv_unsmooth']
    sv_smooth = results['sv_smooth']
    cer_unsmooth = results['cer_unsmooth']
    cer_smooth = results['cer_smooth']

    max_sv_local = min(64, len(sv_unsmooth)) # 固定显示前64个

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Smoothing SVD Analysis for {layer_name}', fontsize=16)

    # (0,0) 全谱
    axes[0,0].plot(sv_unsmooth, label='unsmooth')
    axes[0,0].plot(sv_smooth, label='smooth')
    axes[0,0].set_yscale('log')
    axes[0,0].set_title('Full Spectrum (log)')
    axes[0,0].set_xlabel('Index'); axes[0,0].set_ylabel('Singular Value')
    axes[0,0].grid(True); axes[0,0].legend()

    # (0,1) 前 max_sv 局部
    axes[0,1].plot(np.arange(max_sv_local), sv_unsmooth[:max_sv_local], label='unsmooth')
    axes[0,1].plot(np.arange(max_sv_local), sv_smooth[:max_sv_local], label='smooth')
    axes[0,1].set_yscale('log')
    axes[0,1].set_title(f'First {max_sv_local} Singular Values (log)')
    axes[0,1].set_xlabel('Index'); axes[0,1].set_ylabel('Singular Value')
    axes[0,1].grid(True); axes[0,1].legend()

    # (1,0) 能量覆盖曲线
    axes[1,0].plot(cer_unsmooth, label='unsmooth')
    axes[1,0].plot(cer_smooth, label='smooth')
    axes[1,0].set_title('Cumulative Energy Ratio (Coverage)')
    axes[1,0].set_xlabel('k (rank)'); axes[1,0].set_ylabel('Energy Ratio')
    axes[1,0].grid(True); axes[1,0].legend()

    # (1,1) s / activation_scales 统计分布
    s = results['s']
    act = results['activation_scales']
    axes[1,1].hist(s.flatten(), bins=60, alpha=0.6, label='s (scale factor)')
    axes[1,1].hist(act.flatten(), bins=60, alpha=0.6, label='activation_scales (surrogate)')
    axes[1,1].set_title('Distribution of s and activation_scales (surrogate)')
    axes[1,1].set_xlabel('Value'); axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend(); axes[1,1].grid(True)

    # 更新文件名以包含 alpha 值
    filename_prefix = f'sv_spectrum_{layer_name.replace(".", "_")}_alpha_{alpha}'
    out_path = os.path.join(output_dir, f'{filename_prefix}.png')
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"图像已保存: {out_path}")

    # 保存数值统计
    stats_path = os.path.join(output_dir, f'stats_{layer_name.replace(".", "_")}_alpha_{alpha}.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Alpha: {alpha}\n\n")
        for k in [8,16,32,64,128,256]:
            if k <= len(results['cer_unsmooth']):
                eu = results['cer_unsmooth'][k-1]
                es = results['cer_smooth'][k-1]
                f.write(f"Energy@{k}: unsmooth={eu*100:.2f}%, smooth={es*100:.2f}% (Δ={(es-eu)*100:.2f}pp)\n")
        f.write(f"s stats: min={s.min():.4e}, max={s.max():.4e}, mean={s.mean():.4e}\n")
    print(f"统计已保存: {stats_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', type=str, required=True, help='HF 模型权重路径')
    ap.add_argument('--layer_name', type=str, required=True, help='目标线性层名称 (例如 model.layers.10.mlp.down_proj)')
    ap.add_argument('--alpha', type=float, default=0.5, help='smoothing 强度 (0~1)')
    ap.add_argument('--output_dir', type=str, default='smooth_analysis_demo', help='输出目录')
    ap.add_argument('--device', type=str, default='cuda:0', help='主设备 (SVD / 权重)')
    ap.add_argument('--no_cuda', action='store_true', help='强制使用 CPU')
    ap.add_argument('--nsamples', type=int, default=128, help='用于校准的样本数量')
    ap.add_argument('--seed', type=int, default=42, help='校准数据集的随机种子')
    return ap.parse_args()


def main():
    args = parse_args()
    if args.no_cuda or not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    # 加载模型
    print(f"正在从 '{args.model_path}' 加载模型...")
    try:
        from transformers import AutoTokenizer
        model = get_llm(args.model_path)
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        sys.exit(f"加载模型失败: {e}")

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # 加载校准数据
    print("加载 c4 校准数据...")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer
    )
    print("数据集加载完成。")

    results = analyze_layer(
        model=model,
        dataloader=dataloader,
        layer_name=args.layer_name,
        alpha=args.alpha,
        device=device
    )
    if results is not None:
        plot_and_save(results, args.layer_name, args.output_dir, args.alpha)



if __name__ == '__main__':
    main()

