import time 
import heapq 
import torch 
import torch.nn as nn 

import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import os

from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 

# 用于存储补偿参数的局部字典（在函数内使用）
# COMPENSATION_PAaRAMS 将在函数内部创建，避免全局状态
#
# class CompensatedSparseLinear(nn.Module):
#     def __init__(self, original_linear_layer, compensation_params):
#         super().__init__()
#         # 基础层是稀疏的
#         self.sparse_linear = original_linear_layer
#
#         # 加载补偿参数
#         self.L1 = nn.Parameter(compensation_params['L1'], requires_grad=False) # (out_features, rank)
#         self.L2 = nn.Parameter(compensation_params['L2'], requires_grad=False)  # (rank, in_features)
#         self.s_inv = nn.Parameter(compensation_params['s_inv'], requires_grad=False)
#
#     def forward(self, x):
#             # 分支1: 原始的稀疏计算
#             output_sparse = self.sparse_linear(x)
#
#             # 分支2: 并行的低秩补偿计算
#             # 注意：x的形状可能是处理完整序列得到三维张量的 (batch_size, seq_len, input_dim)
#             # 或 生成单个新词元得到二维张量的(batch_size*seq_len, input_dim)
#             original_shape = x.shape
#             # 为了健壮性加的，可以删除：将输入重塑为二维矩阵以便进行矩阵乘法
#             if x.dim() > 2:
#                 x_2d = x.view(-1, x.shape[-1])  # (batch_size*seq_len, input_dim)
#             else:
#                 x_2d = x
#
#             # 确保设备一致性并应用smoothing，同时确保数据类型一致
#             x_smoothed = x_2d.to(self.s_inv.device, dtype=self.s_inv.dtype) * self.s_inv  # (batch*seq, in_features)
#
#             # 低秩补偿计算: x_smoothed @ (L1 @ L2)^T = x_smoothed @ L2^T @ L1^T
#             # L2^T: (in_features, rank), L1^T: (rank, out_features)
#             # 确保所有参数都在同一设备和数据类型
#             L2_t = self.L2.t().to(x_smoothed.device, dtype=x_smoothed.dtype)
#             L1_t = self.L1.t().to(x_smoothed.device, dtype=x_smoothed.dtype)
#
#             temp = x_smoothed @ L2_t   # (batch*seq, rank)
#             output_compensation = temp @ L1_t  # (batch*seq, out_features)
#
#             # 将补偿输出重塑回原始输出形状
#             if len(original_shape) > 2:
#                 # 需要重塑为与output_sparse相同的形状
#                 output_compensation = output_compensation.view(*original_shape[:-1], -1)
#
#             # 确保补偿输出与稀疏输出在同一设备和数据类型上
#             output_compensation = output_compensation.to(output_sparse.device, dtype=output_sparse.dtype)
#
#             # 合并结果
#             return output_sparse + output_compensation
#
#
#     def to(self, *args, **kwargs):
#         # 确保所有参数都被移动
#         super().to(*args, **kwargs)
#         self.sparse_linear.to(*args, **kwargs)
#         return self

class LowRankLinear(nn.Module):
    """A linear layer represented by low-rank factors with smoothing."""

    def __init__(self, final_weight_factors, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Load the low-rank factors and inverse scaling factor
        self.L1 = nn.Parameter(final_weight_factors['L1'], requires_grad=False)  # (out_features, rank)
        self.L2 = nn.Parameter(final_weight_factors['L2'], requires_grad=False)  # (rank, in_features)
        self.s_inv = nn.Parameter(final_weight_factors['s_inv'], requires_grad=False)  # (1, in_features)

    def forward(self, x):
        # The forward pass computes: (x * s_inv) @ (L1 @ L2)^T
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])

        # Ensure device and dtype consistency
        target_device = self.L1.device
        target_dtype = self.L1.dtype
        x = x.to(target_device, dtype=target_dtype)

        # Apply smoothing to the input activation
        x_smoothed = x * self.s_inv

        # Perform the low-rank matrix multiplication
        # (x @ L2^T) @ L1^T
        output = (x_smoothed @ self.L2.t()) @ self.L1.t()

        # Reshape output to match the original shape's batch/sequence dimensions
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], self.out_features)

        return output

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self



def create_final_compensated_model(structured_model, compensation_params):
    # 遍历模型的所有模块
    for name, module in structured_model.named_modules():
        if name in compensation_params:
            if isinstance(module, nn.Linear):
                print(f"Replacing layer: {name}")
                
                # 创建新的补偿层
                # 将补偿参数移动到与模型相同的设备和数据类型
                params = compensation_params[name]
                target_dtype = module.weight.dtype
                target_device = module.weight.device
                
                for k, v in params.items():
                    params[k] = v.to(device=target_device, dtype=target_dtype)

                new_layer = CompensatedSparseLinear(module, params)
                
                # 获取父模块并替换
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = structured_model.get_submodule(parent_name)
                setattr(parent_module, child_name, new_layer)
                
    return structured_model

def calculate_smoothing_scale_factor(activation_scales, delta_B, alpha=0.5):
    """
    根据激活尺度和误差矩阵计算Smoothing缩放因子s，然后作用于输入通道
    参数:
    - activation_scales: 形状为 (1, in_features)，代表每个输入通道的激活尺度。
    - delta_B: 形状为 (out_features, in_features)，即误差矩阵。
    - alpha: 平滑强度，一个0到1之间的超参数。
    
    返回:
    - s: 形状为 (1, in_features) 的缩放因子。
    """
    # 计算 delta_B 每一列的绝对值最大值
    # 对应 SmoothQuant 论文中的 max(|W_j|)
    delta_B_input_channel_max_abs = torch.max(torch.abs(delta_B), dim=0, keepdim=True)[0].clamp(min=1e-5)

    # 平滑公式
    # 为了防止除零，添加一个小的epsilon
    #epsilon = 1e-6
    #s = torch.pow(activation_scales, alpha) / (torch.pow(delta_B_input_channel_max_abs, 1 - alpha) + epsilon)
    #s = torch.pow(activation_scales, alpha) / (torch.pow(delta_B_input_channel_max_abs, 1 - alpha))
    s = torch.pow(activation_scales, alpha)
    # 如果一个通道在 delta_B 中全为零，
    # 那么它的缩放因子应该为1，即不进行缩放。
    #s[delta_B_input_channel_max_abs == 0] = 1.0


    # 对s进行裁剪，防止出现极端值
    s = torch.clamp(s, min=1e-5) 
    return s

def low_rank_approximation_factors(matrix, rank):
    """对矩阵进行SVD并返回低秩因子"""
    print(f"    Computing SVD factors for matrix shape: {matrix.shape}")
    
    # 如果rank为None，设置为矩阵最小维度的1/4
    if rank is None:
        rank = min(matrix.shape[0], matrix.shape[1]) // 4
    
    # 确保rank不超过矩阵的最小维度
    max_rank = min(matrix.shape[0], matrix.shape[1])
    rank = min(rank, max_rank)
    
    if rank <= 0:
        print(f"    Warning: rank {rank} is invalid, returning zero factors")
        return (torch.zeros(matrix.shape[0], 1, device=matrix.device, dtype=matrix.dtype), 
                torch.zeros(1, matrix.shape[1], device=matrix.device, dtype=matrix.dtype))
    
    print(f"    Using rank: {rank}")
    
    # 保存原始数据类型和设备
    original_dtype = matrix.dtype
    original_device = matrix.device
    try:
        # 转换为float32进行SVD计算，并移到CPU以节省GPU内存
        matrix_cpu = matrix.float().cpu()
        
        U, S, Vh = torch.linalg.svd(matrix_cpu, full_matrices=False)
        # --- SVD计算结束 ---
        
        # 截断到指定秩
        U_k = U[:, :rank]  # [m, rank]
        S_k = S[:rank]     # [rank]
        Vh_k = Vh[:rank]   # [rank, n]
        
        # 将奇异值分配给两个因子
        sqrt_S = torch.sqrt(S_k + 1e-10)  # 添加小常数以避免数值问题
        L1 = U_k @ torch.diag(sqrt_S)      # [m, rank]
        L2 = torch.diag(sqrt_S) @ Vh_k      # [rank, n]
        
        # 转换回原始数据类型，但保持在CPU上
        L1 = L1.to(dtype=original_dtype)
        L2 = L2.to(dtype=original_dtype)
        
        print(f"    Factor dimensions - L1: {L1.shape}, L2: {L2.shape}")
        print(f"    Successfully computed low-rank factors")
        return L1, L2
        
    except Exception as e:
        print(f"    Error in SVD factor computation: {e}")
        print(f"    Returning zero factors for safety")
        return (torch.zeros(matrix.shape[0], 1, dtype=original_dtype), 
                torch.zeros(1, matrix.shape[1], dtype=original_dtype))


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


def analyze_activation_scales(activation_scales, layer_index, layer_name):
    """分析并绘制激活尺度的直方图"""
    print(f"  - Analyzing activation_scales for layer {layer_name}...")
    if activation_scales is not None and activation_scales.numel() > 0:
        # 确保是一维张量以便统计
        scales_flat = activation_scales.flatten()

        # 计算核心统计指标
        min_val = torch.min(scales_flat).item()
        max_val = torch.max(scales_flat).item()
        mean_val = torch.mean(scales_flat).item()
        std_val = torch.std(scales_flat).item()

        print(f"    - Statistics for activation_scales:")
        print(f"      - Min:     {min_val:.6f}")
        print(f"      - Max:     {max_val:.6f}")
        print(f"      - Mean:    {mean_val:.6f}")
        print(f"      - Std Dev: {std_val:.6f}")

        # 关键判断
        if (max_val - min_val) < 1e-5 or std_val < 1e-5:
            print(f"    - [!!] WARNING: Scales are almost UNIFORM...")
        else:
            print(f"    - [OK] OK: Scales show significant variation, as expected.")

        # 绘制带有统计数值标注的直方图
        stats_text = (f'Max: {max_val:.4f}\n'
                      f'Min: {min_val:.4f}\n'
                      f'Mean: {mean_val:.4f}\n'
                      f'Std Dev: {std_val:.4f}')

        # 定义保存路径
        save_dir = "activation_scales_histograms"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"hist_layer_{layer_index}_{layer_name.replace('.', '_')}.png")

        # 开始绘图
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(scales_flat.cpu().numpy(), bins=100, log=True)

        # 设置标题和坐标轴标签
        ax.set_title(f'Activation Scales Distribution\nLayer {layer_index} - {layer_name}', fontsize=16)
        ax.set_xlabel('Scale Value', fontsize=12)
        ax.set_ylabel('Frequency (log scale)', fontsize=12)
        ax.grid(True, which="both", ls="--", linewidth=0.5)

        # 在图表的右上角添加文本框
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        # 保存并关闭图形
        plt.savefig(filepath)
        plt.close(fig)
        print(f"    - Histogram plot with stats saved to: {filepath}")
    else:
        print("    - [!!] ERROR: activation_scales is None or empty! Cannot perform analysis.")


def plot_dense_matrix_experiment(layer_index, layer_name, original_weight, activation_scales, args):
    """对稠密矩阵进行平滑实验并绘图"""
    print("\n  [TASK 2 EXPERIMENT] Testing smoothing on the original DENSE weight matrix...")

    # 1. 选择理想的测试对象: 使用原始的、未经剪枝的稠密权重
    dense_matrix_for_test = original_weight.clone()

    # 2. 应用平滑: 复用刚刚为该层计算出的同一个 activation_scales
    s_for_dense_test = calculate_smoothing_scale_factor(activation_scales, dense_matrix_for_test, alpha=args.alpha)
    dense_matrix_smoothed = dense_matrix_for_test * s_for_dense_test

    # 3. 对比和验证: 复用已有的 SVD 分析函数，独立生成一组对比图
    print(f"  - Analyzing SVD Spectrum for DENSE matrix vs. SMOOTHED DENSE matrix...")
    dense_save_dir = "svd_analysis_dense_test"
    os.makedirs(dense_save_dir, exist_ok=True)
    dense_filepath = os.path.join(dense_save_dir, f"svd_dense_test_layer_{layer_index}_{layer_name.replace('.', '_')}.png")

    # 创建一个新的图纸用于本次独立实验
    fig_dense, axes_dense = plt.subplots(2, 2, figsize=(20, 14))
    dense_title_prefix = f'DENSE TEST - Layer {layer_index} {layer_name}'
    fig_dense.suptitle(f'SVD Analysis on Dense Matrix (Task 2)\n{dense_title_prefix}', fontsize=16)

    # 上面一行: 分析 original_weight (平滑前)
    plot_svd_analysis(axes_dense[0], dense_matrix_for_test, "Original Dense Weight", dense_title_prefix)

    # 下面一行: 分析 dense_matrix_smoothed (平滑后)
    plot_svd_analysis(axes_dense[1], dense_matrix_smoothed, "Smoothed Dense Weight", dense_title_prefix)

    fig_dense.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(dense_filepath)
    plt.close(fig_dense)
    print(f"  - Dense matrix experiment plot saved to: {dense_filepath}")

    # 清理本次实验用的临时变量，避免影响后续流程
    del dense_matrix_for_test, dense_matrix_smoothed, s_for_dense_test, fig_dense, axes_dense
    torch.cuda.empty_cache()
    print("  [TASK 2 EXPERIMENT] Dense matrix test completed.\n")


def smooth_and_compensate(delta_B, activation_scales, args):
    """执行平滑处理并返回平滑后的矩阵和逆缩放因子"""
    # 计算缩放因子 s 和其倒数 s_inv
    s = calculate_smoothing_scale_factor(activation_scales, delta_B, alpha=args.alpha)
    s_inv = 1.0 / s
    
    # 对 delta_B 进行变换
    delta_B_smoothed = delta_B * s
    
    return delta_B_smoothed, s_inv


def analyze_delta_B_and_plot_svd(delta_B, delta_B_smoothed, layer_index, layer_name):
    """分析 delta_B 并绘制 SVD 对比图"""
    # 分析 delta_B
    print(f"    - Analyzing delta_B for layer {layer_name}...")
    delta_B_col_max_abs = torch.max(torch.abs(delta_B), dim=0)[0]
    zero_cols_count = (delta_B_col_max_abs == 0).sum().item()

    if zero_cols_count > 0:
        print(f"    - WARNING: Found {zero_cols_count} columns in delta_B that are entirely zero.")
    else:
        print("    - OK: No all-zero columns found in delta_B.")

    # SVD 分析和绘图
    print(f"  Step 3b: Analyzing SVD Spectrum for delta_B vs delta_B_smoothed...")
    
    # 设置保存路径
    save_dir = "svd_analysis_plots_2x2"
    os.makedirs(save_dir, exist_ok=True)
    full_module_path = f"model.layers.{layer_index}.{layer_name}"
    filename_part = full_module_path.replace('.', '_')
    filepath = os.path.join(save_dir, f"svd_comparison_{filename_part}.png")

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    title_prefix = f'Layer {layer_index} Sublayer {layer_name}'
    fig.suptitle(f'SVD Analysis Comparison\n{title_prefix}', fontsize=16)

    # 上面一行：分析 delta_B (平滑前)
    plot_svd_analysis(axes[0], delta_B, "delta_B (Before Smoothing)", title_prefix)

    # 下面一行：分析 delta_B_smoothed (平滑后)
    plot_svd_analysis(axes[1], delta_B_smoothed, "delta_B_smoothed (After Smoothing)", title_prefix)

    # 保存并关闭图形
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局为总标题留出空间
    plt.savefig(filepath)
    plt.close(fig)
    print(f"    2x2 Comparison plot saved to: {filepath}")


def check_nm_sparsity_ratio(matrix, n, m, dimension='row'):
    """
    计算一个矩阵在多大程度上满足N:M稀疏约束。

    Args:
        matrix (torch.Tensor or np.ndarray): 要分析的矩阵。
        n (int): N:M约束中的N值（最多允许的非零数）。
        m (int): N:M约束中的M值（块大小）。
        dimension (str): 要检查的维度, 'row' 或 'col'。

    Returns:
        float: 满足N:M约束的块所占的比例 (0.0 to 1.0)。
    """
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.cpu().numpy()
    else:
        matrix_np = np.asarray(matrix)

    if dimension == 'col':
        # 如果检查列，则转置矩阵，后续逻辑统一按行处理
        matrix_np = matrix_np.T

    rows, cols = matrix_np.shape

    if m <= 0:
        print("Error: M (block size) must be positive.")
        return 0.0

    total_blocks = 0
    compliant_blocks = 0

    for i in range(rows):
        row = matrix_np[i, :]
        # 使用 // 确保处理完整的块
        num_blocks_in_row = cols // m
        for j in range(num_blocks_in_row):
            block = row[j * m: (j + 1) * m]
            non_zeros_in_block = np.count_nonzero(block)

            if non_zeros_in_block <= n:
                compliant_blocks += 1
            total_blocks += 1

    if total_blocks == 0:
        return 1.0

    compliance_ratio = compliant_blocks / total_blocks

    print(f"--- N:M Sparsity Compliance Analysis ---")
    print(f"Pattern: {n}:{m} Sparsity")
    print(f"Dimension: {dimension}")
    print(f"Compliant Blocks: {compliant_blocks:,} / {total_blocks:,}")
    print(f"Compliance Ratio: {compliance_ratio:.4f} ({compliance_ratio:.2%})")
    print("-" * 38)

    return compliance_ratio


def analyze_delta_B_key_patterns(delta_B, layer_index, layer_name, save_dir="pattern_analysis"):
    """
    简化版：分析 ΔB 矩阵中非零元素分布的关键模式
    """
    print(f"  [PATTERN ANALYSIS] Analyzing key distribution patterns in delta_B matrix...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为numpy数组
    if isinstance(delta_B, torch.Tensor):
        delta_B_np = delta_B.detach().cpu().numpy()
    else:
        delta_B_np = delta_B
    
    # 创建2x2子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 基本统计
    abs_matrix = np.abs(delta_B_np)
    threshold = 1e-8  # 设定阈值区分真正的零和非零
    nonzero_mask = abs_matrix > threshold
    sparsity = 1.0 - np.mean(nonzero_mask)
    
    # 1. 非零元素分布热图（最重要）
    im1 = ax1.imshow(nonzero_mask.astype(int), cmap='RdBu_r', aspect='auto', interpolation='nearest')
    ax1.set_title(f'Non-zero Distribution Pattern\n(Red=NonZero, Blue=Zero)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column Index (Input Features)')
    ax1.set_ylabel('Row Index (Output Features)')
    
    # 添加稀疏度信息
    ax1.text(0.02, 0.98, f'Sparsity: {sparsity:.3f}\n({sparsity*100:.1f}%)', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
    
    # 2. 行方向密度分析 - 查看是否某些输出特征更重要
    row_density = np.mean(nonzero_mask, axis=1)
    ax2.plot(row_density, range(len(row_density)), 'b-', linewidth=1.5)
    ax2.fill_betweenx(range(len(row_density)), 0, row_density, alpha=0.3, color='blue')
    ax2.set_title('Row-wise Non-zero Density\n(Output Feature Importance)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Non-zero Ratio')
    ax2.set_ylabel('Row Index (Output Features)')
    ax2.grid(True, alpha=0.3)
    
    # 标注最重要的几行
    top_rows = np.argsort(row_density)[-5:]  # 前5个最密集的行
    for row_idx in top_rows:
        ax2.axhline(y=row_idx, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
    
    # 3. 列方向密度分析 - 查看是否某些输入特征更重要
    col_density = np.mean(nonzero_mask, axis=0)
    ax3.plot(col_density, 'r-', linewidth=1.5)
    ax3.fill_between(range(len(col_density)), 0, col_density, alpha=0.3, color='red')
    ax3.set_title('Column-wise Non-zero Density\n(Input Feature Importance)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Column Index (Input Features)')
    ax3.set_ylabel('Non-zero Ratio')
    ax3.grid(True, alpha=0.3)
    
    # 标注最重要的几列
    top_cols = np.argsort(col_density)[-10:]  # 前10个最密集的列
    for col_idx in top_cols:
        ax3.axvline(x=col_idx, color='blue', linestyle='--', alpha=0.7, linewidth=0.8)
    
    # 4. 统计摘要和模式结论
    ax4.axis('off')
    
    # 计算关键统计量
    row_std = np.std(row_density)
    col_std = np.std(col_density)
    max_row_density = np.max(row_density)
    max_col_density = np.max(col_density)
    
    # 分析模式
    patterns = []
    if row_std > 0.05:
        patterns.append(f"• Row variation: HIGH (std={row_std:.3f})")
        patterns.append("  → Some output features are much more affected")
    else:
        patterns.append(f"• Row variation: LOW (std={row_std:.3f})")
        patterns.append("  → Output features equally affected")
        
    if col_std > 0.05:
        patterns.append(f"• Column variation: HIGH (std={col_std:.3f})")
        patterns.append("  → Some input features are much more important")
    else:
        patterns.append(f"• Column variation: LOW (std={col_std:.3f})")
        patterns.append("  → Input features equally important")
    
    # 查看是否有明显的聚集模式
    if max_row_density > 3 * np.mean(row_density):
        patterns.append("• HOTSPOT detected in output features")
    if max_col_density > 3 * np.mean(col_density):
        patterns.append("• HOTSPOT detected in input features")
    
    # 检查边缘vs中心模式
    edge_size = min(20, min(delta_B_np.shape) // 10)
    if edge_size > 0:
        center_density = np.mean(nonzero_mask[edge_size:-edge_size, edge_size:-edge_size])
        edge_density = (np.mean(nonzero_mask[:edge_size, :]) + 
                       np.mean(nonzero_mask[-edge_size:, :]) + 
                       np.mean(nonzero_mask[:, :edge_size]) + 
                       np.mean(nonzero_mask[:, -edge_size:])) / 4
        
        if center_density > 1.5 * edge_density:
            patterns.append("• CENTER-concentrated pattern")
        elif edge_density > 1.5 * center_density:
            patterns.append("• EDGE-concentrated pattern")
        else:
            patterns.append("• UNIFORM distribution")
    
    summary_text = f"""KEY PATTERN ANALYSIS

Matrix Shape: {delta_B_np.shape}
Overall Sparsity: {sparsity:.3f} ({sparsity*100:.1f}%)

DISTRIBUTION PATTERNS:
{chr(10).join(patterns)}

TOP AFFECTED FEATURES:
• Most important output rows: {top_rows[-3:]}
• Most important input cols: {top_cols[-5:]}

CONCLUSION:
"""
    
    if row_std > 0.05 and col_std > 0.05:
        conclusion = "STRUCTURED compensation needed\n- Both input and output show selectivity"
    elif row_std > 0.05:
        conclusion = "OUTPUT-selective compensation\n- Specific output features targeted"
    elif col_std > 0.05:
        conclusion = "INPUT-selective compensation\n- Specific input features important"
    else:
        conclusion = "UNIFORM compensation\n- No clear selectivity pattern"
    
    summary_text += conclusion
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # 设置总标题
    fig.suptitle(f'ΔB Matrix Key Pattern Analysis\nLayer {layer_index} - {layer_name}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 保存图像
    filename = f"key_patterns_layer_{layer_index}_{layer_name.replace('.', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Key pattern analysis saved to: {filepath}")
    print(f"    Pattern summary: Row_std={row_std:.3f}, Col_std={col_std:.3f}")
    print(f"    Conclusion: {conclusion.split('-')[0].strip()}")
    
    return filepath

#
# def process_layer_compensation(layer_index, layer_name, original_weight, wrapped_layer, args, dev, prune_n=0, prune_m=0):
#     """处理单个层的补偿逻辑"""
#     print(f"Processing layer {layer_index} sublayer {layer_name}")
#
#     W_metric = torch.abs(original_weight) * torch.sqrt(wrapped_layer.scaler_row.reshape((1, -1)))
#
#     print(f"  Step 1: Computing unstructured pruning target...")
#     # 计算非结构化剪枝目标
#     W_mask_unstructured = (torch.zeros_like(W_metric) == 1)
#     sort_res = torch.sort(W_metric, dim=-1, stable=True)
#     indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
#     W_mask_unstructured.scatter_(1, indices, True)
#
#     B_unstructured = original_weight.clone()
#     B_unstructured[W_mask_unstructured] = 0
#     print(f"    B_unstructured sparsity: {(B_unstructured == 0).float().mean():.6f}")
#
#
#     print(f"  Step 2: Computing structured pruning result...")
#     # 计算结构化剪枝目标
#     W_mask_structured = (torch.zeros_like(W_metric) == 1)
#     if prune_n != 0:
#         for ii in range(W_metric.shape[1]):
#             if ii % prune_m == 0:
#                 tmp = W_metric[:, ii:(ii + prune_m)].float()
#                 W_mask_structured.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
#     else:
#         # 如果没有指定n:m，则结构化掩码与非结构化掩码相同
#         W_mask_structured = W_mask_unstructured.clone()
#
#     B_structured = original_weight.clone()
#     B_structured[W_mask_structured] = 0
#     print(f"    B_structured sparsity: {(B_structured == 0).float().mean():.6f}")
#
#     print(f"  Step 3: Computing compensation matrix (delta_B)...")
#     delta_B = B_unstructured - B_structured
#
#     # ======================= 在这里添加 N:M 分析调用 =======================
#     print(f"\n  Step 3a: Analyzing N:M sparsity compliance for delta_B...")
#
#     # 调用您想测试的N:M组合
#     # 检查行维度上的 2:4 稀疏满足度
#     #check_nm_sparsity_ratio(delta_B, n=2, m=4, dimension='row')
#
#     # 检查行维度上的 4:8 稀疏满足度
#     #check_nm_sparsity_ratio(delta_B, n=4, m=8, dimension='row')
#
#     # 您还可以添加其他想测试的组合
#     print("-" * 50)
#     # ====================================================================
#
#     # ======================= 新增：生成delta_B的spy图 =======================
#     #print(f"  Step 3b: Analyzing key patterns in delta_B matrix...")
#     #analyze_delta_B_key_patterns(delta_B, layer_index, layer_name)
#     # ====================================================================
#
#     compensation_params = None
#
#     if torch.norm(delta_B) > 1e-8:
#         print(f"  Step 3a: Applying Smoothing to delta_B...")
#
#         try:
#             # 1. 获取激活尺度
#             activation_scales = wrapped_layer.act_scales.to(dev)
#
#             # 分析激活尺度
#             analyze_activation_scales(activation_scales, layer_index, layer_name)
#
#             # 确保activation_scales是正确的形状 (1, in_features)
#             if activation_scales.dim() == 1:
#                 activation_scales = activation_scales.unsqueeze(0)
#
#             # 稠密矩阵实验
#             #plot_dense_matrix_experiment(layer_index, layer_name, original_weight, activation_scales, args)
#
#             # ======================= 任务三：新增的核心假说验证代码块 开始 =======================
#             """
#             print("\n  [TASK 3 ANALYSIS] Correlating activation scales with zero-columns in delta_B...")
#
#             # 1. 找到 activation_scales 最大的 Top-K 通道的索引
#             top_k = 50  # 我们可以关注最重要的50个通道
#             s_flat = activation_scales.flatten()
#             top_k_scales_val, top_k_scales_indices = torch.topk(s_flat, top_k)
#
#             # 2. 找到 delta_B 中所有全零列的索引
#             delta_B_col_max_abs = torch.max(torch.abs(delta_B), dim=0)[0]
#             zero_col_indices = torch.where(delta_B_col_max_abs == 0)[0]
#
#             # 3. 量化分析: 计算重合度
#             # 将索引转换为集合(set)以便快速查找
#             zero_col_set = set(zero_col_indices.cpu().numpy())
#             top_k_set = set(top_k_scales_indices.cpu().numpy())
#
#             overlap_indices = top_k_set.intersection(zero_col_set)
#             overlap_count = len(overlap_indices)
#
#             print(f"  - Analysis Result: Out of the Top {top_k} most important channels (by activation scale):")
#             print(
#                 f"  - {overlap_count} ({overlap_count / top_k * 100:.2f}%) of them correspond to an all-zero column in delta_B.")
#
#             if overlap_count / top_k > 0.5:  # 如果超过一半都重合，说明假说很可能成立
#                 print("  - [!!] CRITICAL FINDING: A significant overlap was found. The hypothesis is likely correct.")
#             else:
#                 print("  - NOTE: The overlap is not significant.")
#
#             # 4. 可视化分析: 绘制相关性散点图
#             print("  - Generating correlation scatter plot for Task 3...")
#
#             # 准备绘图数据
#             x_data = activation_scales.flatten().cpu().numpy()
#             y_data = delta_B_col_max_abs.cpu().numpy()
#
#             # 创建图纸
#             fig, ax = plt.subplots(figsize=(12, 8))
#
#             # 绘制散点图
#             ax.scatter(x_data, y_data, alpha=0.5, s=10)
#
#             ax.set_yscale('log')
#             ax.set_xlabel('Activation Scale Value (Per-Input-Channel)', fontsize=12)
#             ax.set_ylabel('Max Abs Value of delta_B Column (log scale)', fontsize=12)
#             ax.set_title(
#                 f'Correlation between Activation Scales and delta_B Column Norms\nLayer {layer_index} - {layer_name}',
#                 fontsize=16)
#             ax.grid(True, which="both", ls="--", linewidth=0.5)
#
#             # 保存图表
#             scatter_save_dir = "correlation_analysis_plots"
#             os.makedirs(scatter_save_dir, exist_ok=True)
#             scatter_filepath = os.path.join(scatter_save_dir,
#                                             f"corr_layer_{layer_index}_{layer_name.replace('.', '_')}.png")
#             plt.savefig(scatter_filepath)
#             plt.close(fig)
#             print(f"  - Correlation plot saved to: {scatter_filepath}\n")
#             # ======================= 任务三：验证代码块 结束 =======================
#             """
#
#             # 平滑处理
#             delta_B_smoothed, s_inv = smooth_and_compensate(delta_B, activation_scales, args)
#
#             # 分析和绘图
#             #analyze_delta_B_and_plot_svd(delta_B, delta_B_smoothed, layer_index, layer_name)
#
#             # 清理中间变量以节省内存
#             del activation_scales
#             torch.cuda.empty_cache()
#
#             # 计算低秩因子
#             print(f"  Step 4: Computing low-rank factors...")
#             compensation_rank = getattr(args, 'compensation_rank', None)
#
#             if compensation_rank is None:
#                 compensation_rank = min(delta_B_smoothed.shape[0], delta_B_smoothed.shape[1]) // 4
#
#             L1, L2 = low_rank_approximation_factors(delta_B_smoothed, rank=compensation_rank)
#
#             # 保存参数
#             layer_key = f"model.layers.{layer_index}.{layer_name}"
#             compensation_params = {
#                 'L1': L1.cpu(),      # 移到CPU以节省GPU内存
#                 'L2': L2.cpu(),
#                 's_inv': s_inv.cpu()
#             }
#             print(f"    Compensation parameters for {layer_key} have been generated and stored.")
#
#             # 清理GPU内存
#             del L1, L2, s_inv, delta_B_smoothed
#             torch.cuda.empty_cache()
#
#         except Exception as e:
#             print(f"    Error in smoothing/compensation: {e}")
#             print(f"    Skipping compensation for this layer.")
#     else:
#         print(f"    No compensation needed (delta_B norm is too small).")
#
#     return B_structured, compensation_params
#

def process_layer_compensation(layer_index, layer_name, original_weight, wrapped_layer, args, dev, prune_n=0, prune_m=0):
    """
    Directly applies smoothing and SVD to the dense weight matrix (W_dense).
    """
    print(f"Processing layer {layer_index} sublayer {layer_name} with new SVD logic")

    final_weight_factors = None

    try:
        # Step 1: Get activation scales for smoothing.
        print(f"  Step 1: Retrieving activation scales...")
        activation_scales = wrapped_layer.act_scales.to(dev)

        if activation_scales.dim() == 1:
            activation_scales = activation_scales.unsqueeze(0)

        # Step 2: Apply smoothing to the original dense weight matrix.
        # We reuse `smooth_and_compensate` but pass `original_weight` instead of `delta_B`.
        print(f"  Step 2: Applying smoothing to the dense weight matrix...")
        W_dense_smoothed, s_inv = smooth_and_compensate(original_weight, activation_scales, args)
        print(f"    Smoothing complete.")

        # Clean up memory
        del activation_scales
        torch.cuda.empty_cache()

        # Step 3: Compute low-rank factors using SVD.
        print(f"  Step 3: Computing low-rank factors via SVD...")
        # You can control the rank via an argument, e.g., args.compensation_rank
        final_rank = getattr(args, 'compensation_rank', 64) # 如果没拿到，最后一个参数64作为默认值
        print(f"    Target rank for final weight: {final_rank}")
        
        L1, L2 = low_rank_approximation_factors(W_dense_smoothed, rank=final_rank)

        # Store the factors that will represent the entire layer.
        layer_key = f"model.layers.{layer_index}.{layer_name}"
        final_weight_factors = {
            'L1': L1.cpu(),
            'L2': L2.cpu(),
            's_inv': s_inv.cpu()
        }
        print(f"    Low-rank factors for {layer_key} have been generated and stored.")

        # Clean up GPU memory
        del L1, L2, s_inv, W_dense_smoothed
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"    An error occurred during smoothing or SVD for layer {layer_name}: {e}")
        print(f"    Skipping factorization for this layer.")

    # Return None for the first value (as there's no B_structured) and the new factors.
    return None, final_weight_factors



def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module. #默认找nn.Linear类型的线性层

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    #下面是递归终止条件：
    if type(module) in layers:
        return {name: module}


    res = {}
    for name1, child in module.named_children(): #named_children返回当前所有直接子模块的迭代器
        #update将新的键值对添加到res字典中，如果键已存在，则更新其值
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, nsamples, seqlen):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        layers = model.model.layers
    except AttributeError:
        try:
            layers = model.base_model.layers
        except AttributeError:
            layers = model.base_model.decoder.layers
    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]
    else:
        device = model.device
    dtype = next(iter(model.parameters())).dtype
    # dtype = torch.float
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "cache_position": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "self_attn"):
                self.self_attn = module.self_attn
            elif hasattr(module, "attn"):
                self.attn = module.attn

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            if 'cache_position' in kwargs and 'position_embeddings' in kwargs:
                cache['cache_position'] = kwargs['cache_position']
                cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(torch.rand_like(batch[0].to(device)))
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    if 'cache_position' in cache and 'position_embeddings' in cache:
        cache_position = cache['cache_position']
        position_embeddings = cache['position_embeddings']
        return inps, outs, attention_mask, position_ids, cache_position, position_embeddings
    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    #alpha是剪枝比例因子
    thres_cumsum = sum_before * alpha #为每一行设置一个剪枝预算： sum_before * alpha
    #移除这一行里最不重要的权重，直到它们的W_metric之和达到每行的剪枝预算
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))

    #sort_mask.sum(dim=1, keepdims=True) 计算每行有多少个True，最后的-1是为了获取最后一个True的索引
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    #thres是每一行的剪枝阈值，表示每一行中最大的W_metric值，超过这个值的权重将被剪枝
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel() #.numel返回张量的元素总数， number of element
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W) #根据权重的绝对值进行剪枝
            if prune_n != 0:
                # 结构化剪枝：n:m 稀疏性
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                #非结构化剪枝
                #torch.sort(W_metric.flatten().cuda())[0]返回值表示拿到排序后的值，flatten()将W_metric展平为一维
                #然后根据W.numel()*args.sparsity_ratio得到要剪枝的门槛，注意这里阈值元素只有一个
                #之后W_metric<=thresh，表示哪些元素小于等于阈值
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        #拿到每个token的输入向量表示，outs是输出向量表示，attention_mask是注意力掩码，position_ids是位置编码
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)


    layers = model.model.layers
    for i in range(len(layers)):
        #layer指的是拿到一整个大块的LlamaDecoderLayer
        layer = layers[i]
        subset = find_layers(layer) #递归找这个layer模块下的线性层

        ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            #一个layer的所有线性层注册钩子函数，捕获每个样本的输入和输出
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            #注册完钩子函数后，开始进行这批样本的前向传播（无梯度更新）,这样钩子函数能够捕获到每个样本的输入和输出
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0] #相当于layer.forward(第j个样本),最后【0】拿到返回值第一个值，输出
        #移除钩子函数
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # 结构化剪枝：n:m 稀疏性
                # structured n:m sparsity, 对权重矩阵 W_metric进行剪枝
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        #topk返回的是最小的prune_n个元素的值和索引, 最后访问[1]拿到索引
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                #use_variant : whether to use the wanda variant described in the appendix
                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1) #求累积和
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            #当前稀疏度大于目标稀疏度，说明alpha太大了，需要减小， alpha_hist[1]用来放较大的alpha值
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            #当前稀疏度小于目标稀疏度，说明alpha太小了，需要增大， alpha_hist[0]用来放较小的alpha值
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    #sort_res每一行是由小到大排列，[1]拿到索引， 按照每行的前args.sparsity_ratio比例进行剪枝
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            #配合W_mask完成剪枝置0
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 


        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
#上面这个装饰器表示以下函数的所有计算都不会追踪梯度，可以节省显存和计算资源
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # 如果模型的 token embeddings 在特定设备上，则将 dev 设置为该设备
    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    #捕获模型原始输入经过embedding层后的表示， 同时拿到attention_mask和position_ids
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    # 将模型的第一层替换为 Catcher 实例
    layers[0] = Catcher(layers[0])
    #遍历dataloader中的每个batch，通过model.forward()捕获输入
    for batch in dataloader:
        try:
            model(batch[0].to(dev)) #这里相当于 model.forward(batch[0].to(dev)),Cather就会捕捉
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        # 如果当前层被分配到特定的设备上，则将所有相关张量移动到该设备
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        # 为每个 SparseGPT 实例注册前向钩子，以捕获输入和输出
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            #核心是fasterprune函数，跳进去演示
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_with_compensation(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    执行带有补偿逻辑的Wanda剪枝并返回补偿参数
    调用以下模块完成：
    - analyze_activation_scales: 分析激活尺度
    - plot_dense_matrix_experiment: 稠密矩阵实验
    - smooth_and_compensate: 平滑处理
    - analyze_delta_B_and_plot_svd: SVD分析和绘图
    - process_layer_compensation: 处理单个层的补偿逻辑
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # 用于存储补偿参数的局部字典
    compensation_params = {}
    
    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        #inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        returned_data = prepare_calibration_input(model, dataloader, args.nsamples, model.seqlen)
        inps, outs, attention_mask, position_ids = returned_data[:4]

    layers = model.model.layers
    
    for i in range(len(layers)):
        print(f"\n=== Processing Layer {i} ===")
        layer = layers[i]
        subset = find_layers(layer)

        # 设置设备，默认使用传入的device参数
        dev = device
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)


        # 收集激活值
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        rotary_emb = model.model.rotary_emb
        if position_ids is None:
            seq_len = inps.shape[1]
            position_ids = torch.arange(seq_len, device=dev).unsqueeze(0)


        for j in range(args.nsamples):
            with torch.no_grad():
                position_embeddings = rotary_emb(inps[j], position_ids)
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )[0]


        for h in handles:
            h.remove()

        # 对每个线性层进行剪枝
        for name in subset:
            original_weight = subset[name].weight.data.clone()
            in_features = subset[name].in_features
            out_features = subset[name].out_features
            # 使用模块化的处理函数
            #B_structured, layer_compensation_params = process_layer_compensation(
            #    i, name, original_weight, wrapped_layers[name], args, dev, prune_n, prune_m
            #)

            # 线性层的权重被设置为纯粹的、稀疏的结构化剪枝结果
            #subset[name].weight.data = B_structured

            # 如果有补偿参数，保存到总的字典中
            # if layer_compensation_params is not None:
            #     print(f"Replacing layer: model.layers.{i}.{name}")
            #     params = layer_compensation_params
            #     new_layer = CompensatedSparseLinear(subset[name], params)
            #     parent_name, child_name = name.rsplit('.', 1)
            #     parent_module = model.get_submodule(f"model.layers.{i}.{parent_name}")
            #     setattr(parent_module, child_name, new_layer)
            #print(f"    Layer weight is set to the sparse structured matrix.")
            _, final_weight_factors = process_layer_compensation(
                i, name, original_weight, wrapped_layers[name], args, dev, prune_n, prune_m
            )
            subset[name].weight.data = original_weight
            if final_weight_factors is not None:
                print(f"  -> Replacing layer 'model.layers.{i}.{name}' with LowRankLinear.")
                # Create the new layer with the computed factors.
                new_layer = LowRankLinear(final_weight_factors, in_features, out_features)
                # Get the parent module to perform the replacement.
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(f"model.layers.{i}.{parent_name}")
                setattr(parent_module, child_name, new_layer)
            

        # 重新计算层输出
        for j in range(args.nsamples):
            with torch.no_grad():
                position_embeddings = rotary_emb(inps[j], position_ids)
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("\n=== Wanda compensation completed ===")
    return model



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m,
                                   percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()