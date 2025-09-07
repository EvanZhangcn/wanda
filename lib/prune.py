import time 
import heapq 
import torch 
import torch.nn as nn 

import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import os
import gc
from torch.sparse import to_sparse_semi_structured

from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 

# 用于存储补偿参数的局部字典（在函数内使用）
# COMPENSATION_PAaRAMS 将在函数内部创建，避免全局状态

class CompensatedSparseLinear(nn.Module):
    def __init__(self, original_linear_layer, compensation_params):
        super().__init__()
        # 基础层是稀疏的
        self.sparse_linear = original_linear_layer
        
        # 从加载补偿参数 修改为 加载CSR格式的deltaB矩阵
        self.register_buffer('delta_B_csr', compensation_params['delta_B_csr'])
        
    def forward(self, x):
            # 分支1: 原始的结构化剪枝后矩阵的稀疏计算
            output_sparse = self.sparse_linear(x)

            # 分支2: 并行的低秩补偿计算
            original_shape = x.shape

            # 为了健壮性加的，可以删除：将输入重塑为二维矩阵以便进行矩阵乘法
            if x.dim() > 2:
                x_2d = x.view(-1, x.shape[-1])  # (batch_size*seq_len, input_dim)
            else:
                x_2d = x

            delta_B_csr_casted = self.delta_B_csr.to(device=x_2d.device, dtype=x.dtype)
            output_compensation = torch.sparse.mm(delta_B_csr_casted, x_2d.t()).t()

            # 将补偿输出重塑回原始输出形状
            if len(original_shape) > 2:
                # 需要重塑为与output_sparse相同的形状
                output_compensation = output_compensation.view(*original_shape[:-1], -1)
            
            # 确保补偿输出与稀疏输出在同一设备和数据类型上
            output_compensation = output_compensation.to(output_sparse.device, dtype=output_sparse.dtype)
            
            # 合并结果
            return output_sparse + output_compensation


    def to(self, *args, **kwargs):
        # 确保所有参数都被移动
        super().to(*args, **kwargs)
        self.sparse_linear.to(*args, **kwargs)
        return self


class CompensatedSparseLinear24(nn.Module):
    def __init__(self, original_linear_layer, compensation_params):
        super().__init__()
        # 分支1: 原始的2:4稀疏层
        self.sparse_linear = original_linear_layer

        # 核心修改：初始化时将稠密ΔB转换为2:4稀疏格式
        delta_B_dense = compensation_params['delta_B_dense']
        delta_B_sparse_2_4 = to_sparse_semi_structured(delta_B_dense)
        self.delta_B = nn.Parameter(delta_B_sparse_2_4, requires_grad=False)

    def forward(self, x):
        output_sparse = self.sparse_linear(x)
        original_shape = x.shape
        if x.dim() > 2:
            x_2d = x.view(-1, x.shape[-1])
        else:
            x_2d = x

        # 直接使用 torch.mm，PyTorch会自动调用2:4稀疏的硬件加速内核
        output_compensation = torch.mm(self.delta_B, x_2d.t()).t()

        if len(original_shape) > 2:
            output_compensation = output_compensation.view(*original_shape[:-1], -1)

        output_compensation = output_compensation.to(output_sparse.device, dtype=output_sparse.dtype)
        return output_sparse + output_compensation

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.sparse_linear.to(*args, **kwargs)
        return self
    
    
    
    
def check_nm_sparsity_ratio(matrix, n, m, dimension='row'):
    """
    计算一个矩阵在多大程度上满足N:M稀疏约束。
    Args:
        matrix (torch.Tensor or np.ndarray): 要分析的矩阵。
        n (int): N:M约束中的N值（最多允许的非零数）。
        m (int): N:M约束中的M值（块大小）。
        dimension (str): 要检查的维度, 'row' 或 'col'。
    Returns:
        float: 满足N:M约束的块所占的比例 (0-1)。
    """
    #转numpy
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
    compliant_blocks = 0 #服从2:4的块有多少

    #沿着行维度，实际操作就是每m列检查是否至少有n个0
    for i in range(rows):
        row = matrix_np[i, :]
        # 使用 // 确保处理完整的块
        num_blocks_in_row = cols // m
        for j in range(num_blocks_in_row):
            #每次间隔为m列
            block = row[j * m: (j + 1) * m]
            non_zeros_in_block = np.count_nonzero(block)

            if non_zeros_in_block <= n:
                compliant_blocks += 1
            total_blocks += 1

    # 特殊情况：如果没有块，则返回1.0，表示完全合规
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


def process_layer_compensation(layer_index, layer_name, original_weight, wrapped_layer, args, dev, prune_n=0,
                               prune_m=0):
    """处理单个层的补偿逻辑"""
    print(f"Processing layer {layer_index} sublayer {layer_name}")

    W_metric = torch.abs(original_weight) * torch.sqrt(wrapped_layer.scaler_row.reshape((1, -1)))

    print(f"  Step 1: Computing unstructured pruning target...")
    # --- 步骤 1: 计算非结构化剪枝后的矩阵 B_unstructured ---
    W_mask_unstructured = (torch.zeros_like(W_metric) == 1)
    sort_res = torch.sort(W_metric, dim=-1, stable=True)
    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
    W_mask_unstructured.scatter_(1, indices, True)
    B_unstructured = original_weight.clone()
    B_unstructured[W_mask_unstructured] = 0
    print(f"    B_unstructured sparsity: {(B_unstructured == 0).float().mean():.6f}")

    # --- 步骤 2: 计算2:4结构化剪枝后的矩阵 B_structured ---
    print(f"  Step 2: Computing structured pruning result...")
    # 计算结构化剪枝目标
    W_mask_structured = (torch.zeros_like(W_metric) == 1)
    if prune_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii:(ii + prune_m)].float()
                W_mask_structured.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
    else:
        # 如果没有指定n:m，则结构化掩码与非结构化掩码相同
        W_mask_structured = W_mask_unstructured.clone()
    B_structured = original_weight.clone()
    B_structured[W_mask_structured] = 0
    print(f"    B_structured sparsity: {(B_structured == 0).float().mean():.6f}")

    # --- 步骤 3: 计算补偿矩阵 ΔB ---
    print(f"  Step 3: Computing compensation matrix (delta_B)...")
    delta_B = B_unstructured - B_structured
    # --- 步骤 4: 将 ΔB 转换为 CSR 格式 ---
    compensation_params = None
    if torch.norm(delta_B) > 1e-8:
        print(f"  Step 4: Converting delta_B to CSR format...")

        #后来怀疑：精度有损失是这里改变了数值，所以建议注释掉，再运行
        # 设置一个阈值，将非常接近零的数值彻底清零，以最大化稀疏性
        #threshold = 1e-8
        #delta_B[torch.abs(delta_B) < threshold] = 0.0

        # 转换为 CSR 格式
        delta_B_csr = delta_B.to_sparse_csr()

        print(f"    Successfully converted. CSR non-zero elements: {delta_B_csr.values().numel()}")

        # 准备要返回的补偿参数字典
        compensation_params = {
            'delta_B_csr': delta_B_csr,
        }
        print(f"    Compensation parameters created.")

        # 清理GPU内存
        # del delta_B_csr
        # torch.cuda.empty_cache()
    else:
        print(f"    No compensation needed (delta_B norm is too small).")

    return B_structured, compensation_params


def process_layer_compensation_dense(layer_index, layer_name, original_weight, wrapped_layer, args, dev, prune_n=0,
                                     prune_m=0):
    """处理单个层的补偿逻辑（返回稠密ΔB版本）"""

    # 这部分逻辑与原函数完全相同
    W_metric = torch.abs(original_weight) * torch.sqrt(wrapped_layer.scaler_row.reshape((1, -1)))

    W_mask_unstructured = (torch.zeros_like(W_metric) == 1)
    sort_res = torch.sort(W_metric, dim=-1, stable=True)
    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
    W_mask_unstructured.scatter_(1, indices, True)
    B_unstructured = original_weight.clone()
    B_unstructured[W_mask_unstructured] = 0

    W_mask_structured = (torch.zeros_like(W_metric) == 1)
    if prune_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii:(ii + prune_m)].float()
                W_mask_structured.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
    else:
        W_mask_structured = W_mask_unstructured.clone()
    B_structured = original_weight.clone()
    B_structured[W_mask_structured] = 0

    delta_B = B_unstructured - B_structured

    # 核心区别：不再转换为CSR，直接返回稠密张量
    compensation_params = None
    if torch.norm(delta_B) > 1e-8:
        compensation_params = {
            'delta_B_dense': delta_B,
        }

    return B_structured, compensation_params


def process_layer_selective_compensation(layer_index, layer_name, original_weight, wrapped_layer, args, dev, prune_n=0,
                               prune_m=0):
    """
    处理单个层的选择性补偿逻辑.
    仅补偿那些因结构化约束被移除、但根据Wanda分数评估为高重要性的权重子集.
    """
    print(f"Processing layer {layer_index} sublayer {layer_name} with SELECTIVE COMPENSATION")

    # --- 通用步骤: 计算 Wanda score (W_metric) ---
    W_metric = torch.abs(original_weight) * torch.sqrt(wrapped_layer.scaler_row.reshape((1, -1)))

    # --- 步骤 1: 生成非结构化剪枝掩码 (Unstructured Pruning Mask) ---
    # W_mask_unstructured 中为 True 的位置代表要被剪枝的权重
    sort_res = torch.sort(W_metric, dim=-1, stable=True)
    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
    W_mask_unstructured = torch.zeros_like(W_metric, dtype=torch.bool)
    #指定的 indices 位置设为 True,代表要剪枝的权重
    W_mask_unstructured.scatter_(1, indices, True)
    # 取反得到Keep_mask_unstructured 中为 True 的位置代表在非结构化剪枝中要保留的权重
    Keep_mask_unstructured = ~W_mask_unstructured
    print(f"  Step 1: Unstructured mask generated.")


    # --- 步骤 2: 生成结构化剪枝矩阵 (B_structured) 及对应掩码 ---
    # W_mask_structured 中为 True 的位置代表要被剪枝的权重
    W_mask_structured = torch.zeros_like(W_metric, dtype=torch.bool)
    if prune_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii:(ii + prune_m)].float()
                W_mask_structured.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
    else:
        # 如果没有指定n:m，则认为结构化和非结构化相同
        W_mask_structured = W_mask_unstructured.clone()

    B_structured = original_weight.clone()
    B_structured[W_mask_structured] = 0
    print(f"  Step 2: B_structured created (sparsity: {(B_structured == 0).float().mean():.4f}).")


    # --- 步骤 3: 识别补偿候选权重 ---
    # 候选权重是指：在非结构化标准下应保留(Keep_mask_unstructured=True)，但为满足结构化约束而被剪除(W_mask_structured=True)的权重。
    compensation_candidate_mask = Keep_mask_unstructured & W_mask_structured
    num_candidates = torch.sum(compensation_candidate_mask).item()
    print(f"  Step 3: Identified {num_candidates} potential compensation candidates.")

    if num_candidates == 0:
        print("    No weights qualify for compensation. Skipping.")
        return B_structured, None

    # --- 步骤 4: 根据Wanda分数对候选权重进行排序 ---
    candidate_wanda_scores = W_metric[compensation_candidate_mask]
    
    # 按降序排列，以优先补偿最重要的权重
    sorted_scores, sorted_indices = torch.sort(candidate_wanda_scores, descending=True)
    print(f"  Step 4: Ranked compensation candidates by Wanda score.")

    # --- 步骤 5: 构建高稀疏度的补偿矩阵 (ΔB) ---
    # 仅选择 Wanda 分数排名最高的 Top-K% 候选权重进行补偿
    compensation_ratio = args.compensation_ratio
    num_to_compensate = 0  # 默认不补偿

    # 只有在补偿率大于0时才计算补偿数量
    if compensation_ratio > 0:
        num_to_compensate = int(num_candidates * compensation_ratio)

        # 如果计算出的数量为0，但确实存在候选者，则至少补偿一个
        if num_to_compensate == 0 and num_candidates > 0:
            num_to_compensate = 1

    print(f"  Step 5: Building sparse delta_B by compensating top {num_to_compensate} ({compensation_ratio:.2%}) candidates.")

    # 获取分数最高的候选权重在一维列表中的索引
    top_candidate_indices = sorted_indices[:num_to_compensate]

    # 创建一个空的 ΔB 矩阵
    delta_B = torch.zeros_like(original_weight)

    # 我们需要候选权重在原始矩阵中的二维坐标来填充 ΔB
    # `torch.where` 返回 (row_indices, col_indices) 元组
    candidate_original_rows, candidate_original_cols = torch.where(compensation_candidate_mask)
    
    # 从所有候选者的坐标中，选出分数最高的那些
    rows_to_compensate = candidate_original_rows[top_candidate_indices]
    cols_to_compensate = candidate_original_cols[top_candidate_indices]

    # 在 ΔB 中恢复这些被选中的权重值
    delta_B[rows_to_compensate, cols_to_compensate] = original_weight[rows_to_compensate, cols_to_compensate]
    
    # --- 步骤 6: 封装补偿参数 ---
    compensation_params = None
    if torch.norm(delta_B) > 1e-8:
        print(f"  Step 6: Converting sparse delta_B to CSR format...")
        delta_B_csr = delta_B.to_sparse_csr()
        compensation_params = {
            'delta_B_csr': delta_B_csr,
        }
        print(f"    Compensation parameters created. Non-zero elements: {delta_B_csr.values().numel()}")
    else:
        print(f"    No compensation needed after selection (delta_B norm is too small).")

    return B_structured, compensation_params




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
                    with torch.no_grad():
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]


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
    执行带有补偿逻辑的Wanda剪枝，逐层替换模型层，直接返回最终模型
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

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

        # 设置设备
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

        # 对每个线性层进行剪枝并立即替换
        for name in subset:
            original_weight = subset[name].weight.data.clone()

            # 使用模块化的处理函数
            B_structured, layer_compensation_params = process_layer_compensation(
                i, name, original_weight, wrapped_layers[name], args, dev, prune_n, prune_m
            )

            # 立即处理：要么替换为补偿层，要么设置稀疏权重
            if layer_compensation_params is not None:
                print(f"Replacing layer: model.layers.{i}.{name}")
                params = layer_compensation_params
                new_layer = CompensatedSparseLinear(subset[name], params)
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(f"model.layers.{i}.{parent_name}")
                setattr(parent_module, child_name, new_layer)

                # 关键的内存泄漏修复
                del params
                del layer_compensation_params
                print(f"    Compensated layer created.")
            else:
                subset[name].weight.data = B_structured
                print(f"    Layer weight set to sparse structured matrix.")

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

        # 简化的内存清理
        del wrapped_layers
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("\n=== Wanda compensation completed ===")
    return model



def prune_wanda_with_2_4_compensation(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    执行带有补偿逻辑的Wanda剪枝2:4，逐层替换模型层，直接返回最终模型
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

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

        # 设置设备
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

        # 对每个线性层进行剪枝并立即替换
        for name in subset:
            original_weight = subset[name].weight.data.clone()

            # --- 核心区别：调用新的 process_layer... 函数 ---
            B_structured, layer_compensation_params = process_layer_compensation_dense(
                i, name, original_weight, wrapped_layers[name], args, dev, prune_n, prune_m
            )
            subset[name].weight.data = B_structured
            print(f"    Layer weight is set to the sparse structured matrix.")
            # 立即处理：要么替换为补偿层，要么设置稀疏权重
            if layer_compensation_params is not None:
                print(f"Replacing layer: model.layers.{i}.{name}")
                params = layer_compensation_params
                # --- 核心区别：使用新的 CompensatedSparseLinear24 类 ---
                new_layer = CompensatedSparseLinear24(subset[name], params)
                #new_layer = CompensatedSparseLinear(subset[name], params)
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(f"model.layers.{i}.{parent_name}")
                setattr(parent_module, child_name, new_layer)

                # 关键的内存泄漏修复
                del params
                del layer_compensation_params
                print(f"    Compensated layer created.")

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

        # 简化的内存清理
        del wrapped_layers
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("\n=== Wanda 2:4 compensation and layer replacement completed ===")
    return model

def prune_wanda_with_dense_compensation(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    执行带有补偿逻辑的Wanda剪枝，逐层替换模型层，直接返回最终模型
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

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

        # 设置设备
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

        # 对每个线性层进行剪枝并立即替换
        for name in subset:
            original_weight = subset[name].weight.data.clone()

            # 使用模块化的处理函数
            B_structured, layer_compensation_params = process_layer_compensation_dense(
                i, name, original_weight, wrapped_layers[name], args, dev, prune_n, prune_m
            )

            # --- 核心区别：计算最终稠密权重并直接赋值，不使用自定义层 ---
            # 1. 检查返回的字典是否为 None
            if layer_compensation_params is not None:
                # 2. 从字典中提取出 delta_B
                delta_B = layer_compensation_params['delta_B_dense']
                final_weight = B_structured + delta_B
                print(f"Applying dense compensation to layer: model.layers.{i}.{name}")
                subset[name].weight.data = final_weight
            else:
                subset[name].weight.data = B_structured
                print(f"    Layer weight is set to the sparse structured matrix.")

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

        # 简化的内存清理
        del wrapped_layers
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("\n=== Wanda compensation completed ===")
    return model

def prune_wanda_with_selective_compensation(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    执行带有选择性补偿逻辑的Wanda剪枝，逐层替换模型层，直接返回最终模型.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        returned_data = prepare_calibration_input(model, dataloader, args.nsamples, model.seqlen)
        inps, outs, attention_mask, position_ids = returned_data[:4]

    layers = model.model.layers

    for i in range(len(layers)):
        print(f"\n=== Processing Layer {i} with Selective Compensation ===")
        layer = layers[i]
        subset = find_layers(layer)

        # 设置设备
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

        # 对每个线性层进行剪枝并立即替换
        for name in subset:
            original_weight = subset[name].weight.data.clone()

            # --- 核心区别: 调用新的 process_layer_selective_compensation 函数 ---
            B_structured, layer_compensation_params = process_layer_selective_compensation(
                i, name, original_weight, wrapped_layers[name], args, dev, prune_n, prune_m
            )

            # 立即处理：要么替换为补偿层，要么设置稀疏权重
            if layer_compensation_params is not None:
                print(f"Replacing layer: model.layers.{i}.{name}")
                params = layer_compensation_params
                new_layer = CompensatedSparseLinear(subset[name], params)
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(f"model.layers.{i}.{parent_name}")
                setattr(parent_module, child_name, new_layer)

                # 关键的内存泄漏修复
                del params
                del layer_compensation_params
                print(f"    Selectively compensated layer created.")
            else:
                subset[name].weight.data = B_structured
                print(f"    Layer weight set to sparse structured matrix (no compensation).")

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

        # 简化的内存清理
        del wrapped_layers
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("\n=== Wanda selective compensation completed ===")
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