import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 


from lib.layerwrapper import WrappedGPT


# --- 配置参数 ---
# 模拟 Llama2-7B MLP 层 (如 gate_proj/up_proj) 的维度
# Llama2-7B hidden_size = 4096, intermediate_size = 11008
INPUT_DIM = 4096    # 线性层输入特征维度 (模拟 hidden_size)
OUTPUT_DIM = 11008   # 线性层输出特征维度 (模拟 intermediate_size)

NUM_SAMPLES = 2000 # 用于生成激活数据的样本数量
# TRAIN_EPOCHS = 1000 # 不再需要训练，注释掉



SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 将计算移动到 GPU (如果可用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 辅助函数：计算稀疏度 ---
def calculate_sparsity(weight_matrix):
    """计算矩阵的稀疏度 (零元素的比例)"""
    total_elements = weight_matrix.numel()
    zero_elements = (weight_matrix == 0).sum().item()
    sparsity = zero_elements / total_elements
    return sparsity

# --- 2:4 结构化剪枝函数 ---
def prune_to_2_4(matrix, scaler_row):
    """
    对输入矩阵按行做2:4结构化剪枝：每4个元素保留2个最大幅度，其余置零。
    修改为Wanda原版处理方式：忽略末尾不足4列的部分。
    """
    # 计算权重重要性，使用Wanda原版的度量方法
    # Wanda 的度量标准: abs(W) * sqrt(scaler_row)
    W_metric = torch.abs(matrix) * torch.sqrt(scaler_row.reshape((1, -1)))
    
    # 创建剪枝掩码
    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
    
    # 2:4结构化剪枝：每4列中保留2个最重要的，剪掉2个最不重要的
    prune_n, prune_m = 2, 4  # 每4个元素中剪掉2个（保留2个）
    
    # Wanda原版逻辑：只处理完整的4列块
    for ii in range(0, W_metric.shape[1] - W_metric.shape[1] % prune_m, prune_m):
        # 完整的4列块，正常2:4剪枝
        tmp = W_metric[:, ii:ii+prune_m].float()
        # topk返回最小的prune_n个元素的索引，largest=False表示取最小的
        _, indices = torch.topk(tmp, prune_n, dim=1, largest=False)
        # 在掩码中标记要剪枝的位置
        W_mask.scatter_(1, ii + indices, True)

    # 应用掩码进行剪枝
    result = matrix.clone()
    result[W_mask] = 0
    
    return result


# --- Wanda非结构化剪枝函数 ---
def wanda_unstructured_prune(matrix, scaler_row, sparsity_ratio=0.5):
    """
    使用Wanda重要性度量进行非结构化剪枝
    Args:
        matrix: 输入权重矩阵
        scaler_row: 激活统计信息
        sparsity_ratio: 稀疏率 (0.5表示50%稀疏)
    """
    print(f"  应用Wanda非结构化剪枝，稀疏率: {sparsity_ratio:.1%}")
    
    # 计算Wanda重要性度量
    W_metric = torch.abs(matrix) * torch.sqrt(scaler_row.reshape((1, -1)))
    
    # 计算需要剪枝的元素数量
    total_elements = matrix.numel()
    num_prune = int(total_elements * sparsity_ratio)
    
    # 找到重要性最小的元素进行剪枝
    threshold = torch.kthvalue(W_metric.flatten(), num_prune).values
    mask = W_metric <= threshold
    
    # 应用剪枝掩码
    result = matrix.clone()
    result[mask] = 0
    
    # 验证稀疏率
    actual_sparsity = (result == 0).float().mean().item()
    print(f"  实际稀疏率: {actual_sparsity:.1%}")
    
    return result



# --- 2. 应用Wanda非结构化剪枝得到W_unstructured ---
def apply_wanda_unstructured_pruning(W_dense, scaler_row, sparsity_ratio=0.5):
    print(f"\n--- 2. 应用Wanda非结构化剪枝 (稀疏率: {sparsity_ratio:.1%}) ---")
    
    W_unstructured = wanda_unstructured_prune(W_dense, scaler_row, sparsity_ratio)
    
    # 验证稀疏性
    actual_sparsity = calculate_sparsity(W_unstructured)
    print(f"  W_unstructured 形状: {W_unstructured.shape}, 稀疏率: {actual_sparsity:.1%}")
    print(f"  W_unstructured norm: {torch.norm(W_unstructured):.4f}")
    
    return W_unstructured

# --- 2:4 基线剪枝方法 ---
def prune_2_4_baseline(weight_matrix, scaler_row):
    """
    直接对权重矩阵进行2:4结构化剪枝作为基线
    """
    print(f"\n--- 2:4 基线剪枝 ---")
    print(f"  正在处理 {weight_matrix.shape} 的权重矩阵...")
    print(f"  使用Wanda重要性度量 (权重 * sqrt(scaler_row))")
    result = prune_to_2_4(weight_matrix, scaler_row)
    print(f"  2:4 基线剪枝完成")
    return result




def check_2_4_sparsity(matrix):
    """
    检查矩阵是否符合2:4稀疏模式
    基于Wanda原始逻辑实现
    """
    rows, cols = matrix.shape
    
    # 检查每4列为一组，每组中是否最多有2个非零元素
    prune_m = 4
    tolerance = 1e-10
    
    for ii in range(0, cols, prune_m):
        # 取出当前的4列块（如果不足4列则取剩余的）
        end_col = min(ii + prune_m, cols)
        if end_col - ii < prune_m:
            # 如果最后一组不足4列，我们可以放宽检查
            continue
            
        block = matrix[:, ii:end_col]  # [rows, 4]
        
        # 计算每行中非零元素的数量
        non_zero_count = (torch.abs(block) > tolerance).sum(dim=1)
        
        # 检查每行是否最多有2个非零元素
        if torch.any(non_zero_count > 2):
            return False
    
    return True



# --- 辅助函数：计算 ΔB 的秩和核范数 ---
def calculate_delta_metrics(W_unstructured, W_main_pruned, svd_tolerance=1e-10):
    """
    计算 ΔB = W_unstructured - W_main_pruned 的核范数和有效秩。
    有效秩通过过滤小于svd_tolerance的奇异值来计算。
    """
    delta_B = W_unstructured - W_main_pruned
    
    nuclear_norm_delta_B = torch.linalg.matrix_norm(delta_B, ord='nuc').item()

    rank_delta_B = 0
    try:
        if torch.norm(delta_B) > 1e-12: # 避免对几乎为零的矩阵进行SVD
            U, S_vals, Vh = torch.linalg.svd(delta_B, full_matrices=False)
            effective_singular_values = S_vals[S_vals > svd_tolerance]
            rank_delta_B = len(effective_singular_values)
        else:
            rank_delta_B = 0
    except RuntimeError:
        rank_delta_B = 0 # SVD计算失败时秩为0
    
    return rank_delta_B, nuclear_norm_delta_B



# --- 功能损失计算函数 ---
def functional_loss(W1, W2, X):
    """
    计算L2 Error损失: ||W1 @ X.T - W2 @ X.T||_F
    """
    return torch.norm((W1 - W2) @ X.T, p='fro').item()


# 低秩近似函数
def low_rank_approximation(matrix, k):
    """
    对输入矩阵做低秩近似（截断SVD），只保留前k个奇异值分量
    Args:
        matrix: 输入矩阵 (torch.Tensor)
        k: 保留的秩
    Returns:
        低秩近似矩阵 (torch.Tensor)
    """
    original_device = matrix.device
    try:
        # 尝试在原设备上进行SVD
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"    GPU内存不足，将SVD计算移到CPU: {e}")
            # 移到CPU进行计算
            matrix_cpu = matrix.cpu()
            U, S, Vh = torch.linalg.svd(matrix_cpu, full_matrices=False)
            # 将结果移回原设备
            U = U.to(original_device)
            S = S.to(original_device)
            Vh = Vh.to(original_device)
        else:
            raise e
    
    S_trunc = torch.zeros_like(S)
    S_trunc[:k] = S[:k]
    approx = (U * S_trunc.unsqueeze(0)) @ Vh
    return approx


# --- 1. 生成随机稠密矩阵和模拟激活统计 ---
def generate_dense_matrix_and_activations(input_dim, output_dim, num_samples):
    print("\n--- 1. 生成随机稠密权重矩阵和模拟激活统计 ---")
    
    # 生成随机稠密权重矩阵 (out_features, in_features)
    #W_dense = torch.randn(output_dim, input_dim, dtype=torch.float32, device=device)
    # 使用Xavier初始化
    # 使用kaiming初始化
    #torch.nn.init.xavier_uniform_(W_dense)
    temp = torch.nn.Linear(input_dim, output_dim).to(device)
    W_dense = temp.weight.data
    print(f"  生成稠密权重矩阵: {W_dense.shape}, (norm: {torch.norm(W_dense):.4f})")
    
    # 生成模拟的激活数据来计算scaler_row
    print(f"  生成 {num_samples} 个样本的模拟激活数据...")
    X_activation = torch.randn(num_samples, input_dim, dtype=torch.float32, device=device)
    
    # 创建临时线性层来使用WrappedGPT计算激活统计
    temp_model = nn.Linear(input_dim, output_dim, bias=False).to(device)
    temp_model.weight.data = W_dense.clone()
    
    print("  使用 WrappedGPT 计算 scaler_row (激活统计)...")
    wrapped_layer = WrappedGPT(temp_model)
    
    # 计算激活统计，用于算wanda重要性指标
    with torch.no_grad():
        batch_size = 32
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_input = X_activation[i:batch_end]
            batch_output = temp_model(batch_input)
            wrapped_layer.add_batch(batch_input, batch_output)
    
    scaler_row = wrapped_layer.scaler_row.detach().clone()
    print(f"  计算的 scaler_row 形状: {scaler_row.shape}, (norm: {torch.norm(scaler_row):.4f})")
    
    return W_dense, scaler_row, X_activation






# --- 基于核范数 subgradient 的单步优化 2:4 剪枝 ---
def prune_2_4_with_nuclear_norm_importance(W_unstructured, W_dense):
    """
    基于核范数subgradient的单步优化2:4剪枝。
    该方法旨在找到一个2:4稀疏矩阵 W_main, 使得 ||W_unstructured - W_main||_* 最小化。
    W_main 是通过对 W_dense 进行2:4剪枝得到的。

    步骤:
    1. 计算变化矩阵 ΔB = W_unstructured - W_dense。
    2. 对 ΔB 进行SVD, 得到核范数的subgradient G = U @ Vh。
    3. 计算重要性分数: importance = |ΔB * G|。
    4. 使用此 importance 分数对 W_dense 进行2:4剪枝，得到最终的 W_main。

    Args:
        W_unstructured (torch.Tensor): 50%非结构化稀疏的目标矩阵。
        W_dense (torch.Tensor): 原始的稠密矩阵。

    Returns:
        torch.Tensor: 经过核范数优化剪枝后的2:4稀疏矩阵。
    """
    print(f"\n--- 基于核范数 subgradient 的单步优化 2:4 剪枝 ---")
    
    # 1. 计算变化矩阵 ΔB = W_unstructured - W_dense
    delta_B = W_unstructured - W_dense
    print(f"  计算 ΔB (W_unstructured - W_dense), norm: {torch.norm(delta_B):.4f}")

    try:
        # 2. 对 ΔB 进行SVD, 得到 subgradient G
        # 使用 float32 以提高 SVD 的稳定性
        U, S, Vh = torch.linalg.svd(delta_B.to(torch.float32), full_matrices=False)
        
        # 核范数的 subgradient G = U @ Vh
        G = (U @ Vh).to(W_dense.dtype)
        print(f"  SVD 成功, 计算 subgradient G")

        # 3. 计算重要性分数: importance
        importance = torch.abs(delta_B * G)
        #importance = delta_B * G
        #importance = -1 * importance
        print(f"  计算 importance = |delta_B * G|, norm: {torch.norm(importance):.4f}")

        # 打印一些统计信息
        current_nuclear_norm = torch.sum(S).item()
        current_rank = (S > 1e-10).sum().item()
        print(f"  ΔB 的初始核范数 = {current_nuclear_norm:.6f}, 有效秩 = {current_rank}")

    except RuntimeError as e:
        print(f"  警告: SVD 失败 ({e})，回退到二阶近似 importance = |ΔB * ΔB|")
        importance = torch.abs(delta_B * delta_B)

    # 4. 使用新的 importance 分数对 W_dense 进行 2:4 结构化剪枝
    print(f"  使用 importance 对 W_dense 进行 2:4 剪枝...")
    W_main_optimized = prune_to_2_4_by_importance(W_dense, importance)

    return W_main_optimized


def prune_to_2_4_by_importance(matrix, importance):
    """
    基于 importance 矩阵进行 2:4 结构化剪枝
    每 4 个元素的小组里，保留 importance 最大的 2 个，其他置零
    修改为Wanda原版处理方式：忽略末尾不足4列的部分。
    """
    result = matrix.clone()
    rows, cols = matrix.shape

    prune_mask = torch.zeros_like(matrix, dtype=torch.bool)

    prune_n, prune_m = 2, 4  # 每4个元素中保留2个

    # 原版Wanda逻辑：只处理完整的4列块
    for ii in range(0, cols - cols % prune_m, prune_m):
        importance_block = importance[:, ii:ii+prune_m]
        # 完整 4 列块，剪掉 importance 最小的 2 个
        _, to_prune_indices = torch.topk(importance_block, prune_n, dim=1, largest=True)
        prune_mask.scatter_(1, ii + to_prune_indices, True)

    # 置零最不重要的元素
    result[prune_mask] = 0

    return result


# 绘制奇异值分布图
def plot_singular_values_of_delta_B(W_unstructured, W_main_pruned, title, save_dir="plots", svd_tolerance=1e-10):
    """绘制 ΔB = W_unstructured - W_main_pruned 的奇异值分布图"""
    os.makedirs(save_dir, exist_ok=True)
    
    delta_B = W_unstructured - W_main_pruned
    
    if torch.norm(delta_B) < 1e-12: # 避免对几乎为零的矩阵进行SVD
        print(f"  Warning: Delta B matrix ({title}) is close to zero, skipping SVD analysis")
        return
    
    with torch.no_grad():
        try:
            # 尝试在GPU上进行SVD
            U, S_vals, Vh = torch.linalg.svd(delta_B, full_matrices=False)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  GPU内存不足，将SVD计算移到CPU进行: {e}")
                try:
                    # 移到CPU进行计算
                    delta_B_cpu = delta_B.cpu()
                    U, S_vals, Vh = torch.linalg.svd(delta_B_cpu, full_matrices=False)
                    S_vals = S_vals.cpu()  # 确保在CPU上
                except Exception as cpu_e:
                    print(f"  Error: Delta B matrix ({title}) SVD calculation failed even on CPU - {cpu_e}")
                    return
            else:
                print(f"  Error: Delta B matrix ({title}) SVD calculation failed - {e}")
                return
        
        # 过滤掉数值上接近零的奇异值
        S_vals_filtered = S_vals[S_vals > svd_tolerance]
        
        if len(S_vals_filtered) == 0:
            print(f"  Warning: All singular values of Delta B matrix ({title}) are close to zero (below {svd_tolerance}), skipping plot")
            return
        
        plt.figure(figsize=(10, 5))
        plt.plot(S_vals_filtered.cpu().numpy(), 'o-', markersize=4)
        plt.title(f'{title} - Delta B Singular Values Distribution', fontsize=14)
        plt.xlabel('Singular Value Index', fontsize=12)
        plt.ylabel('Singular Value (Log Scale)', fontsize=12)
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 添加统计信息
        plt.text(0.7, 0.8, f'Effective Singular Values: {len(S_vals_filtered)}\n'
                          f'Max Singular Value: {S_vals_filtered[0]:.2e}\n'
                          f'Nuclear Norm: {torch.sum(S_vals_filtered):.2e}', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        filename = f"{title.replace(' ', '_').replace('-', '_').replace(':', '')}_singular_values.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {filepath}")


# 绘制累积能量分布图
def plot_cumulative_energy_of_delta_B(W_unstructured, W_main_pruned, title, save_dir="plots", svd_tolerance=1e-10):
    """
    绘制 ΔB = W_unstructured - W_main_pruned 的累积能量分布图
    X轴：保留的奇异值数量（N）
    Y轴：保留的前N个奇异值所累积的能量占总能量的百分比
    """
    os.makedirs(save_dir, exist_ok=True)
    
    delta_B = W_unstructured - W_main_pruned
    
    if torch.norm(delta_B) < 1e-12: # 避免对几乎为零的矩阵进行SVD
        print(f"  Warning: Delta B matrix ({title}) is close to zero, skipping cumulative energy analysis")
        return
    
    with torch.no_grad():
        try:
            # 尝试在GPU上进行SVD
            U, S_vals, Vh = torch.linalg.svd(delta_B, full_matrices=False)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  GPU内存不足，将SVD计算移到CPU进行: {e}")
                try:
                    # 移到CPU进行计算
                    delta_B_cpu = delta_B.cpu()
                    U, S_vals, Vh = torch.linalg.svd(delta_B_cpu, full_matrices=False)
                    S_vals = S_vals.cpu()  # 确保在CPU上
                except Exception as cpu_e:
                    print(f"  Error: Delta B matrix ({title}) SVD calculation failed even on CPU - {cpu_e}")
                    return
            else:
                print(f"  Error: Delta B matrix ({title}) SVD calculation failed - {e}")
                return
        
        # 过滤掉数值上接近零的奇异值
        S_vals_filtered = S_vals[S_vals > svd_tolerance]
        
        if len(S_vals_filtered) == 0:
            print(f"  Warning: All singular values of Delta B matrix ({title}) are close to zero (below {svd_tolerance}), skipping cumulative energy plot")
            return
        
        # 计算累积能量
        S_vals_np = S_vals_filtered.cpu().numpy()
        total_energy = np.sum(S_vals_np)  # 总能量（核范数）
        cumulative_energy = np.cumsum(S_vals_np)  # 累积能量
        cumulative_energy_ratio = cumulative_energy / total_energy * 100  # 累积能量百分比
        
        # 绘制累积能量图
        plt.figure(figsize=(12, 6))
        
        # 主图：累积能量百分比
        plt.plot(range(1, len(cumulative_energy_ratio) + 1), cumulative_energy_ratio, 'b-', linewidth=2, label='Cumulative Energy %')
        plt.fill_between(range(1, len(cumulative_energy_ratio) + 1), 0, cumulative_energy_ratio, alpha=0.3, color='blue')
        
        # 添加一些重要的水平线
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% Energy')
        plt.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% Energy')
        plt.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90% Energy')
        plt.axhline(y=95, color='purple', linestyle='--', alpha=0.7, label='95% Energy')
        
        # 找到达到各个能量阈值所需的奇异值数量
        thresholds = [50, 80, 90, 95]
        threshold_positions = []
        for thresh in thresholds:
            idx = np.argmax(cumulative_energy_ratio >= thresh)
            if cumulative_energy_ratio[idx] >= thresh:
                threshold_positions.append((idx + 1, thresh))
                plt.annotate(f'{thresh}% at N={idx + 1}', 
                           xy=(idx + 1, thresh), xytext=(idx + 1 + len(S_vals_np)*0.1, thresh + 5),
                           arrowprops=dict(arrowstyle='->', color='black', alpha=0.6),
                           fontsize=10, ha='left')
        
        plt.title(f'{title} - Cumulative Energy Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Singular Values Retained (N)', fontsize=12)
        plt.ylabel('Cumulative Energy Percentage (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 105)
        
        # 添加统计信息
        info_text = f'Total Singular Values: {len(S_vals_filtered)}\n'
        info_text += f'Total Energy (Nuclear Norm): {total_energy:.2e}\n'
        info_text += f'Max Singular Value: {S_vals_np[0]:.2e}\n'
        
        # 添加能量分布关键信息
        for pos, thresh in threshold_positions:
            info_text += f'{thresh}% Energy: Top {pos} SVs\n'
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        # 保存图像
        filename = f"{title.replace(' ', '_').replace('-', '_').replace(':', '')}_cumulative_energy.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Cumulative energy plot saved: {filepath}")
        
        # 打印一些关键统计信息
        print(f"    Total energy (nuclear norm): {total_energy:.6f}")
        for pos, thresh in threshold_positions:
            energy_ratio = pos / len(S_vals_filtered) * 100
            print(f"    {thresh}% energy achieved with top {pos} singular values ({energy_ratio:.1f}% of all SVs)")
        
        return filepath
        

# --- 主执行逻辑 ---
if __name__ == '__main__':
    print("="*60)
    print("核范数最小化 + 2:4结构化稀疏约束优化实验")
    print("="*60)
    
    # 1. 生成随机稠密权重矩阵和激活统计
    W_dense, scaler_row, X_activation = generate_dense_matrix_and_activations(
        INPUT_DIM, OUTPUT_DIM, NUM_SAMPLES
    )
    
    # 记录原始列数，用于后续评估
    original_cols = W_dense.shape[1]
    print(f"  使用原始权重矩阵，列数: {original_cols} (可能不是4的倍数)")
    
    # 2. 应用Wanda 50%非结构化剪枝得到W_unstructured
    W_unstructured = apply_wanda_unstructured_pruning(W_dense, scaler_row, sparsity_ratio=0.5)
    
    # 3. 由W_dense直接2:4结构化剪枝得到W_structured (Wanda baseline)
    print("\n" + "="*50)
    print("步骤3: 由稠密矩阵直接2:4结构化剪枝 (Wanda baseline)")
    print("="*50)
    W_structured = prune_2_4_baseline(W_dense, scaler_row)
    
    # 4. 用核范数importance-based方法优化2:4剪枝
    print("\n" + "="*50)
    print("步骤4: 核范数importance-based 2:4剪枝优化")
    print("="*50)
    print("  该方法旨在找到一个2:4稀疏矩阵，使之与W_unstructured的差值核范数最小化")
    W_nuclear_optimized = prune_2_4_with_nuclear_norm_importance(W_unstructured, W_dense)
    
    # 验证所有矩阵的2:4稀疏性
    print("\n" + "="*50)
    print("稀疏性验证")
    print("="*50)
    matrices = {
        'W_structured (Wanda baseline)': W_structured,
        'W_nuclear_optimized (核范数优化)': W_nuclear_optimized
    }
    
    for name, matrix in matrices.items():
        is_2_4 = check_2_4_sparsity(matrix)
        sparsity = calculate_sparsity(matrix)
        print(f"  {name}: 2:4模式={is_2_4}, 稀疏率={sparsity:.1%}")
    
    # 5. 综合评估和比较
    print("\n" + "="*60)
    print("步骤5: 综合评估和比较结果")
    print("="*60)
    
    # 5a. 核范数和秩的比较 - 以W_unstructured为参考
    print("\n--- A. 两种2:4剪枝方法的ΔB分析 (相对于W_unstructured) ---")
    
    # 计算各矩阵相对于W_unstructured的ΔB指标
    rank_structured, nuc_norm_structured = calculate_delta_metrics(W_unstructured, W_structured)
    rank_nuclear_optimized, nuc_norm_nuclear_optimized = calculate_delta_metrics(W_unstructured, W_nuclear_optimized)
    
    print(f"  W_structured (Wanda baseline) ΔB: 有效秩={rank_structured:4d}, Nuclear Norm={nuc_norm_structured:10.6f}")
    print(f"  W_nuclear_optimized (核范数优化) ΔB: 有效秩={rank_nuclear_optimized:4d}, Nuclear Norm={nuc_norm_nuclear_optimized:10.6f}")
    
    # 比较改进效果
    print(f"\n  核范数优化相对于Wanda baseline的改进效果:")
    rank_improvement = rank_structured - rank_nuclear_optimized
    nuc_norm_improvement = nuc_norm_structured - nuc_norm_nuclear_optimized
    print(f"    秩降低: {rank_improvement} ({rank_improvement/max(rank_structured,1)*100:.1f}%)")
    print(f"    核范数降低: {nuc_norm_improvement:.6f} ({nuc_norm_improvement/max(nuc_norm_structured,1e-8)*100:.1f}%)")
    
    # 5b. 功能损失比较
    print("\n--- B. 功能损失比较 (相对于W_unstructured) ---")
    print("  (计算 ||W_unstructured @ X - W_method @ X||_F)")
    
    # 使用激活数据进行评估
    X_eval = X_activation
    W_unstructured_eval = W_unstructured
    W_structured_eval = W_structured
    W_nuclear_optimized_eval = W_nuclear_optimized
    
    error_structured = functional_loss(W_unstructured_eval, W_structured_eval, X_eval)
    error_nuclear_optimized = functional_loss(W_unstructured_eval, W_nuclear_optimized_eval, X_eval)
    
    print(f"  W_structured (Wanda baseline)功能损失: {error_structured:.6f}")
    print(f"  W_nuclear_optimized (核范数优化)功能损失: {error_nuclear_optimized:.6f}")
    
    print(f"\n  功能损失改进:")
    error_improvement = error_structured - error_nuclear_optimized
    print(f"    核范数优化 vs Wanda baseline: {error_improvement:.6f} ({'改善' if error_improvement > 0 else '略有增加' if error_improvement < 0 else '保持不变'})")




    # === 新增：低秩补偿功能损失评估 ===
    print("\n--- C. 低秩补偿功能损失评估 ---")
    # 调整修改低秩近似的秩k
    k = 64
    print(f"  低秩近似保留前 {k} 个奇异值")

    # 计算ΔB
    deltaB_structured = W_unstructured - W_structured
    deltaB_nuclear_optimized = W_unstructured - W_nuclear_optimized

    print(f"  正在计算 ΔB_structured 的低秩近似...")
    # 低秩近似
    deltaB_structured_LRA = low_rank_approximation(deltaB_structured, k)
    print(f"  正在计算 ΔB_nuclear_optimized 的低秩近似...")
    deltaB_nuclear_optimized_LRA = low_rank_approximation(deltaB_nuclear_optimized, k)

    # 构造补偿后权重
    W_compensated_structured = W_structured + deltaB_structured_LRA
    W_compensated_nuclear_optimized = W_nuclear_optimized + deltaB_nuclear_optimized_LRA

    # 计算补偿后功能损失
    error_compensated_structured = functional_loss(W_unstructured, W_compensated_structured, X_eval)
    error_compensated_nuclear_optimized = functional_loss(W_unstructured, W_compensated_nuclear_optimized, X_eval)

    print(f"  W_structured + ΔB_structured_LRA 功能损失: {error_compensated_structured:.6f}")
    print(f"  W_nuclear_optimized + ΔB_nuclear_optimized_LRA 功能损失: {error_compensated_nuclear_optimized:.6f}")
    
    # 比较低秩补偿后的改进效果
    print(f"\n  低秩补偿后的改进效果:")
    compensated_improvement = error_compensated_structured - error_compensated_nuclear_optimized
    print(f"    核范数优化 vs Wanda baseline (补偿后): {compensated_improvement:.6f} ({'改善' if compensated_improvement > 0 else '略有增加' if compensated_improvement < 0 else '保持不变'})")

    # 5c. 稀疏度总结
    print("\n--- D. 稀疏度总结 ---")
    sparsity_dense = calculate_sparsity(W_dense)
    sparsity_unstructured = calculate_sparsity(W_unstructured)
    sparsity_structured = calculate_sparsity(W_structured)
    sparsity_nuclear_optimized = calculate_sparsity(W_nuclear_optimized)
    
    print(f"  W_dense (原始稠密): {sparsity_dense:.2%}")
    print(f"  W_unstructured (Wanda 50%非结构化): {sparsity_unstructured:.2%}")
    print(f"  W_structured (Wanda baseline 2:4): {sparsity_structured:.2%}")
    print(f"  W_nuclear_optimized (核范数优化2:4): {sparsity_nuclear_optimized:.2%}")
    print(f"  2:4理论稀疏度: 50.00%")
    
    # 6. 可视化结果
    print("\n--- E. Delta B Analysis Plots ---")
    print("  生成奇异值分布图...")
    plot_singular_values_of_delta_B(W_unstructured, W_structured, "W_structured_vs_W_unstructured")
    plot_singular_values_of_delta_B(W_unstructured, W_nuclear_optimized, "W_nuclear_optimized_vs_W_unstructured")
    
    print("  生成累积能量分布图...")
    plot_cumulative_energy_of_delta_B(W_unstructured, W_structured, "W_structured_vs_W_unstructured")
    plot_cumulative_energy_of_delta_B(W_unstructured, W_nuclear_optimized, "W_nuclear_optimized_vs_W_unstructured")
    
    # 7. 结论
    print("\n" + "="*60)
    print("步骤6: 实验结论")
    print("="*60)
    
    if rank_nuclear_optimized < rank_structured and nuc_norm_nuclear_optimized < nuc_norm_structured:
        print("✅ 成功! 核范数优化方法在降低ΔB的秩和核范数方面都优于Wanda baseline")
        print(f"   秩从 {rank_structured} 降低到 {rank_nuclear_optimized}")
        print(f"   核范数从 {nuc_norm_structured:.6f} 降低到 {nuc_norm_nuclear_optimized:.6f}")
    elif rank_nuclear_optimized < rank_structured:
        print("✅ 部分成功! 核范数优化方法成功降低了ΔB的秩")
        print(f"   秩从 {rank_structured} 降低到 {rank_nuclear_optimized}")
        print(f"   但核范数变化: {nuc_norm_structured:.6f} -> {nuc_norm_nuclear_optimized:.6f}")
    elif nuc_norm_nuclear_optimized < nuc_norm_structured:
        print("✅ 部分成功! 核范数优化方法成功降低了ΔB的核范数")
        print(f"   核范数从 {nuc_norm_structured:.6f} 降低到 {nuc_norm_nuclear_optimized:.6f}")
        print(f"   但秩变化: {rank_structured} -> {rank_nuclear_optimized}")
    else:
        print("❌ 实验未达到预期效果，需要调整参数或方法")
    
    print(f"\n功能损失方面: {'改善' if error_improvement > 0 else '略有增加' if error_improvement < 0 else '保持不变'}")
    
    # 8. 核心发现总结
    print(f"\n步骤7: 核心发现总结")
    print(f"🎯 核心发现:")
    if nuc_norm_nuclear_optimized < nuc_norm_structured:
        print("   - 核范数importance-based方法有效降低了ΔB的核范数，证明了方法的有效性")
    if rank_nuclear_optimized < rank_structured:
        print("   - 优化方法实现了更好的低秩近似")
    if error_improvement > 0:
        print("   - 在降低核范数的同时保持了更好的功能近似质量")
    elif error_improvement < 0:
        print("   - 核范数降低但功能损失略有增加，这是低秩近似的常见trade-off")
    
    print("\n--- 实验完成 ---")