import torch
from torch.sparse import to_sparse_semi_structured
import time
import numpy as np

# -----------------
# 1. 初始化设置
# -----------------
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

torch.manual_seed(42)
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
    print("需要支持半结构化稀疏的 NVIDIA GPU (计算能力8.0+)")
    exit()
device = torch.device("cuda:0")

matrix_rows = 4096
matrix_cols = 4096
input_features = 2048
sparsity_ratio = 0.5
prune_n, prune_m = 2, 4 # For 2:4 sparsity

# ----------------------------------------------------
# 2. 模拟 Wanda 剪枝流程以生成真实的 ΔB
#    (逻辑移植自您的 prune.py)
# ----------------------------------------------------
print("--- 步骤1: 通过模拟Wanda剪枝生成真实的ΔB矩阵 ---")

# a) 创建模拟的原始权重和激活值尺度
original_weight = torch.rand((matrix_rows, matrix_cols), dtype=torch.float16, device=device)
# 模拟激活值尺度 (scaler_row)，其形状需要与权重匹配
scaler_row = torch.rand((matrix_rows, 1), dtype=torch.float16, device=device) * 10
W_metric = torch.abs(original_weight) * torch.sqrt(scaler_row) # Wanda 重要性分数

# b) 计算非结构化剪枝矩阵 B_unstructured
W_mask_unstructured = (torch.zeros_like(W_metric) == 1)
sort_res = torch.sort(W_metric, dim=-1, stable=True)
indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
W_mask_unstructured.scatter_(1, indices, True)
B_unstructured = original_weight.clone()
B_unstructured[W_mask_unstructured] = 0

# c) 计算 2:4 结构化剪枝矩阵 B_structured
W_mask_structured = (torch.zeros_like(W_metric) == 1)
for ii in range(W_metric.shape[1]):
    if ii % prune_m == 0:
        tmp = W_metric[:, ii:(ii + prune_m)].float()
        W_mask_structured.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
B_structured = original_weight.clone()
B_structured[W_mask_structured] = 0

# d) 计算最终的 ΔB 矩阵
delta_B = B_unstructured - B_structured
delta_B_temp = torch.zeros_like(delta_B)
#for i in range(167772):
for i in range(409):
    for j in range(4):
        delta_B_temp[i,j] = 1
check_nm_sparsity_ratio(delta_B, 2, 4, dimension='row')
check_nm_sparsity_ratio(delta_B_temp, 2, 4, dimension='row')
print("真实的ΔB矩阵已生成。\n")


# ----------------------------------------------------
# 3. 准备不同格式的矩阵用于对比
# ----------------------------------------------------
print("--- 步骤2: 准备不同格式的矩阵进行对比 ---")
# a) 稠密矩阵 (基准)
dense_matrix_for_test = delta_B.clone() # 使用生成的 delta_B

# b) 2:4 半结构化稀疏矩阵
# 注意：这里我们传入的是 delta_B 本身，因为它天然符合2:4模式
sparse_2_4_matrix_for_test = to_sparse_semi_structured(delta_B.clone())

# c) CSR 稀疏矩阵
# 我们也使用生成的 delta_B 来创建CSR矩阵
sparse_csr_matrix_for_test = delta_B.clone().to_sparse_csr()
delta_B_temp_for_test = delta_B_temp.clone().to_sparse_csr()

# d) 创建输入张量
input_tensor = torch.rand((matrix_cols, input_features), dtype=torch.float16, device=device)
print("所有矩阵格式准备完毕。\n")


# ----------------------------------------------------------------
# 4. 进行计时比较
# ----------------------------------------------------------------
print("--- 步骤3: 开始性能基准测试 ---")
# ... (这部分计时代码与之前完全相同)
# 预热 GPU
for _ in range(20):
    _ = torch.mm(dense_matrix_for_test, input_tensor)
torch.cuda.synchronize()

# a) 测试稠密矩阵乘法
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = torch.mm(dense_matrix_for_test, input_tensor)
torch.cuda.synchronize()
dense_time = (time.time() - start_time) / 100 * 1000

# b) 测试 2:4 稀疏矩阵乘法
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = torch.mm(sparse_2_4_matrix_for_test, input_tensor)
torch.cuda.synchronize()
sparse_2_4_time = (time.time() - start_time) / 100 * 1000

# c) 测试 CSR 稀疏矩阵乘法
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        #_ = torch.sparse.mm(input_tensor.t(), sparse_csr_matrix_for_test)
        #_ = input_tensor.t() @ sparse_csr_matrix_for_test
        #_ = torch.matmul(input_tensor.t(), sparse_csr_matrix_for_test)
        _ = torch.sparse.mm(input_tensor.t(), delta_B_temp_for_test)
torch.cuda.synchronize()
sparse_csr_time = (time.time() - start_time) / 100 * 1000


# ----------------------------------------------------
# 5. 量化并报告结果
# ----------------------------------------------------
print("\n--- 微基准测试结果 ---")
print(f"稠密 ΔB 乘法 (基准): {dense_time:.4f} ms")
print(f"2:4 稀疏 ΔB 乘法: {sparse_2_4_time:.4f} ms")
print(f"CSR 稀疏 ΔB 乘法: {sparse_csr_time:.4f} ms")
print("-" * 25)

speedup_2_4 = dense_time / sparse_2_4_time if sparse_2_4_time > 0 else float('inf')
speedup_csr = dense_time / sparse_csr_time if sparse_csr_time > 0 else float('inf')

print(f"2:4 硬件加速带来的 Speed Up: {speedup_2_4:.2f}x")
print(f"CSR (高稀疏度) 带来的 Speed Up: {speedup_csr:.2f}x")