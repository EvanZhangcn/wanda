import torch

small_val = 1e-5
device = 'cuda'

# 1. 创建包含非正规数的 FP16 稀疏矩阵 (我们已知这步是OK的)
A_dense = torch.tensor([[small_val]], dtype=torch.float16, device=device)
A_csr = A_dense.to_sparse_csr()

# 2. 创建一个简单的 FP16 稠密矩阵用于乘法
B_dense = torch.tensor([[1.0]], dtype=torch.float16, device=device)

# --- 实验A: 在 FP16 下进行稀疏计算 ---
print("--- Running SpMM in FP16 ---")
C_fp16 = torch.sparse.mm(A_csr, B_dense)
print(f"Result in FP16: {C_fp16.item():.8f}")
if C_fp16.item() == 0:
    print("✅ The value was flushed to zero during FP16 COMPUTATION.\n")
else:
    print("❌ The value survived FP16 COMPUTATION.\n")


# --- 实验B: 在 FP32 下进行稀疏计算 ---
print("--- Running SpMM in FP32 ---")
A_csr_32 = A_csr.to(torch.float32)
B_dense_32 = B_dense.to(torch.float32)
C_fp32 = torch.sparse.mm(A_csr_32, B_dense_32)
print(f"Result in FP32: {C_fp32.item():.8f}")
if C_fp32.item() != 0:
    print("✅ The value survived during FP32 COMPUTATION.\n")
else:
    print("❌ The value was flushed to zero during FP32 COMPUTATION.\n")