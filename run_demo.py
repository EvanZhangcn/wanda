import torch
import torch.nn as nn
import argparse
import math
import sys
from typing import Tuple

def res_svd_compress(weight_matrix: torch.Tensor, r1: int, r2: int, use_lowrank=False) -> torch.Tensor:
    """
    使用ResSVD方法压缩给定的权重矩阵。
    更高效：避免 torch.diag；支持可选低秩近似 API。
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError(f"r1({r1}) 和 r2({r2}) 必须为正整数。")
    device = weight_matrix.device
    dtype = torch.float32  # 计算用 float32
    W = weight_matrix.to(device='cpu', dtype=dtype)

    def _svd_trunc(mat: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 可选使用低秩 SVD (在较大矩阵且 k 远小于 min(m,n) 时更省时)
        if use_lowrank and hasattr(torch.linalg, "svd") and k < min(mat.shape) - 1:
            # torch.svd_lowrank 在部分版本可用; 若不可用回退完整SVD
            try:
                U, S, V = torch.svd_lowrank(mat, q=k, niter=2)
                # 使 V 与 Vh 形式一致
                Vh = V.T
                return U[:, :k], S[:k], Vh[:k, :]
            except Exception:
                pass
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        return U[:, :k], S[:k], Vh[:k, :]

    # 第一次低秩
    U1, S1, Vh1 = _svd_trunc(W, r1)
    # W_r1 = U1 @ diag(S1) @ Vh1  => (U1 * S1) @ Vh1
    W_r1 = (U1 * S1) @ Vh1
    R = W - W_r1
    U2, S2, Vh2 = _svd_trunc(R, r2)
    R_r2 = (U2 * S2) @ Vh2
    W_hat = W_r1 + R_r2
    return W_hat.to(device=device, dtype=weight_matrix.dtype)

# --- 主程序 ---
parser = argparse.ArgumentParser(description="使用命令行参数测试ResSVD压缩算法。")
parser.add_argument("--ratio", type=float, required=True, help="整体目标压缩率(0~1, 例如 0.8 表示希望参数减少80%)。")
parser.add_argument("--r1", type=int, required=True, help="中间秩 (r1) 超参数。")
parser.add_argument("--model_path", type=str, default="/data/zhangliwen/mc/Llama-2-7b-hf", help="模型路径。")
parser.add_argument("--layer", type=str, default="model.layers.5.mlp.gate_proj", help="待压缩线性层模块名。")
parser.add_argument("--seed", type=int, default=42, help="随机种子。")
parser.add_argument("--device", type=str, default="cpu", help="计算设备(cpu 或 cuda:0 等)。")
parser.add_argument("--use_lowrank_svd", action="store_true", help="使用低秩 SVD 近似 (torch.svd_lowrank) 加速。")
# 可选：添加 --seq_len 用于构造更真实的输入张量
parser.add_argument("--seq_len", type=int, default=1, help="随机测试输入的序列长度(用于三维输入 (B, L, H))。")
args = parser.parse_args()

if not (0 < args.ratio < 1):
    sys.exit("错误: --ratio 必须在 (0,1) 范围内。")

if args.r1 <= 0:
    sys.exit("错误: --r1 必须为正整数。")

torch.manual_seed(args.seed)
if torch.cuda.is_available() and "cuda" in args.device:
    torch.cuda.manual_seed_all(args.seed)

overall_compression_ratio = args.ratio
intermediate_rank_r1 = args.r1
model_path = args.model_path
target_layer_name = args.layer

print(f"正在从 '{model_path}' 加载模型...(仅用于获取指定线性层权重)")
try:
    from transformers import AutoModelForCausalLM  # 延迟导入
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=None,  # 可按需要改为 'auto'
    )
    print("模型加载成功！")
except Exception as e:
    sys.exit(f"加载模型失败: {e}")

modules_dict = dict(model.named_modules())
if target_layer_name not in modules_dict:
    sys.exit(f"错误: 指定的层 '{target_layer_name}' 未找到。")
original_linear_layer = modules_dict[target_layer_name]
if not isinstance(original_linear_layer, nn.Linear):
    sys.exit(f"错误: 目标层 '{target_layer_name}' 不是 nn.Linear, 实际类型: {type(original_linear_layer)}")

original_weight = original_linear_layer.weight.data.clone()
output_dim, input_dim = original_weight.shape
print(f"层: {target_layer_name} 维度: (out={output_dim}, in={input_dim})")

# 目标: 压缩后参数 = (r1+r2)*(m+n), 原始 = m*n
params_retained_ratio = 1 - overall_compression_ratio  # 希望保留比例
# 使用 ceil 避免过度向下截断
raw_rank = params_retained_ratio * (output_dim * input_dim) / (output_dim + input_dim)
total_rank = max(1, math.ceil(raw_rank))

if intermediate_rank_r1 >= total_rank:
    sys.exit(f"中间秩 r1({intermediate_rank_r1}) 不能 >= 总秩 r({total_rank})。")
residual_rank_r2 = total_rank - intermediate_rank_r1
if residual_rank_r2 <= 0:
    sys.exit(f"残差秩 r2 计算为 {residual_rank_r2}, 请调整 r1 或 ratio。")

original_params = output_dim * input_dim
compressed_params = total_rank * (output_dim + input_dim)
actual_retained = compressed_params / original_params
actual_compression = 1 - actual_retained

print(f"\n--- 秩与参数统计 ---")
print(f"目标压缩率 (输入)      : {overall_compression_ratio:.4f}")
print(f"计算总秩 (ceil)        : {total_rank}")
print(f"r1 / r2                : {intermediate_rank_r1} / {residual_rank_r2}")
print(f"原参数量               : {original_params:,}")
print(f"压缩后参数量           : {compressed_params:,}")
print(f"实际保留比例           : {actual_retained:.6f}")
print(f"实际压缩率             : {actual_compression:.6f} (偏差 {actual_compression - overall_compression_ratio:+.6f})")

# --- 压缩 ---
compressed_weight_ressvd = res_svd_compress(
    original_weight, intermediate_rank_r1, residual_rank_r2, use_lowrank=args.use_lowrank_svd
)

# 标准低秩 SVD (同总秩)
print("\n执行标准 SVD 低秩近似...")
W32 = original_weight.to(device='cpu', dtype=torch.float32)
U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
U_r = U[:, :total_rank]
S_r = S[:total_rank]
Vh_r = Vh[:total_rank, :]
compressed_weight_svd = ((U_r * S_r) @ Vh_r).to(device=original_weight.device, dtype=original_weight.dtype)

# --- 权重重建误差 ---
fro_norm_W = torch.linalg.matrix_norm(original_weight.float()).item()
err_ressvd_fro = torch.linalg.matrix_norm(original_weight.float() - compressed_weight_ressvd.float()).item()
err_svd_fro = torch.linalg.matrix_norm(original_weight.float() - compressed_weight_svd.float()).item()
rel_ressvd = err_ressvd_fro / fro_norm_W
rel_svd = err_svd_fro / fro_norm_W

print(f"\n--- 1. 权重重建误差 ---")
print(f"Fro 绝对误差   SVD : {err_svd_fro:.6f} | ResSVD: {err_ressvd_fro:.6f}")
print(f"Fro 相对误差   SVD : {rel_svd:.6e} | ResSVD: {rel_ressvd:.6e}")
if err_ressvd_fro < err_svd_fro:
    improvement = (err_svd_fro - err_ressvd_fro) / err_svd_fro * 100
    print(f"结论: ResSVD 优于标准 SVD (绝对 Fro 误差降低 {improvement:.2f}%)")
else:
    print("结论: 本次设定下 ResSVD 未优于标准 SVD。")

# --- 输出误差测试 ---
device = torch.device(args.device)
original_linear_layer = original_linear_layer.to(device)
compressed_linear_layer_ressvd = nn.Linear(input_dim, output_dim, bias=original_linear_layer.bias is not None).to(device)
compressed_linear_layer_svd = nn.Linear(input_dim, output_dim, bias=original_linear_layer.bias is not None).to(device)
compressed_linear_layer_ressvd.weight.data.copy_(compressed_weight_ressvd.to(device))
compressed_linear_layer_svd.weight.data.copy_(compressed_weight_svd.to(device))
if original_linear_layer.bias is not None:
    compressed_linear_layer_ressvd.bias.data.copy_(original_linear_layer.bias.data)
    compressed_linear_layer_svd.bias.data.copy_(original_linear_layer.bias.data)

# 构造随机输入: 允许模拟 (B, L, H)
seq_len = max(1, args.seq_len)
batch = 1
input_data = torch.randn(batch, seq_len, input_dim, device=device, dtype=original_linear_layer.weight.dtype)
# 将 Linear 应用于最后一维
with torch.no_grad():
    original_output = original_linear_layer(input_data)
    output_ressvd = compressed_linear_layer_ressvd(input_data)
    output_svd = compressed_linear_layer_svd(input_data)

diff_ressvd = original_output - output_ressvd
diff_svd = original_output - output_svd
mse_ressvd = torch.mean(diff_ressvd.float() ** 2).item()
mse_svd = torch.mean(diff_svd.float() ** 2).item()
l2_ressvd = torch.linalg.vector_norm(diff_ressvd.float()).item()
l2_svd = torch.linalg.vector_norm(diff_svd.float()).item()

print(f"\n--- 2. 单批随机输入输出误差 ---")
print("指标                  |   SVD          |  ResSVD")
print(f"MSE                   | {mse_svd: .8e} | {mse_ressvd: .8e}")
print(f"L2 Norm               | {l2_svd: .8e} | {l2_ressvd: .8e}")