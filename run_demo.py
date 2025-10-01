import torch
import torch.nn as nn
import argparse
import math
import sys
from typing import Tuple

def get_llm(model_path, cache_dir="llm_weights"):
    """加载LLM模型并设置序列长度（参考wanda）"""
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





def svd_truncate(matrix: torch.Tensor, rank: int, use_lowrank: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    截断SVD分解
    Args:
        matrix: 输入矩阵
        rank: 保留的秩
        use_lowrank: 是否使用低秩近似加速
    Returns:
        截断后的 U, S, Vh
    """
    if use_lowrank and hasattr(torch.linalg, "svd") and rank < min(matrix.shape) - 1:
        try:
            #一种 近似的SVD 方法，适合只需要前 k 个奇异值/向量的场景。
            U, S, V = torch.svd_lowrank(matrix, q=rank, niter=2)
            return U[:, :rank], S[:rank], V.T[:rank, :]
        except Exception:
            pass
    
    # 标准SVD分解
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return U[:, :rank], S[:rank], Vh[:rank, :]




def res_svd_compress(weight_matrix: torch.Tensor, r1: int, r2: int, use_lowrank: bool = False) -> torch.Tensor:
    """
    ResSVD权重压缩算法
    核心思想：
    1. 对原始矩阵W进行r1秩的SVD分解得到W_r1
    2. 计算残差R = W - W_r1
    3. 对残差R进行r2秩的SVD分解得到R_r2  
    4. 最终结果：W_hat = W_r1 + R_r2
    
    Args:
        weight_matrix: 原始权重矩阵
        r1: 第一次SVD的秩
        r2: 残差SVD的秩
        use_lowrank: 是否使用低秩SVD加速
        
    Returns:
        压缩后的权重矩阵
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError(f"r1({r1}) 和 r2({r2}) 必须为正整数")
    
    device = weight_matrix.device
    W = weight_matrix.to(device='cpu', dtype=torch.float32)
    
    # 第一次SVD：W ≈ W_r1
    # S1: 形状 (r1,) （向量，不是对角阵）
    U1, S1, Vh1 = svd_truncate(W, r1, use_lowrank)
    W_r1 = (U1 * S1) @ Vh1
    
    # 第二次SVD：残差 R ≈ R_r2
    residual = W - W_r1
    U2, S2, Vh2 = svd_truncate(residual, r2, use_lowrank)
    R_r2 = (U2 * S2) @ Vh2
    print(f"原始矩阵范数: {torch.linalg.matrix_norm(W).item():.6f}")
    print(f"W_r1范数: {torch.linalg.matrix_norm(W_r1).item():.6f}")  
    print(f"残差范数: {torch.linalg.matrix_norm(residual).item():.6f}")
    print(f"残差占原始矩阵比例: {torch.linalg.matrix_norm(residual).item() / torch.linalg.matrix_norm(W).item():.6f}")
    # 组合结果
    compressed_weight = W_r1 + R_r2
    return compressed_weight.to(device=device, dtype=weight_matrix.dtype)



def standard_svd_compress(weight_matrix: torch.Tensor, total_rank: int) -> torch.Tensor:
    """标准SVD压缩（用于对比）"""
    device = weight_matrix.device
    W = weight_matrix.to(device='cpu', dtype=torch.float32)
    
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :total_rank]
    S_r = S[:total_rank]
    Vh_r = Vh[:total_rank, :]
    
    compressed = (U_r * S_r) @ Vh_r
    return compressed.to(device=device, dtype=weight_matrix.dtype)





def calculate_compression_params(output_dim: int, input_dim: int, compression_ratio: float, r1: int) -> Tuple[int, int, int]:
    """
    计算压缩参数
    
    Args:
        output_dim: 输出维度
        input_dim: 输入维度  
        compression_ratio: 目标压缩率
        r1: 第一次SVD的秩
        
    Returns:
        (总秩, r2, 压缩后参数量)
    """
    original_params = output_dim * input_dim
    retained_ratio = 1 - compression_ratio
    
    # 计算总秩：压缩后参数 = total_rank * (output_dim + input_dim)
    raw_rank = retained_ratio * original_params / (output_dim + input_dim)
    total_rank = max(1, math.ceil(raw_rank))
    
    if r1 >= total_rank:
        raise ValueError(f"r1({r1}) 必须小于总秩({total_rank})")
        
    r2 = total_rank - r1
    compressed_params = total_rank * (output_dim + input_dim)
    
    return total_rank, r2, compressed_params

def evaluate_compression(original_weight: torch.Tensor, compressed_ressvd: torch.Tensor, 
                        compressed_svd: torch.Tensor) -> dict:
    """评估压缩效果"""
    # 权重重建误差
    fro_norm = torch.linalg.matrix_norm(original_weight.float()).item()
    
    err_ressvd = torch.linalg.matrix_norm(original_weight.float() - compressed_ressvd.float()).item()
    err_svd = torch.linalg.matrix_norm(original_weight.float() - compressed_svd.float()).item()
    
    # err_ressvd / fro_norm 和 err_svd / fro_norm表示的是相对误差
    return {
        'fro_norm': fro_norm,
        'ressvd_abs_error': err_ressvd,
        'svd_abs_error': err_svd,
        'ressvd_rel_error': err_ressvd / fro_norm,
        'svd_rel_error': err_svd / fro_norm
    }

def test_output_error(original_layer: nn.Linear, compressed_ressvd: nn.Linear, 
                     compressed_svd: nn.Linear, input_shape: Tuple) -> dict:
    """测试输出误差"""
    # 生成随机输入 - 直接在层所在设备上生成
    device = original_layer.weight.device
    input_data = torch.randn(*input_shape, device=device, dtype=original_layer.weight.dtype)
    with torch.no_grad():
        original_output = original_layer(input_data)
        output_ressvd = compressed_ressvd(input_data)
        output_svd = compressed_svd(input_data)
    
    # 计算误差
    diff_ressvd = original_output - output_ressvd
    diff_svd = original_output - output_svd
    
    return {
        'ressvd_mse': torch.mean(diff_ressvd.float() ** 2).item(),
        'svd_mse': torch.mean(diff_svd.float() ** 2).item(),
        'ressvd_l2': torch.linalg.vector_norm(diff_ressvd.float()).item(),
        'svd_l2': torch.linalg.vector_norm(diff_svd.float()).item()
    }

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ResSVD权重压缩算法测试（基于wanda加载方式）")

    # 模型相关参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--cache_dir", type=str, default="llm_weights", help="模型缓存目录")

    # 压缩相关参数
    parser.add_argument("--ratio", type=float, required=True, help="目标压缩率(0-1，如0.8表示减少80%参数)")
    parser.add_argument("--r1", type=int, required=True, help="第一次SVD的秩")
    parser.add_argument("--layer", type=str, required=True, help="目标线性层名称，例如: model.layers.0.mlp.gate_proj")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_cuda", action="store_true", help="禁用CUDA，强制使用CPU")
    parser.add_argument("--device", type=str, default="cuda:0", help="默认计算设备（当CUDA可用时）")
    parser.add_argument("--use_lowrank_svd", action="store_true", help="使用低秩SVD加速")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设备选择（参考wanda）
    if args.no_cuda or not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 参数验证
    if not (0 < args.ratio < 1):
        sys.exit("错误: --ratio 必须在(0,1)范围内")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.manual_seed_all(args.seed)
    
    # 加载模型（使用wanda原版方式）
    print(f"正在从 '{args.model_path}' 加载模型...")
    try:
        from transformers import AutoTokenizer
        model = get_llm(args.model_path, args.cache_dir)
        model.eval()
        print(f"模型加载成功！序列长度: {model.seqlen}")
    except Exception as e:
        sys.exit(f"加载模型失败: {e}")


    # 加载 Tokenizer
    print("加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        print("分词器加载成功！")
    except Exception as e:
        print(f"分词器加载警告: {e}")
        print("继续运行，不影响权重压缩测试...")
        tokenizer = None

    # 如果需要校准数据，可以在这里加载
    # 现在暂时跳过，专注于权重压缩测试
    if tokenizer is not None and hasattr(args, 'use_calibration_data') and args.use_calibration_data:
        print(f"加载 {args.dataset} 校准数据...")
        # 这里可以添加具体的数据加载逻辑
        # dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, 
        #                            seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
        print("注意: 校准数据加载功能待实现，当前仅进行权重压缩测试")
    
    # 获取目标层
    print(f"\n查找目标层: {args.layer}")
    modules_dict = dict(model.named_modules())
    if args.layer not in modules_dict:
        print("可用的线性层:")
        linear_layers = []
        for name, module in modules_dict.items():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)
        
        # 按层级排序显示
        linear_layers.sort()
        for layer_name in linear_layers[:20]:  # 只显示前20个
            print(f"  - {layer_name}")
        if len(linear_layers) > 20:
            print(f"  ... 还有 {len(linear_layers) - 20} 个层")
        sys.exit(f"错误: 未找到层 '{args.layer}'")
        
    target_layer = modules_dict[args.layer]
    if not isinstance(target_layer, nn.Linear):
        sys.exit(f"错误: '{args.layer}' 不是Linear层，类型: {type(target_layer)}")
    
    # 获取权重信息
    original_weight = target_layer.weight.data.clone()
    output_dim, input_dim = original_weight.shape
    print(f"目标层: {args.layer}")
    print(f"权重维度: (输出={output_dim}, 输入={input_dim})")
    print(f"原始参数量: {output_dim * input_dim:,}")
    
    # 计算压缩参数
    try:
        total_rank, r2, compressed_params = calculate_compression_params(
            output_dim, input_dim, args.ratio, args.r1)
    except ValueError as e:
        print(f"\n参数建议:")
        original_params = output_dim * input_dim
        retained_ratio = 1 - args.ratio
        max_total_rank = math.floor(retained_ratio * original_params / (output_dim + input_dim))
        print(f"当前压缩率 {args.ratio} 下，最大总秩为: {max_total_rank}")
        print(f"建议 r1 设置为: {max_total_rank // 2} 到 {max_total_rank - 1} 之间")
        sys.exit(f"参数错误: {e}")
    
    original_params = output_dim * input_dim
    actual_compression = 1 - compressed_params / original_params
    
    print(f"\n=== 压缩参数统计 ===")
    print(f"目标压缩率: {args.ratio:.4f}")
    print(f"总秩 r1+r2: {total_rank} = {args.r1} + {r2}")
    print(f"压缩后参数量: {compressed_params:,}")
    print(f"实际压缩率: {actual_compression:.6f} (偏差: {actual_compression - args.ratio:+.6f})")
    
    # 执行压缩
    print(f"\n=== 执行ResSVD压缩 ===")
    print("开始ResSVD压缩...")
    compressed_weight_ressvd = res_svd_compress(
        original_weight, args.r1, r2, args.use_lowrank_svd).to(original_weight.dtype)
    
    print("开始标准SVD压缩...")
    compressed_weight_svd = standard_svd_compress(original_weight, total_rank).to(original_weight.dtype)
    
    # 评估权重重建误差
    weight_metrics = evaluate_compression(original_weight, compressed_weight_ressvd, compressed_weight_svd)
    
    print(f"\n=== 权重重建误差 ===")
    print(f"{'方法':<10} {'绝对误差(Fro)':<15} {'相对误差':<15}")
    print(f"{'='*40}")
    print(f"{'SVD':<10} {weight_metrics['svd_abs_error']:<15.6f} {weight_metrics['svd_rel_error']:<15.6e}")
    print(f"{'ResSVD':<10} {weight_metrics['ressvd_abs_error']:<15.6f} {weight_metrics['ressvd_rel_error']:<15.6e}")
    
    if weight_metrics['ressvd_abs_error'] < weight_metrics['svd_abs_error']:
        improvement = (weight_metrics['svd_abs_error'] - weight_metrics['ressvd_abs_error']) / weight_metrics['svd_abs_error'] * 100
        print(f"\n✅ ResSVD优于标准SVD (误差降低{improvement:.2f}%)")
    else:
        print(f"\n❌ ResSVD未优于标准SVD")
    
    # 创建压缩后的层进行输出测试
    print(f"\n=== 准备输出误差测试 ===")
    
    # 构建压缩层（保持在原设备上）
    compressed_layer_ressvd = nn.Linear(input_dim, output_dim, 
                                       bias=target_layer.bias is not None)
    compressed_layer_svd = nn.Linear(input_dim, output_dim, 
                                    bias=target_layer.bias is not None)

    # 将压缩层移动到目标层相同的设备
    target_device = target_layer.weight.device
    compressed_layer_ressvd = compressed_layer_ressvd.to(target_device, dtype=target_layer.weight.dtype)
    compressed_layer_svd = compressed_layer_svd.to(target_device, dtype=target_layer.weight.dtype)

    #compressed_layer_ressvd = compressed_layer_ressvd.to(target_device)
    #compressed_layer_svd = compressed_layer_svd.to(target_device)
    
    # 复制权重
    #compressed_layer_ressvd.weight.data.copy_(compressed_weight_ressvd.to(target_device))
    #compressed_layer_svd.weight.data.copy_(compressed_weight_svd.to(target_device))

    compressed_layer_ressvd.weight.data.copy_(compressed_weight_ressvd.to(target_device, dtype=target_layer.weight.dtype))
    compressed_layer_svd.weight.data.copy_(compressed_weight_svd.to(target_device, dtype=target_layer.weight.dtype))
    if target_layer.bias is not None:
        compressed_layer_ressvd.bias.data.copy_(target_layer.bias.data)
        compressed_layer_svd.bias.data.copy_(target_layer.bias.data)
    
    # 测试输出误差
    input_shape = (1, model.seqlen, input_dim)
    print(f"测试输入形状: {input_shape}")
    
    output_metrics = test_output_error(target_layer, compressed_layer_ressvd, 
                                     compressed_layer_svd, input_shape)
    
    print(f"\n=== 输出误差测试 ===")
    print(f"{'方法':<10} {'MSE':<20} {'L2范数':<20}")
    print(f"{'='*50}")
    print(f"{'SVD':<10} {output_metrics['svd_mse']:<20.8e} {output_metrics['svd_l2']:<20.8e}")
    print(f"{'ResSVD':<10} {output_metrics['ressvd_mse']:<20.8e} {output_metrics['ressvd_l2']:<20.8e}")
    
    print(f"\n=== 总结 ===")
    print(f"模型: {args.model_path}")
    print(f"目标层: {args.layer}")
    print(f"压缩率: {actual_compression:.4f}")
    print(f"参数减少: {original_params - compressed_params:,} ({(original_params - compressed_params)/original_params*100:.2f}%)")
    
    # 保存结果（可选）
    if hasattr(args, 'save_results') and args.save_results:
        results = {
            'model_path': args.model_path,
            'layer': args.layer,
            'compression_ratio': actual_compression,
            'r1': args.r1,
            'r2': r2,
            'weight_metrics': weight_metrics,
            'output_metrics': output_metrics
        }
        torch.save(results, f"ressvd_results_{args.layer.replace('.', '_')}.pt")
        print(f"结果已保存到: ressvd_results_{args.layer.replace('.', '_')}.pt")

if __name__ == "__main__":
    main()