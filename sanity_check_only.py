import argparse
import torch
import sys
import os
from transformers import AutoModelForCausalLM

# 从您现有的prune.py文件中导入我们需要的分析函数
from lib.prune import find_layers, check_nm_sparsity_ratio, analyze_delta_B_key_patterns

def run_sanity_check():
    # 1. 设置一个简化的命令行参数解析器，只包含必需的参数
    parser = argparse.ArgumentParser(description="Run a standalone sanity check for delta_b sparsity using random matrices.")
    parser.add_argument('--model', type=str, required=True, help='LLaMA model identifier (e.g., decapoda-research/llama-7b-hf). Used only to get layer shapes.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level for unstructured pruning.')
    parser.add_argument("--sparsity_type", type=str, default="2:4", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    
    # === 新增代码：用于日志文件保存 ===
    parser.add_argument('--log_file', type=str, default=None, help='Path to save terminal output log.')
    # ====================================
    
    args = parser.parse_args()

    # 解析 N:M 结构化稀疏参数
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    # === 新增代码：日志文件重定向 ===
    if args.log_file:
        # 获取日志文件的目录，如果不存在则创建
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # 将标准输出流重定向到文件，同时保留对原始stdout的引用
        original_stdout = sys.stdout
        sys.stdout = open(args.log_file, 'w')
    # ====================================

    print(f"--- Starting Sanity Check for model config: {args.model} ---")
    print(f"--- Sparsity Target: {args.sparsity_ratio}, Structure: {args.sparsity_type} ---")

    # 2. 以“低资源模式”加载模型，我们只需要其结构信息（各层形状），不需要权重值
    print("Loading model configuration to determine layer shapes...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        cache_dir=args.cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="cpu"
    )
    
    # 3. 遍历模型的每一层，对其中的线性层执行随机实验
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"\n--- Processing Layer {i}, Sublayer {name} ---")
            
            weight_shape = subset[name].weight.shape
            scaler_row_shape = (1, weight_shape[1])
            device = subset[name].weight.device
            dtype = subset[name].weight.dtype

            print(f"   Step 1: Creating random matrices with shape W:{weight_shape}, Act:{scaler_row_shape}...")
            random_weight = torch.randn(weight_shape, device=device, dtype=dtype)
            random_scaler_row = torch.rand(scaler_row_shape, device=device, dtype=dtype)
            
            print("   Step 2: Performing identical pruning operations on random data...")
            random_W_metric = torch.abs(random_weight) * torch.sqrt(random_scaler_row)

            random_W_mask_unstructured = (torch.zeros_like(random_W_metric) == 1)
            random_sort_res = torch.sort(random_W_metric, dim=-1, stable=True)
            random_indices = random_sort_res[1][:, :int(random_W_metric.shape[1] * args.sparsity_ratio)]
            random_W_mask_unstructured.scatter_(1, random_indices, True)
            random_B_unstructured = random_weight.clone()
            random_B_unstructured[random_W_mask_unstructured] = 0

            random_W_mask_structured = (torch.zeros_like(random_W_metric) == 1)
            if prune_n != 0:
                for ii in range(random_W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = random_W_metric[:, ii:(ii + prune_m)].float()
                        random_W_mask_structured.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                random_W_mask_structured = random_W_mask_unstructured.clone()
            random_B_structured = random_weight.clone()
            random_B_structured[random_W_mask_structured] = 0
            
            delta_B_random = random_B_unstructured - random_B_structured

            print(f"   Step 3: Analyzing N:M sparsity compliance for the RANDOM delta_B...")
            check_nm_sparsity_ratio(delta_B_random, n=2, m=4, dimension='row')
            check_nm_sparsity_ratio(delta_B_random, n=4, m=8, dimension='row')

            print(f"   Step 4: Analyzing key patterns in the RANDOM delta_B matrix...")
            analyze_delta_B_key_patterns(delta_B_random, i, f"{name}_sanity_check")

    print("\n--- Sanity Check Completed ---")
    
    # === 新增代码：恢复标准输出 ===
    if args.log_file:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Terminal output has been saved to: {args.log_file}")
    # =============================

if __name__ == '__main__':
    run_sanity_check()