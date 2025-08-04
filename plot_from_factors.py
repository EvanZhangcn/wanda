import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 配置 ---
# 您保存的补偿参数文件的路径
PARAMS_FILE = "./compensation_params/llama2-7b_rank32_params.pt"

# 生成的图像将保存在这个文件夹下
SAVE_DIR = "reconstructed_spectrums_per_layer"
os.makedirs(SAVE_DIR, exist_ok=True)
# --- 结束配置 ---

print(f"正在加载补偿参数文件: {PARAMS_FILE}")
# 加载时，将数据映射到CPU，确保在任何机器上都能运行
compensation_params = torch.load(PARAMS_FILE, map_location=torch.device('cpu'))

print(f"文件中共有 {len(compensation_params)} 个层的补偿参数。")
print("开始为每一个层生成奇异值谱图...")

# 遍历文件中的每一个层
for layer_name, params in compensation_params.items():
    if 'L1' not in params:
        print(f"跳过层 {layer_name}，因为它不包含L1矩阵。")
        continue

    # --- 恢复奇异值 ---
    L1 = params['L1']
    _, s_sqrt, _ = torch.linalg.svd(L1.float())
    singular_values = (s_sqrt**2).numpy()
    
    # --- 开始为当前层画图 ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    title_prefix = f"Top {len(singular_values)} Singular Values (Reconstructed)"
    
    # 1. 绘制左图
    axes[0].plot(singular_values)
    axes[0].set_yscale('log')
    axes[0].set_title(f"Complete Singular Value Spectrum\n{layer_name}")
    axes[0].set_xlabel("Singular Value Index")
    axes[0].set_ylabel("Singular Value (log scale)")
    axes[0].grid(True)

    # 2. 绘制右图 (内容与左图相同，因为我们只有32个值)
    axes[1].plot(singular_values, 'b-')
    axes[1].set_yscale('log')
    axes[1].set_title(f"First {len(singular_values)} Singular Values\n{layer_name}")
    axes[1].set_xlabel("Singular Value Index")
    axes[1].set_ylabel("Singular Value (log scale)")
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # 将层名中的'.'替换为'_'，使其成为有效的文件名
    safe_layer_name = layer_name.replace('.', '_')
    save_path = os.path.join(SAVE_DIR, f"{safe_layer_name}.png")
    
    plt.savefig(save_path)
    print(f"  -> 已保存图像: {save_path}")
    
    # 关闭当前图像，以免在循环中占用过多内存
    plt.close(fig)

print("\n所有图像生成完成！")