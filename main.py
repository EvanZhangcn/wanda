import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers, prune_wanda_with_compensation, create_final_compensated_model
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    #设置模型一次性处理的最大token数（即序列长度）
    if model.config.max_position_embeddings > 2048:
        model.seqlen = 2048
    else:
        model.seqlen = model.config.max_position_embeddings

    return model

def main():
    #命令行参数解析器
    parser = argparse.ArgumentParser()
    #定义预期要接收的参数
    parser.add_argument('--model', type=str, help='LLaMA model') #help是参数的描述信息
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "wanda_compensation", "sparsegpt",
                    "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument('--compensation_rank', type=int, default=None, help='Rank for low-rank approximation in compensation')
    parser.add_argument('--compensation_params_path', type=str, default=None, help='Path to save/load compensation parameters.')
    
    
    parser.add_argument('--alpha', type=float, default=0.5, help='Strength for smoothing migration (alpha hyperparameter).')
    parser.add_argument('--spectral_alpha', type=float, default=0.5, help='Power for spectral amplification (e.g., 0.5, 1.0).')
    #解析传入的参数，存入args
    args = parser.parse_args()

    # Setting seeds for reproducibility，设置随即种子
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        #如果是结构化剪枝
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        #将int()函数，逐个作用到后面的列表中，Eg: "2:4" -> str类型的列表：["2", "4"] -> int类型的列表：[2, 4]
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval() #设置为模型为评估模式
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    #选择不同的剪枝方法
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            #prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            prune_wanda_with_compensation(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                                          enable_compensation=False)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda_compensation":
            # 检查是否提供了补偿参数路径
            if not args.compensation_params_path:
                raise ValueError("Compensation parameters path must be provided for wanda_compensation method.")
            print("Generating compensation parameters...")
            compensation_params = prune_wanda_with_compensation(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, enable_compensation=True)
            # 补偿参数已在函数内部保存，这里只需加载和替换模型
            if compensation_params:
                print("Creating final compensated model...")
                model = create_final_compensated_model(model, args.compensation_params_path)
                print("Final compensated model created.")
            else:
                print("Warning: No compensation parameters were generated!")
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model) #检查剪枝后的稀疏度
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)

    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save) #保存目录不存在就创建
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    #写入日志文件
    with open(save_filepath, "w") as f:
        # eg: 写入log_wanda.txt文件表头：剪枝方法、实际稀疏度、测试困惑度
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()