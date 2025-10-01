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


def benchmark_performance(model, tokenizer, device, batch_size, num_repeats, seq_len, save_dir=None, prune_method=None):
    """
    对给定的模型进行性能评测（使用真实数据），测量延迟和吞吐量。
    """
    print("\n" + "*" * 30)
    print("Running Performance Benchmark (using real data)...")
    print(f"Batch Size: {batch_size}, Repetitions: {num_repeats}, Sequence Length: {seq_len}")
    print("*" * 30)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")


    # 1. 加载并预处理真实数据集
    print("Loading and preprocessing dataset for benchmark...")
    # 使用和您原始脚本相同的设置加载数据
    # 下面这个方式运行崩溃，不再支持了
    #dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]", trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # 定义预处理函数，使用模型自身的序列长度
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=seq_len)

    encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

    # 准备一个批次的数据
    sample_batch = encoded_dataset.select(range(batch_size))
    inputs = {
        key: torch.tensor(val).to(device)
        for key, val in sample_batch.to_dict().items()
        if key in tokenizer.model_input_names
    }
    print("Data preparation complete.")

    # 2. 预热（Warm-up）
    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(**inputs)
    torch.cuda.synchronize()

    # 3. 开始正式测量
    inference_times = []
    torch.cuda.reset_peak_memory_stats(device)  # 重置显存统计

    print(f"Running benchmark for {num_repeats} iterations...")
    for _ in range(num_repeats):
        torch.cuda.synchronize(device)
        start_time = time.time()

        with torch.no_grad():
            _ = model(**inputs)

        torch.cuda.synchronize(device)
        end_time = time.time()

        inference_times.append(end_time - start_time)

    # 清理
    torch.cuda.empty_cache()
    gc.collect()


    # 4. 计算并报告结果
    avg_inference_time = np.mean(inference_times)
    throughput = batch_size / avg_inference_time
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print("\n--- Benchmark Results ---")
    print(f"Average Inference Time (Latency): {avg_inference_time * 1000:.4f} ms")
    print(f"Throughput: {throughput:.2f} samples/second")
    print(f"Peak GPU Memory Allocated: {peak_memory_mb:.2f} MB")
    print("-------------------------\n")

    # 将性能结果也追加到日志文件中
    if save_dir and prune_method:
        save_filepath = os.path.join(save_dir, f"log_{prune_method}.txt")
        with open(save_filepath, "a") as f:  # 使用 "a" (append) 模式追加
            f.write("\n--- Benchmark Results ---\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Latency: {avg_inference_time * 1000:.4f} ms\n")
            f.write(f"Throughput: {throughput:.2f} samples/sec\n")
            f.write(f"Peak Memory: {peak_memory_mb:.2f} MB\n")



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

    parser.add_argument('--alpha', type=float, default=0.5, help='Strength for smoothing migration (alpha hyperparameter).')

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
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda_compensation":
            print("Generating compensation parameters...")
            # 1. 调用函数，直接在内存中获取补偿参数字典
            model = prune_wanda_with_compensation(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
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

    # 运行性能基准测试
    if args.run_benchmark:
        benchmark_performance(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.benchmark_batch_size,
            num_repeats=args.benchmark_repeats,
            seq_len=model.seqlen,  # 使用模型自身的序列长度
            save_dir=args.save,
            prune_method=args.prune_method
        )
    

if __name__ == '__main__':
    main()