import time 
import heapq 
import torch 
import torch.nn as nn 
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
        
        # 加载补偿参数
        self.L1 = nn.Parameter(compensation_params['L1'], requires_grad=False) # (out_features, rank)
        self.L2 = nn.Parameter(compensation_params['L2'], requires_grad=False)  # (rank, in_features)
        self.s_inv = nn.Parameter(compensation_params['s_inv'], requires_grad=False)
        
    def forward(self, x):
            # 分支1: 原始的稀疏计算
            output_sparse = self.sparse_linear(x)
            
            # 分支2: 并行的低秩补偿计算
            # 注意：x的形状可能是 (batch_size, seq_len, input_dim) 或 (batch_size*seq_len, input_dim)
            original_shape = x.shape
            
            # 将输入重塑为二维矩阵以便进行矩阵乘法
            if x.dim() > 2:
                x_2d = x.view(-1, x.shape[-1])  # (batch_size*seq_len, input_dim)
            else:
                x_2d = x
                
            # 确保设备一致性并应用smoothing，同时确保数据类型一致
            x_smoothed = x_2d.to(self.s_inv.device, dtype=self.s_inv.dtype) * self.s_inv  # (batch*seq, in_features)
            
            # 低秩补偿计算: x_smoothed @ (L1 @ L2)^T = x_smoothed @ L2^T @ L1^T
            # L2^T: (in_features, rank), L1^T: (rank, out_features)
            # 确保所有参数都在同一设备和数据类型
            L2_t = self.L2.t().to(x_smoothed.device, dtype=x_smoothed.dtype)
            L1_t = self.L1.t().to(x_smoothed.device, dtype=x_smoothed.dtype)
            
            temp = x_smoothed @ L2_t   # (batch*seq, rank)
            output_compensation = temp @ L1_t  # (batch*seq, out_features)
            
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

def create_final_compensated_model(structured_model, compensation_params):
    # 遍历模型的所有模块
    for name, module in structured_model.named_modules():
        if name in compensation_params:
            if isinstance(module, nn.Linear):
                print(f"Replacing layer: {name}")
                
                # 创建新的补偿层
                # 将补偿参数移动到与模型相同的设备和数据类型
                params = compensation_params[name]
                target_dtype = module.weight.dtype
                target_device = module.weight.device
                
                for k, v in params.items():
                    params[k] = v.to(device=target_device, dtype=target_dtype)

                new_layer = CompensatedSparseLinear(module, params)
                
                # 获取父模块并替换
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = structured_model.get_submodule(parent_name)
                setattr(parent_module, child_name, new_layer)
                
    return structured_model

def calculate_smoothing_scale_factor(activation_scales, delta_B, alpha=0.5):
    """
    根据激活尺度和误差矩阵计算Smoothing缩放因子s。
    
    参数:
    - activation_scales: 形状为 (1, in_features)，代表每个输入通道的激活尺度。
    - delta_B: 形状为 (out_features, in_features)，即误差矩阵。
    - alpha: 平滑强度，一个0到1之间的超参数。
    
    返回:
    - s: 形状为 (1, in_features) 的缩放因子。
    """
    # 计算delta_B每行的范数（代表每个输入通道的误差大小）
    #delta_B_row_norms = torch.linalg.norm(delta_B, dim=0, keepdim=True)
    # 计算 delta_B 每一列的绝对值最大值
    delta_B_row_norms = torch.max(torch.abs(delta_B), dim=0, keepdim=True)[0]

    # 平滑公式
    # 为了防止除零，添加一个小的epsilon
    epsilon = 1e-6
    s = torch.pow(activation_scales, alpha) / (torch.pow(delta_B_row_norms, 1 - alpha) + epsilon)
    
    # 如果一个通道在 delta_B 中全为零，
    # 那么它的缩放因子应该为1，即不进行缩放。
    # 我们通过检查原始的 delta_B_row_norms 来定位这些通道。
    s[delta_B_row_norms == 0] = 1.0


    # 对s进行裁剪，防止出现极端值
    s = torch.clamp(s, min=1e-5) 
    return s

def low_rank_approximation_factors(matrix, rank):
    """对矩阵进行SVD并返回低秩因子"""
    print(f"    Computing SVD factors for matrix shape: {matrix.shape}")
    
    # 如果rank为None，设置为矩阵最小维度的1/4
    if rank is None:
        rank = min(matrix.shape[0], matrix.shape[1]) // 4
    
    # 确保rank不超过矩阵的最小维度
    max_rank = min(matrix.shape[0], matrix.shape[1])
    rank = min(rank, max_rank)
    
    if rank <= 0:
        print(f"    Warning: rank {rank} is invalid, returning zero factors")
        return (torch.zeros(matrix.shape[0], 1, device=matrix.device, dtype=matrix.dtype), 
                torch.zeros(1, matrix.shape[1], device=matrix.device, dtype=matrix.dtype))
    
    print(f"    Using rank: {rank}")
    
    # 保存原始数据类型和设备
    original_dtype = matrix.dtype
    original_device = matrix.device
    try:
        # 转换为float32进行SVD计算，并移到CPU以节省GPU内存
        matrix_cpu = matrix.float().cpu()
        
        U, S, Vh = torch.linalg.svd(matrix_cpu, full_matrices=False)
        # --- SVD计算结束 ---
        
        # 截断到指定秩
        U_k = U[:, :rank]  # [m, rank]
        S_k = S[:rank]     # [rank]
        Vh_k = Vh[:rank]   # [rank, n]
        
        # 将奇异值分配给两个因子
        sqrt_S = torch.sqrt(S_k + 1e-10)  # 添加小常数以避免数值问题
        L1 = U_k @ torch.diag(sqrt_S)      # [m, rank]
        L2 = torch.diag(sqrt_S) @ Vh_k      # [rank, n]
        
        # 转换回原始数据类型，但保持在CPU上
        L1 = L1.to(dtype=original_dtype)
        L2 = L2.to(dtype=original_dtype)
        
        print(f"    Factor dimensions - L1: {L1.shape}, L2: {L2.shape}")
        print(f"    Successfully computed low-rank factors")
        return L1, L2
        
    except Exception as e:
        print(f"    Error in SVD factor computation: {e}")
        print(f"    Returning zero factors for safety")
        return (torch.zeros(matrix.shape[0], 1, dtype=original_dtype), 
                torch.zeros(1, matrix.shape[1], dtype=original_dtype))
        

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

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    # if "model.embed_tokens" in model.hf_device_map:
    #     device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype #获取模型第一层(next函数)参数的数据类型
    #hidden_size就是每一个token的向量表示的维度,原始数据是[batch_size, seq_len]，输入到模型经过词嵌入层，得到每个token的词嵌入向量表示
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    #下面i表示当前batch的索引
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp #保存输入到预分配的inps中，捕获输入
            cache['i'] += 1 #更新索引
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError #故意抛出异常，终止当前的forward函数后续计算
    layers[0] = Catcher(layers[0]) #只捕获第一层的输入，layers是原模型层的引用，这里相当于把原模型第一层替换为Catcher类的实例，

    #替换的是大块的LlamaDecoderLayer(0)，正常情况是：output = layers[0](embedded, attention_mask, position_ids)
    #现在变成Catcher(layers[0])(embedded, attention_mask, position_ids)
    for batch in dataloader:
        try:
            model(batch[0].to(device)) #等价于：model.forward(batch[0].to(device)),进行前向传播
        except ValueError:
            pass 
    layers[0] = layers[0].module #再把Catcher类实例 恢复为 原模型的第一层

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

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
            #注册完钩子函数后，开始进行这批样本的前向传播（无梯度更新）,这样钩子函数能够捕获到每个样本的输入和输出
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0] #相当于layer.forward(第j个样本),最后【0】拿到返回值第一个值，输出
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
    执行带有补偿逻辑的Wanda剪枝并返回补偿参数
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # 用于存储补偿参数的局部字典
    compensation_params = {}
    
    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    
    for i in range(len(layers)):
        print(f"\n=== Processing Layer {i} ===")
        layer = layers[i]
        subset = find_layers(layer)

        # 设置设备，默认使用传入的device参数
        dev = device
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        # 对每个线性层进行剪枝
        for name in subset:
            print(f"Processing layer {i} sublayer {name}")
            
            original_weight = subset[name].weight.data.clone()
            W_metric = torch.abs(original_weight) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            print(f"  Step 1: Computing unstructured pruning target...")
            # 计算非结构化剪枝目标
            W_mask_unstructured = (torch.zeros_like(W_metric) == 1)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
            W_mask_unstructured.scatter_(1, indices, True)

            B_unstructured = original_weight.clone()
            B_unstructured[W_mask_unstructured] = 0
            print(f"    B_unstructured sparsity: {(B_unstructured == 0).float().mean():.6f}")


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



            print(f"  Step 3: Computing compensation matrix (delta_B)...")
            delta_B = B_unstructured - B_structured

            if torch.norm(delta_B) > 1e-8:
                print(f"  Step 3a: Applying Smoothing to delta_B...")
                
                try:
                    # 1. 获取激活尺度 (现在act_scales只包含每个通道的最大激活值)
                    activation_scales = wrapped_layers[name].act_scales.to(dev)
                    
                    # 确保activation_scales是正确的形状 (1, in_features)
                    if activation_scales.dim() == 1:
                        activation_scales = activation_scales.unsqueeze(0)  # 从 (in_features,) 转为 (1, in_features)

                    # ======================= 新增诊断代码块 开始 =======================
                    print(f"    - Analyzing delta_B for layer {name}...")
                    # 计算将作为分母的 delta_B 逐列最大绝对值
                    delta_B_col_max_abs = torch.max(torch.abs(delta_B), dim=0)[0]

                    # 检查有多少列的最大绝对值为零
                    zero_cols_count = (delta_B_col_max_abs == 0).sum().item()

                    if zero_cols_count > 0:
                        print(f"    - WARNING: Found {zero_cols_count} columns in delta_B that are entirely zero.")
                        # （可选）如果想看得更详细，可以取消下面这行的注释
                        # print(f"    - Indices of zero columns: {torch.where(delta_B_col_max_abs == 0)[0].cpu().numpy()}")
                    else:
                        print("    - OK: No all-zero columns found in delta_B.")
                    # ======================= 新增诊断代码块 结束 =======================

                    # 2. 计算缩放因子 s 和其倒数 s_inv
                    #s = calculate_smoothing_scale_factor(activation_scales, delta_B)
                    s = calculate_smoothing_scale_factor(activation_scales, delta_B, alpha=args.alpha)
                    s_inv = 1.0 / s

                    # 3. 对 delta_B 进行变换
                    delta_B_smoothed = delta_B * s

                    import os
                    import numpy as np
                    import matplotlib.pyplot as plt
                    from kneed import KneeLocator
                    # =================================================================================
                    # === SVD分析、能量计算、拐点检测的最终代码 ===
                    # =================================================================================
                    print(f"  Step 3b: Analyzing SVD Spectrum for delta_B vs delta_B_smoothed...")
                    # --- 1. 定义一个可复用的绘图辅助函数 ---
                    def plot_svd_analysis(axes_row, matrix, matrix_name, title_prefix):
                        """
                        对给定的矩阵进行SVD分析和绘图，画在一行子图上。
                        :param axes_row: 一行两个子图 (e.g., axes[0])
                        :param matrix: 要分析的torch矩阵 (delta_B or delta_B_smoothed)
                        :param matrix_name: 矩阵的名字，用于标题 (e.g., "delta_B")
                        :param title_prefix: 图表的主标题前缀 (e.g., "Layer 0 Sublayer mlp.proj")
                        """
                        with torch.no_grad():
                            singular_values = torch.linalg.svdvals(matrix.float()).cpu().numpy()

                        # 计算能量
                        total_energy = np.sum(singular_values**2)
                        cumulative_energy = np.cumsum(singular_values**2)
                        cumulative_energy_ratio = cumulative_energy / total_energy

                        # --- 左侧子图：完整谱图 ---
                        ax_left = axes_row[0]
                        ax_left.plot(singular_values)
                        ax_left.set_yscale('log')
                        ax_left.set_title(f'Complete Spectrum of {matrix_name}\n{title_prefix}')
                        ax_left.set_xlabel('Singular Value Index')
                        ax_left.set_ylabel('Singular Value (log scale)')
                        ax_left.grid(True)

                        # --- 右侧子图：局部放大和标注 ---
                        ax_right = axes_row[1]
                        num_values_to_show = 64
                        plot_indices = np.arange(min(len(singular_values), num_values_to_show))
                        safe_singular_values = singular_values[:len(plot_indices)] + 1e-10

                        ax_right.plot(plot_indices, safe_singular_values, 'b-')
                        ax_right.set_yscale('log')
                        ax_right.set_title(f'First {num_values_to_show} Values of {matrix_name}\n{title_prefix}')
                        ax_right.set_xlabel('Singular Value Index')
                        ax_right.set_ylabel('Singular Value (log scale)')
                        ax_right.grid(True)
                        
                        # 标注肘部点
                        try:
                            if len(plot_indices) > 1:
                                kneedle = KneeLocator(plot_indices, np.log10(safe_singular_values), S=1.0, curve="convex", direction="decreasing")
                                if kneedle.elbow is not None:
                                    x, y = kneedle.elbow, singular_values[kneedle.elbow]
                                    energy = cumulative_energy_ratio[x] * 100
                                    ax_right.annotate(f'Elbow (Rank {x})\nEnergy: {energy:.2f}%', xy=(x, y), xytext=(15, 40), textcoords='offset points', arrowprops=dict(facecolor='green', shrink=0.05), bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
                        except Exception as e:
                            print(f"      - Could not find or annotate elbow point for {matrix_name}: {e}")

                        # 标注秩32
                        if len(plot_indices) >= 32:
                            x, y = 31, singular_values[31]
                            energy = cumulative_energy_ratio[x] * 100
                            ax_right.annotate(f'Top 32 Ranks\nEnergy: {energy:.2f}%', xy=(x, y), xytext=(-80, -40), textcoords='offset points', arrowprops=dict(facecolor='red', shrink=0.05))

                        # 标注秩64
                        if len(plot_indices) >= 64:
                            x, y = 63, singular_values[63]
                            energy = cumulative_energy_ratio[x] * 100
                            ax_right.annotate(f'Top 64 Ranks\nEnergy: {energy:.2f}%', xy=(x, y), xytext=(-80, 20), textcoords='offset points', arrowprops=dict(facecolor='purple', shrink=0.05))

                    # --- 2. 创建 2x2 的图纸并调用辅助函数 ---
                    # 设置保存路径
                    save_dir = "svd_analysis_plots_2x2"
                    os.makedirs(save_dir, exist_ok=True)
                    full_module_path = f"model.layers.{i}.{name}"
                    filename_part = full_module_path.replace('.', '_')
                    filepath = os.path.join(save_dir, f"svd_comparison_{filename_part}.png")

                    # 创建2x2子图
                    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
                    title_prefix = f'Layer {i} Sublayer {name}'
                    fig.suptitle(f'SVD Analysis Comparison\n{title_prefix}', fontsize=16)

                    # 上面一行：分析 delta_B (平滑前)
                    plot_svd_analysis(axes[0], delta_B, "delta_B (Before Smoothing)", title_prefix)

                    # 下面一行：分析 delta_B_smoothed (平滑后)
                    plot_svd_analysis(axes[1], delta_B_smoothed, "delta_B_smoothed (After Smoothing)", title_prefix)

                    # 保存并关闭图形
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局为总标题留出空间
                    plt.savefig(filepath)
                    plt.close(fig)
                    print(f"    2x2 Comparison plot saved to: {filepath}")


                    # 清理中间变量以节省内存
                    del activation_scales, s
                    torch.cuda.empty_cache()
                    
                    # --- 修改：计算低秩因子 ---
                    print(f"  Step 4: Computing low-rank factors...")
                    compensation_rank = getattr(args, 'compensation_rank', None)
                    
                    if compensation_rank is None:
                        compensation_rank = min(delta_B_smoothed.shape[0], delta_B_smoothed.shape[1]) // 4
                    
                    L1, L2 = low_rank_approximation_factors(delta_B_smoothed, rank=compensation_rank)

                    # --- 保存参数而非应用 ---
                    layer_key = f"model.layers.{i}.{name}" # 使用标准模块名作为键
                    compensation_params[layer_key] = {
                        'L1': L1.cpu(),      # 移到CPU以节省GPU内存
                        'L2': L2.cpu(),
                        's_inv': s_inv.cpu()
                    }
                    print(f"    Compensation parameters for {layer_key} have been generated and stored.")
                    
                    # 清理GPU内存
                    del L1, L2, s_inv, delta_B_smoothed
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"    Error in smoothing/compensation: {e}")
                    print(f"    Skipping compensation for this layer.")                       
            else:
                print(f"    No compensation needed (delta_B norm is too small).")

            #线性层的权重被设置为纯粹的、稀疏的结构化剪枝结果
            subset[name].weight.data = B_structured
            print(f"    Layer weight is set to the sparse structured matrix.")

        # 重新计算层输出
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    
    print("\n=== Wanda compensation parameter generation completed ===")
    return compensation_params


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