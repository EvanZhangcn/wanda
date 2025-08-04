# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    import os
    print(f"--- 尝试从镜像加载C4数据集，使用的 Endpoint 是: {os.getenv('HF_ENDPOINT')} ---")
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation')
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            #train enc是一个字典，包含了input_ids， attention_mask中1表示有效token有内容，0表示没有内容
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt') #pt表示返回pytorch张量格式

            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        #i范围是[0, trainenc.input_ids.shape[1] - seqlen - 1]， 由此知道j范围是[i, trainenc.input_ids.shape[1] - 1]
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        #tar除了最后一个token，其余的都被设置为-100，表示不计算损失
        tar[:, :-1] = -100
        #inp是模型输入， tar是监督信号，用于实现预测下一个token任务
        trainloader.append((inp, tar))

    # Prepare validation dataset
    #用空格将前1100个样本的文本连接起来，形成一个长文本，然后token化
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    #封装成一个自定义的 TokenizerWrapper 对象,就是为了方便后续访问input_ids
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    #nsamples取的样本数目，seqlen是序列长度，每个样本长度(学习到的特征维度数目)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)