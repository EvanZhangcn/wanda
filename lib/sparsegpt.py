import math
import time

import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

## SparseGPT: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        #dead表示拿出 H 中的零对角线元素（代表没有被激活过的神经元）
        dead = torch.diag(H) == 0
        #例如：H是一个(columns,columns)的权重矩阵，得到dead是(columns,)的一维向量
        H[dead, dead] = 1
        W[:, dead] = 0 #死亡神经元的权重置为0，其形状为 (输出特征数, 输入特征数)。W[:, j] 是与第 j 个输入特征相连的所有权重。

        Losses = torch.zeros(self.rows, device=self.dev)


        #计算海森矩阵的逆矩阵
        #添加一个阻尼项(damp)到 H 的对角线，以增加数值稳定性
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        #https://docs.pytorch.org/docs/stable/generated/torch.linalg.cholesky.html
        H = torch.linalg.cholesky(H) #得到下三角矩阵L
        H = torch.cholesky_inverse(H) #给定L， 计算出乔列斯基分解的最终结果： 这里已经拿到黑塞矩阵的逆了

        # 对黑塞矩阵的逆进行 Cholesky 分解，H_inv = U^T * U，其中 U 是上三角矩阵
        H = torch.linalg.cholesky(H, upper=True)
        #此时H为 U， Hinv是U
        Hinv = H

        mask = None

        #权重矩阵每blocksize个列分成一块，按块(block)进行迭代剪枝，默认blocksize为128，每128列为一块
        for i1 in range(0, self.columns, blocksize):
            #逐块处理
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            #W1是W的一个子矩阵，从i1到i2的列取出来作一块
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1) #标量性损失，记录剪枝操作造成了多大的“损失”或“损害”
            Losses1 = torch.zeros_like(W1) #补偿其他权重的“误差信号”
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                #非结构化剪枝
                if mask is not None: # 如果已有全局掩码 (一般不使用)
                    mask1 = mask[:, i1:i2]
                else:
                    # 计算重要性分数temp：(w^2) / (diag(H_inv)^2)，这是基于 OBS 误差公式得出的
                    # 我们的分母不用黑塞矩阵的逆的对角线上元素直接算，而是用U^2 来计算
                    # Hinv1是U的一部分，U的对角线元素的平方是 H⁻¹ 的对角线元素
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)] #按重要性排序后，根据tmp的元素个数和稀疏度计算阈值
                    mask1 = tmp <= thresh
            else:
                #结构化剪枝，先创建掩码矩阵mask1
                #这里创建一个全为False的掩码矩阵
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                #该块里面逐列处理
                w = W1[:, i]
                d = Hinv1[i, i]

                #结构化剪枝走这个if
                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2 #由分子形状，知道tmp此时形状为(2048, prune_m)
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0 #将掩码矩阵mask1中为True的元素对应的q置为0，完成对一列的剪枝置0

                Q1[:, i] = q #Q1 = torch.zeros_like(W1)
                Losses1[:, i] = (w - q) ** 2 / d ** 2 #剪枝后带来的损失， 分母d是黑塞矩阵逆的对角线元素


                err1 = (w - q) / d  #对应（除被剪枝那列外）其他列的权重需要补偿更新的值， w-q表示被剪掉的权重值，对应obs的w
                #debug查看err1，此时应该是一个列向量，形状为(2048, 1)，我们对第i列的权重进行剪枝，err1对应第i个元素补偿是0
                #所以下面这个W1尽管从第i列开始操作，但只是把对应第i列置0，而其他列的权重进行正常补偿
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
             #由上面每次更新Q1，Q1是i1：i2这块剪枝后的权重矩阵
            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2 #OBS误差公式，计算每一行的损失
            #同时对其他块的权重进行补偿更新
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        #Waits for all kernels in all streams on a CUDA device to complete.
        torch.cuda.synchronize()
        #若是Conv1D层，则需要转置W
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()