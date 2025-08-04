import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT(Generative Pretrained Transformer) layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0] #权重矩阵行数
        self.columns = layer.weight.data.shape[1]# 权重矩阵列数
        self.inp1 = None

        self.scaler_row = torch.zeros((self.columns), device=self.dev) #创建一个大小为(columns)的一维全0张量
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        #第一维是批次大小batch size
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3: #形状此时是(batch, seqlen, hidden_size)
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t() #形状此时是(hidden_size， batch*seqlen)
        
        # 只保存激活值的最大值用于缩放计算，而不是所有数据
        # 只保存激活值的最大值用于缩放计算，而不是所有数据
        if self.inp1 is None:
            self.inp1 = torch.max(torch.abs(inp), dim=1, keepdim=True)[0].cpu()
        else:
            current_max = torch.max(torch.abs(inp), dim=1, keepdim=True)[0].cpu()
            self.inp1 = torch.max(self.inp1, current_max)

        #新样本加入后，计算新样本所占全部样本的比例，计算加权平均
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        #更新样本总数
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        #最后/self.nsamples也是为了加权 贡献
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples