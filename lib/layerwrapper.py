import torch
import torch.nn as nn

class WrappedGPT:
    """
    此类用于包装一个GPT层，以便于执行特定的操作，
    例如捕获和处理激活值，为剪枝和量化平滑等算法做准备。
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        """
        初始化方法
        Args:
            layer (nn.Module): 要包装的神经网络层 (例如 nn.Linear)。
            layer_id (int): 层的ID。
            layer_name (str): 层的名称。
        """
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        # --- 为 Smoothing 初始化 ---
        # 用于存储每个输入通道在所有校准样本中遇到过的最大绝对值
        self.act_scales = None

        # --- 为 Wanda 剪枝初始化 ---
        # 用于累积计算激活值L2范数的平方，以计算Wanda剪枝度量
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        """
        处理一个批次的输入和输出，以收集所需的统计信息。
        借鉴了 smoothquant.calibration.py 的实现风格。
        """
        if isinstance(inp, tuple):
            inp = inp[0]
        
        # 记录当前批次的样本数 (用于 Wanda 的流式计算)
        # inp 初始形状通常是 (batch, seq_len, hidden_size)
        current_nsamples = inp.shape[0]

        # 1. 为 Smoothing 计算逐通道最大绝对值 (参考 smoothquant.calibration.py)
        # ----------------------------------------------------------------------
        # 将输入统一为二维张量 (num_tokens, hidden_dim)
        inp_2d = inp.view(-1, inp.shape[-1])
        # 分离计算图，并取绝对值
        inp_2d_abs = inp_2d.abs().detach()
        # 计算当前批次中每个通道的最大值
        comming_max = torch.max(inp_2d_abs, dim=0)[0].float().cpu()

        # 更新全局最大值记录
        if self.act_scales is None:
            self.act_scales = comming_max.clone()
        else:
            self.act_scales = torch.max(self.act_scales, comming_max)

        # 2. 为 Wanda 剪枝计算激活值的L2范数平方 (保留原逻辑)
        # ----------------------------------------------------------------------
        # a. 按比例缩放旧的累加和
        self.scaler_row *= self.nsamples / (self.nsamples + current_nsamples)
        # b. 更新总样本数
        self.nsamples += current_nsamples
        # c. 将新批次的贡献加入
        #    Wanda 的原始实现需要转置为 (hidden_size, num_tokens)
        inp_for_wanda = inp_2d.t()
        self.scaler_row += torch.norm(inp_for_wanda.type(torch.float32), p=2, dim=1) ** 2 / self.nsamples