import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.in_channels = in_channels

        # 计算内部通道数（缩放的维度）
        self.inter_channels = max(in_channels // reduction_ratio, 1)

        # 1x1卷积层替代线性变换
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 缩放因子
        self.scale = self.inter_channels ** -0.5

        # 可学习的缩放参数（用于残差连接）
        self.gamma = nn.Parameter(torch.zeros(1))

        # 空间维度softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # 生成query和key特征图 [B, inter_C, H, W]
        query = self.query_conv(x)
        key = self.key_conv(x)

        # 重塑为 [B, inter_C, H*W]
        query = query.view(batch_size, self.inter_channels, -1)
        key = key.view(batch_size, self.inter_channels, -1)

        # 计算注意力图 [B, H*W, H*W]
        attention = torch.bmm(query.permute(0, 2, 1), key)  # [B, HW, HW]
        attention = attention * self.scale
        attention = self.softmax(attention)

        # 生成value [B, C, H*W]
        value = self.value_conv(x)
        value = value.view(batch_size, C, -1)

        # 应用注意力 [B, C, HW]
        out = torch.bmm(value, attention.permute(0, 2, 1))

        # 重塑为原始空间维度 [B, C, H, W]
        out = out.view(batch_size, C, H, W)

        # 残差连接
        out = self.gamma * out + x

        return out


# 测试示例
if __name__ == "__main__":
    # 创建输入: [batch, channels, height, width]
    x = torch.randn(4, 64, 32, 32)

    # 初始化自注意力模块
    attn = ImageSelfAttention(in_channels=64)

    # 前向传播
    out = attn(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Gamma value:", attn.gamma.item())