import copy

import torch
from einops import rearrange
from thop import profile
from torch import nn
import torch.nn.functional as F
import math

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        # 深度卷积（不改变通道数）
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels  # 关键参数：groups=in_channels
        )
        # 逐点卷积（1x1卷积调整通道数）
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # 可选：添加BatchNorm和激活函数
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class globalinfograb(nn.Module):
    def __init__(self,channel,size=[32,32]):
        super(globalinfograb, self).__init__()
        self.size = size
        self.global_mod = nn.Sequential(nn.Linear(size[0]*size[1],size[0]*size[1]),
                                        nn.LayerNorm(size[0]*size[1]),
                                        nn.GELU())
        self.local_mod = nn.Sequential(nn.Conv2d(size[0]*size[1],size[0]*size[1],kernel_size=3,padding=3//2,groups=max(1,size[0]*size[1]//8)),
                                        nn.BatchNorm2d(size[0]*size[1]),
                                        nn.GELU())
        self.conv = DepthwiseSeparableConv(2*channel,channel,3,padding=3//2)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ori_size = x.shape[2:]
        H,W = ori_size
        pad_h,pad_w = self.size[0] - H%self.size[0],self.size[1] - W%self.size[1]
        rdn_h,rdn_w = pad_h%2,pad_w%2
        padding = (pad_w//2, pad_w//2+rdn_w, pad_h//2, pad_h//2+rdn_h)  # 左右各填充1，上下各填充1
        # 应用填充
        padded_x = F.pad(x, padding, mode='constant', value=0)
        pad_size = padded_x.shape[2:]

        x2 = padded_x.reshape(-1,self.size[0],pad_size[0]//self.size[0],self.size[1],pad_size[1]//self.size[1]).permute(0,1,3,2,4)
        x2 = x2.reshape(-1,self.size[0]*self.size[1],(pad_size[0]//self.size[0]),(pad_size[1]//self.size[1]))

        x2 = self.local_mod(x2)
        x2 = x2.reshape(-1,self.size[0],self.size[1],(pad_size[0]//self.size[0]),(pad_size[1]//self.size[1])).permute(0,1,3,2,4).reshape(padded_x.shape[0],padded_x.shape[1],padded_x.shape[2],padded_x.shape[3])
        x2 = x2[:,:,pad_h//2:pad_size[0]-(pad_h//2+rdn_h),pad_w//2:pad_size[1]-(pad_w//2+rdn_w)]


        x1 = F.interpolate(x, self.size, mode='bilinear', align_corners=True)
        b,c,h,w = x1.shape
        x1 = x1.reshape(b,c,h*w)
        x1 = self.global_mod(x1)
        x1 = x1.reshape(b,c,h,w)
        x1 = F.interpolate(x1, ori_size, mode='bilinear', align_corners=True)
        x1 = x1+x2
        x = torch.cat([x,x1],dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicSenseModule(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super(BasicSenseModule, self).__init__()
        #self.conv1 = BasicConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,padding=kernel_size // 2)
        self.conv1 = DepthwiseSeparableConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride,padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(1,kernel_size * 2 + 1),
                               stride=stride, padding=(0,(kernel_size * 2 + 1) // 2), groups=out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=(kernel_size * 2 + 1,1),
                               stride=stride, padding=((kernel_size * 2 + 1) // 2,0), groups=out_planes)

    def forward(self, x):
        x1 = self.conv1(x)

        # 优化后的变换操作
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        x = (x2 + x3) / 2
        return x


if __name__=='__main__':
    #测试模型大小
    device = "cuda"
    model = BasicSenseModule(3,32,3,stride=1).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    test_input = torch.randn([1, 3, 128, 128]).float().cuda()  # 输入尺寸需匹配模型

    # 计算FLOPs和参数量
    flops, _ = profile(copy.deepcopy(model), inputs=(test_input,))
    gflops = flops / 1e9  # 转换为GFLOPs
    print(f"FLOPs: {flops}")
    print(f"GFLOPs: {gflops:.2f}")