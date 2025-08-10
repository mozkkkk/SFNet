import copy
import math

import torch
from einops import rearrange
from thop import profile
from torch import nn
import torch.nn.functional as F

from network.SFNet_Moudle import BasicSenseModule, DepthwiseSeparableConv, globalinfograb

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.R_SCAN import Sq_SCAN
from network.SelfAttention import ImageSelfAttention


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


class SFNet(nn.Module):
    def __init__(self, num_class):
        super(SFNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3,16,3,1,padding=3//2),
            BasicSenseModule(16,32,7,1),
            Sq_SCAN(32,2,4),
            globalinfograb(32,[32,32]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=3 // 2),
            BasicSenseModule(32,64,5,1),
            Sq_SCAN(64,2,4),
            globalinfograb(64,[16,16]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.encoder3 = nn.Sequential(
            BasicSenseModule(64,128,3,1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.encoder4 = nn.Sequential(
            BasicSenseModule(128,256,3,1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.up_layer4 = BasicConv2d(256,128,3,1,1)
        self.up_layer3 = BasicConv2d(128,64,3,1,1)
        self.up_layer2 = BasicConv2d(64,32,3,1,1)
        self.up_layer1 = BasicConv2d(32,128,3,1,1)

        self.deocde1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.Conv2d(64, num_class, 3, 1, 1, bias=True))
        self.deocde2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.Conv2d(64, num_class, 3, 1, 1, bias=True))
        self.deocde3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, num_class, 3, 1, 1, bias=True))
        self.deocde4 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.Conv2d(64, num_class, 3, 1, 1, bias=True))

        self.attn1 = ImageSelfAttention(128,2)
        self.attn2 = ImageSelfAttention(64,2)
        #self.attn = nn.MultiheadAttention(embed_dim=128,num_heads=2,dropout=0.1)
    def forward(self,A):
        size = A.size()[2:]
        layer1 = self.encoder1(A)
        layer2 = self.encoder2(layer1)
        layer3 = self.encoder3(layer2)
        layer4 = self.encoder4(layer3)


        layer4 = self.up_layer4(layer4)
        layer4 = self.attn1(layer4)
        layer4_up = F.interpolate(layer4, layer3.shape[2:], mode='bilinear', align_corners=True)

        layer3 = layer3 + layer4_up
        layer3 = self.up_layer3(layer3)
        layer3 = self.attn2(layer3)
        layer3_up = F.interpolate(layer3, layer2.shape[2:], mode='bilinear', align_corners=True)

        layer2 = layer2 + layer3_up
        layer2 = self.up_layer2(layer2)
        layer2_up = F.interpolate(layer2, layer1.shape[2:], mode='bilinear', align_corners=True)

        layer1 = layer1 + layer2_up
        layer1 = self.up_layer1(layer1)
        # os = layer1.shape
        # layer1, pad = space_to_blocks(layer1,[8,8])
        # layer1,_ = self.attn(layer1,layer1,layer1)
        # layer1 = blocks_to_space(layer1,[8,8],os,pad)
        outputs = []
        seg_map = self.deocde1(layer1)

        seg_map = F.interpolate(seg_map.clone(), size, mode='bilinear', align_corners=True)

        outputs.append(F.interpolate(self.deocde2(layer2), size, mode='bilinear', align_corners=True))
        outputs.append(F.interpolate(self.deocde3(layer3), size, mode='bilinear', align_corners=True))
        outputs.append(F.interpolate(self.deocde4(layer4), size, mode='bilinear', align_corners=True))

        return outputs, seg_map



if __name__=='__main__':
    #测试热图
    # net=HCGMNet().cuda()
    # out=net(torch.rand((2,3,256,256)).cuda(),torch.rand((2,3,256,256)).cuda())

    #测试模型大小
    device = "cuda"
    model = SFNet(3).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    test_input1 = torch.randn(1, 3, 256, 256).float().cuda()  # 输入尺寸需匹配模型
    # 计算FLOPs和参数量
    flops, _ = profile(copy.deepcopy(model), inputs=(test_input1,))
    gflops = flops / 1e9  # 转换为GFLOPs
    print(f"FLOPs: {flops}")
    print(f"GFLOPs: {gflops:.2f}")