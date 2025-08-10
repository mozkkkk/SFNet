
import torch.nn.functional as F


import torch
import torch.nn as nn



# 更稳定的完全向量化版本
class Sq_SCAN(nn.Module):
    """
    最简洁的螺旋扫描卷积模块
    避免复杂的螺旋计算，使用更直接的方法
    """

    def __init__(self, channels: int, down_ratio: int = 2, extra_kernel=4):
        super(Sq_SCAN, self).__init__()
        self.channels = channels
        self.down_ratio = down_ratio
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=down_ratio*down_ratio+extra_kernel,
            stride=down_ratio*down_ratio,
            padding=extra_kernel//2
        )
        self._spiral_cache = {}

    def _generate_spiral_indices_simple(self, h: int, w: int, device: torch.device):
        """修正的螺旋索引生成，按方形圈顺时针顺序排列"""
        cache_key = (h, w, str(device))
        if cache_key in self._spiral_cache:
            return self._spiral_cache[cache_key]

        # 计算整数中心坐标
        center_y = int(round((h - 1) / 2.0))
        center_x = int(round((w - 1) / 2.0))

        # 计算最大圈数
        max_r = max(center_y, center_x, h - 1 - center_y, w - 1 - center_x)

        # 存储螺旋顺序的坐标
        spiral_coords = []

        # 单独处理中心点
        spiral_coords.append((center_y, center_x))

        # 从内到外遍历每一圈
        for r in range(1, max_r + 1):
            # 上边：从左到右 (包括两个角)
            y_top = center_y - r
            if y_top >= 0:
                for x in range(center_x - r, center_x + r + 1):
                    if 0 <= x < w:
                        spiral_coords.append((y_top, x))

            # 右边：从上到下 (排除右上角，包括右下角)
            x_right = center_x + r
            if x_right < w:
                for y in range(center_y - r + 1, center_y + r + 1):
                    if 0 <= y < h:
                        spiral_coords.append((y, x_right))

            # 下边：从右到左 (排除右下角，包括左下角)
            y_bottom = center_y + r
            if y_bottom < h:
                for x in range(center_x + r - 1, center_x - r - 1, -1):
                    if 0 <= x < w:
                        spiral_coords.append((y_bottom, x))

            # 左边：从下到上 (排除左下角和左上角)
            x_left = center_x - r
            if x_left >= 0:
                for y in range(center_y + r - 1, center_y - r, -1):
                    if 0 <= y < h:
                        spiral_coords.append((y, x_left))

        # 转换为Tensor
        indices_tensor = torch.tensor(spiral_coords, device=device, dtype=torch.long)

        self._spiral_cache[cache_key] = indices_tensor
        return indices_tensor

    def spiral_to_image(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        将螺旋顺序的张量恢复为原始图像形状

        参数:
            x: 输入张量，形状为 (b, c, len)，其中 len = h * w
            h: 原始图像高度
            w: 原始图像宽度

        返回:
            恢复后的张量，形状为 (b, c, h, w)
        """
        b, c, total_len = x.shape
        if total_len != h * w:
            x = F.interpolate(x.unsqueeze(2), size=(1,h * w), mode='bilinear', align_corners=True).squeeze(2)

        # 获取螺旋索引
        indices = self._generate_spiral_indices_simple(h, w, x.device)  # (h*w, 2)

        # 创建空白图像张量
        output = torch.zeros(b, c, h, w, device=x.device, dtype=x.dtype)

        # 提取坐标索引
        y_indices = indices[:, 0]  # 形状 (h*w,)
        x_indices = indices[:, 1]  # 形状 (h*w,)

        # 使用高级索引将值放回正确位置
        # 这里使用:表示保留批次和通道维度
        output[:, :, y_indices, x_indices] = x

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """简化的前向传播"""
        b, c, h, w = x.shape
        device = x.device

        # 获取螺旋坐标
        spiral_coords = self._generate_spiral_indices_simple(h, w, device)

        # 按螺旋顺序提取数据
        spiral_data = x[:, :, spiral_coords[:, 0], spiral_coords[:, 1]]  # [b, c, h*w]

        #spiral_data_flip = torch.flip(spiral_data,dims=[2])
        # 应用卷积
        conv_output1 = self.conv1(spiral_data)  # [b, c, 1, output_len]
        #conv_output2 = torch.flip(self.conv1(spiral_data_flip),dims=[2])
        conv_output = conv_output1
        #conv_output = spiral_data
        # 计算输出尺寸
        out_h, out_w = x.shape[2], x.shape[3]

        # 恢复
        output = self.spiral_to_image(conv_output,out_h, out_w)

        output = F.sigmoid(output) * x

        return output
# 使用示例和测试
if __name__ == "__main__":
    # 创建测试数据
    x = torch.arange(1, 26).reshape(1, 1, 5, 5).float()
    t = torch.tensor([1.0],requires_grad=True).float()
    x = x*t
    print(x)

    # 测试优化版本
    print("Testing OptimizedSq_SCAN...")
    model1 = Sq_SCAN(channels=1, down_ratio=2)

    output1 = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Output: {output1}")


