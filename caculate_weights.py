import os
import numpy as np
from PIL import Image
from collections import defaultdict
import pickle


def calculate_dataset_stats(label_folder, img_folder, num_classes,start=0):
    """
    计算类别权重、图像均值和标准差

    参数:
        label_folder (str): 存放标签PNG的文件夹路径
        img_folder (str): 存放原始图像的文件夹路径
        num_classes (int): 类别总数（0到n-1）

    返回:
        stats (dict): 包含classWeights、mean、std的字典
    """
    # 初始化统计变量
    class_counts = defaultdict(int)
    total_pixels = 0
    pixel_sum = np.zeros(3)  # 用于计算mean (RGB三通道)
    pixel_sq_sum = np.zeros(3)  # 用于计算std
    num_images = 0

    # 遍历所有图像和标签文件
    for filename in os.listdir(label_folder):
        if filename.endswith('.png'):
            # 处理标签图像
            label_path = os.path.join(label_folder, filename)
            label = np.array(Image.open(label_path))

            # 统计类别像素数
            unique, counts = np.unique(label, return_counts=True)
            for cls, cnt in zip(unique, counts):
                if cls < num_classes:
                    class_counts[cls] += cnt
            total_pixels += label.size

            # 处理对应的原始图像
            img_path = os.path.join(img_folder, filename)
            img = np.array(Image.open(img_path).convert('RGB')) / 255.0  # 归一化到[0,1]

            # 更新均值和方差统计量
            pixel_sum += img.sum(axis=(0, 1))
            pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
            num_images += 1

    # 计算类别权重（逆频率）
    class_weights = np.zeros(num_classes)
    for cls in range(num_classes):
        if class_counts[cls] == 0:
            class_weights[cls] = 0
        else:
            class_weights[cls] = total_pixels / (num_classes * class_counts[cls])
    class_weights = class_weights / class_weights.sum()  # 归一化

    # 计算均值和标准差
    mean = pixel_sum / (num_images * label.shape[0] * label.shape[1])
    std = np.sqrt(pixel_sq_sum / (num_images * label.shape[0] * label.shape[1]) - mean ** 2)

    return {
        'classWeights': class_weights[start:],
        'mean': mean,
        'std': std
    }


# 示例用法
label_folder = r"D:\river\river\EncodeSegmentationClass"  # 标签文件夹
img_folder = r"D:\river\river\JPEGImages"  # 原始图像文件夹
num_classes = 4  # 类别数

stats = calculate_dataset_stats(label_folder, img_folder, num_classes,0)
print("Class Weights:", stats['classWeights'])
print("Mean (RGB):", stats['mean'])
print("Std (RGB):", stats['std'])

# 保存到pkl文件
with open("dataset/inform/river_inform.pkl", "wb") as f:
    pickle.dump(stats, f)