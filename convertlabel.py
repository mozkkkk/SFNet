import os
from PIL import Image
import numpy as np


def convert_landcover_images(input_dir, output_dir):
    """
    根据特定颜色映射转换土地覆盖分割图
    颜色到类别映射:
        Land (0,0,0) -> 0
        Shore Ice (0,255,255) -> 1
        Water (0,255,0) -> 2
        Streaming (255,0,255) -> 3
    """
    # 定义颜色到类别的映射
    COLOR_MAPPING = {
        (0, 0, 0): 0,  # Land -> 0
        (0, 255, 255): 1,  # Shore Ice -> 1
        (0, 255, 0): 2,  # Water -> 2
        (255, 0, 255): 3  # Streaming -> 3
    }

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # 只处理图像文件
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        try:
            # 打开图像并转换为RGB（确保处理彩色图像）
            img = Image.open(input_path).convert('RGB')
            img_array = np.array(img)

            # 创建空的输出数组（单通道）
            output_array = np.zeros(img_array.shape[:2], dtype=np.uint8)

            # 根据特定颜色映射转换像素值
            output_array[(img_array == [0, 0, 0]).all(axis=2)] = 0  # Land
            output_array[(img_array == [0, 255, 255]).all(axis=2)] = 1  # Shore Ice
            output_array[(img_array == [0, 255, 0]).all(axis=2)] = 2  # Water
            output_array[(img_array == [255, 0, 255]).all(axis=2)] = 3  # Streaming

            # 检查是否有未映射的像素
            unmapped_pixels = np.logical_not(
                (output_array == 0) | (output_array == 1) |
                (output_array == 2) | (output_array == 3)
            )

            if np.any(unmapped_pixels):
                unique_colors = np.unique(img_array[unmapped_pixels], axis=0)
                print(f"警告: {filename} 中包含未映射的颜色: {unique_colors.tolist()}")

            # 创建输出图像并保存
            output_img = Image.fromarray(output_array)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
            output_img.save(output_path)

            print(f"已处理: {filename} -> 保存到 {output_path}")

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
            continue


# 使用示例
input_directory = r"D:\river\river\labels"  # 替换为你的输入文件夹路径
output_directory = r"D:\river\river\EncodeSegmentationClass"  # 替换为你的输出文件夹路径

convert_landcover_images(input_directory, output_directory)