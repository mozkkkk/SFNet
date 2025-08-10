import os
import numpy as np
from PIL import Image


def get_label_classes(directory):
    """
    统计给定目录中所有PNG分割标签的类别范围

    参数:
        directory (str): 包含PNG标签文件的目录路径

    返回:
        set: 所有出现过的唯一类别值集合
    """
    unique_classes = set()

    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath)
                img_array = np.array(img)
                unique_classes.update(np.unique(img_array))
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    return unique_classes


def decrease_label_values(directory, output_dir=None):
    """
    将给定目录中所有PNG分割标签的类别值减1并保存

    参数:
        directory (str): 包含PNG标签文件的目录路径
        output_dir (str, optional): 输出目录路径。默认为None（覆盖原文件）

    说明:
        - 0值保持不变（避免产生负值）
        - 当output_dir为None时覆盖原文件
        - 输出目录不存在时会自动创建
    """
    if output_dir is None:
        output_dir = directory
    else:
        os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                img = Image.open(input_path)
                img_array = np.array(img)

                # 创建掩码：仅处理大于0的像素
                mask = img_array > 0
                img_array[mask] -= 1

                # 保存处理后的图像
                result_img = Image.fromarray(img_array)
                result_img.save(output_path)
                print(f"处理成功: {filename} -> {output_path}")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


# 使用示例
if __name__ == "__main__":
    # 示例1: 统计类别范围
    label_dir = r"D:\river\river\EncodeSegmentationClass"
    classes = get_label_classes(label_dir)
    print("出现的类别值:", sorted(classes))

    # 示例2: 类别值减1操作 (覆盖原文件)
    #decrease_label_values(label_dir,"../test/")

    #classes = get_label_classes(label_dir)
    #print("出现的类别值:", sorted(classes))

    # 示例3: 保存到新目录
    # decrease_label_values(label_dir, "/path/to/output_directory")