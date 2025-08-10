import os


def save_filenames_to_txt(directory_path, output_file):
    """
    将指定目录下的所有文件名保存到文本文件中，每个文件名占一行

    参数:
        directory_path (str): 要遍历的目录路径
        output_file (str): 输出文本文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in os.listdir(directory_path):
            # 获取完整路径
            full_path = os.path.join(directory_path, filename)
            # 只处理文件，不包含子目录
            if os.path.isfile(full_path):
                f.write(filename + '\n')


# 使用示例
directory_path = r'D:\river\NWPU_YRCC_EX\test'  # 替换为你的目录路径
output_file = 'river_test_list.txt'  # 输出文件名
save_filenames_to_txt(directory_path, output_file)