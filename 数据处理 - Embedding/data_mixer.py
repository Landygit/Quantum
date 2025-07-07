'''mix_npy_data(
    npy_file_list=[...],       # 多个 .npy 文件路径
    output_path="mixed.npy",   # 输出文件路径
    samples_per_file=(1, 3),   # 每轮随机选几个样本
    total_output_rows=1000,    # 最终目标样本数
    seed=42                    # 可复现性控制（可选）
)
'''
"""mixed_file = mix_npy_data(
    ["/path/to/a.npy", "/path/to/b.npy", "/path/to/c.npy"],
    output_path="/path/to/mixed_output.npy",
    samples_per_file=(1, 3),
    total_output_rows=1000,
    seed=42
)
"""



import numpy as np
import os
import random

'''def mix_npy_data(npy_file_list, output_path, samples_per_file=(1, 3), total_output_rows=20000, seed=None):
    """
    从多个 .npy 文件中随机抽取部分行，混合生成一个新的数据集文件。

    参数：
    - npy_file_list: List[str]，包含多个 .npy 文件的路径
    - output_path: str，生成混合数据的新 .npy 文件保存路径
    - samples_per_file: Tuple[int, int]，从每个文件中随机抽取的行数范围（闭区间）
    - total_output_rows: int，总共希望生成的样本数（超出则截断，不足则随机补充）
    - seed: int，可选，设置随机种子以保证可复现

    返回：
    - 保存的输出文件路径
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_samples = []

    while len(all_samples) < total_output_rows:
        file = random.choice(npy_file_list)
        arr = np.load(file)  # shape: (N, L)
        N = arr.shape[0]

        # 从当前文件中抽取 k 行
        k = random.randint(*samples_per_file)
        indices = np.random.choice(N, size=min(k, N), replace=False)
        sampled = arr[indices]
        all_samples.append(sampled)

    # 拼接所有抽取的行，并裁剪至总数限制
    mixed_array = np.concatenate(all_samples, axis=0)[:total_output_rows]

    np.save(output_path, mixed_array)
    return output_path

# 示例：使用模拟路径（请替换为真实文件路径）
example_files = ["/Users/guwenlan/Desktop/DURF/Gernerated_/input_for_2_qubits_mixed_10000_datapoints.npy", "/Users/guwenlan/Desktop/DURF/Gernerated_/input_for_3_qubits_mixed_10000_datapoints.npy"]
example_output = "/Users/guwenlan/Desktop/DURF/Mixed_data_2_3.npy"
# mix_npy_data(example_files, example_output)  # 示例调用，需实际文件路径
'''

import numpy as np
import random

def mix_npy_data(npy_file_list, output_path, samples_per_file=(1, 3), seed=None):
    """
    从多个 .npy 文件中轮换抽取 1–3 行，直至所有文件的行都被使用完毕，生成混合数据集。

    参数：
    - npy_file_list: List[str]，所有要读取的 .npy 文件路径
    - output_path: str，保存合并输出的 .npy 文件路径
    - samples_per_file: Tuple[int, int]，每次从某文件中随机抽取的行数范围（闭区间）
    - seed: int，可选，设定随机种子以保证可复现性

    返回：
    - output_path: 保存成功后的路径
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 加载每个文件的数据和初始化未抽取行索引
    file_data = []  # [(data_array, remaining_indices_list), ...]
    for file in npy_file_list:
        arr = np.load(file)  # shape: (N, L)
        indices = list(np.random.permutation(len(arr)))  # 打乱行索引
        file_data.append((arr, indices))

    all_samples = []
    file_count = len(file_data)
    finished = [False] * file_count  # 每个文件是否处理完毕的标志
    active_files = file_count  # 当前仍有剩余行的文件数
    pointer = 0  # 当前轮到哪个文件抽取

    while active_files > 0:
        arr, indices = file_data[pointer]

        if not finished[pointer]:
            # 当前文件仍有剩余行
            k = random.randint(*samples_per_file)
            k = min(k, len(indices))  # 如果不足 k 行，取剩下所有行
            chosen = indices[:k]
            indices[:] = indices[k:]  # 从剩余列表中移除已使用的行
            file_data[pointer] = (arr, indices)  # 更新对应文件的剩余数据

            # 加入最终混合数据中
            all_samples.append(arr[chosen])

            # 如果该文件已抽完，标记为完成
            if len(indices) == 0:
                finished[pointer] = True
                active_files -= 1

        # 移动到下一个文件（循环轮换）
        pointer = (pointer + 1) % file_count

    # 合并所有批次并保存
    mixed_array = np.concatenate(all_samples, axis=0)
    np.save(output_path, mixed_array)
    return output_path

