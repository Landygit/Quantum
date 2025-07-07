import numpy as np
import os

def generate_embedding_2_1(npy_file_path, save_path=None):
    """
    生成 Embedding 2.1：区分 diagonal (1) 和 off-diagonal (0) 的标记。

    参数：
    - npy_file_path: str，原始 .npy 文件路径，文件中每行是一个向量化的密度矩阵
    - save_path: str，可选，保存 embedding 的路径，若为 None 则自动命名为 <原文件名>_emb2.1.npy

    输出：
    - 保存一个 shape = (N, L, 1) 的 embedding 文件，其中：
        - N 是样本数（向量行数）
        - L 是向量长度（等于 D^2）
        - 每个位置的值为：
            - 1：对角项
            - 0：非对角项
    """
    # 加载原始向量数据，shape = (N, L)
    rho = np.load(npy_file_path)
    N, L = rho.shape

    # 计算矩阵维度 D：向量长度应为 D²
    D = int(np.sqrt(L))
    assert D * D == L, f"向量长度 {L} 不能开方成整数，可能不是 D×D 的向量化矩阵。"

    # 初始化全 0（默认是 off-diagonal）
    embedding = np.zeros((N, L, 1), dtype=np.int8)

    # 将前 D 个元素标记为 diagonal
    embedding[:, :D, 0] = 1

    # 自动保存路径处理
    if save_path is None:
        base, ext = os.path.splitext(npy_file_path)
        save_path = f"{base}_emb2.1.npy"

    np.save(save_path, embedding)
    return save_path
