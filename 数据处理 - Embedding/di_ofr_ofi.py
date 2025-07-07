import numpy as np
import os

def generate_embedding_2_2(npy_file_path, save_path=None):
    """
    生成 Embedding 2.2：区分 diagonal (2), off-diagonal real part (1), off-diagonal imag part (0)

    参数：
    - npy_file_path: str，原始 .npy 文件路径，文件中每行是一个向量化的密度矩阵
    - save_path: str，可选，保存 embedding 的路径，若为 None 则自动命名为 <原文件名>_emb2.2.npy

    输出：
    - 保存一个 shape = (N, L, 1) 的 embedding 文件，其中：
        - 每个位置的值为：
            - 2：对角项
            - 1：off-diagonal 的实部
            - 0：off-diagonal 的虚部
    """
    rho = np.load(npy_file_path)  # shape: (N, L)
    N, L = rho.shape

    D = int(np.sqrt(L))
    assert D * D == L, f"向量长度 {L} 无法开方为整数，可能不是 D×D 的向量化密度矩阵"

    embedding = np.zeros((N, L, 1), dtype=np.int8)

    # Diagonal entries 在向量开头，共 D 项
    embedding[:, :D, 0] = 2

    # 剩下的位置是 off-diagonal 部分的实部 + 虚部，按对出现：
    # 对于 (i,j) 且 i < j，共 D(D-1)/2 对，每对包含 Re, Im 共 2 项
    # 即：embedding[:, D::2, 0] 是 Re，embedding[:, D+1::2, 0] 是 Im
    embedding[:, D::2, 0] = 1  # Re
    embedding[:, D+1::2, 0] = 0  # Im（默认值其实是 0，可省略，但写出来更清晰）

    # 保存路径
    if save_path is None:
        base, ext = os.path.splitext(npy_file_path)
        save_path = f"{base}_emb2.2.npy"

    np.save(save_path, embedding)
    return save_path

# 示例路径（请替换为真实文件）
# generate_embedding_2_2("/mnt/data/example_density_vectors.npy")
