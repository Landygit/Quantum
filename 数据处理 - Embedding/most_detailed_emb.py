import numpy as np
import os

def generate_embedding_2_3_encoded(npy_file_path, save_path=None):
    """
    生成 Embedding 2.3 的编码版本：将每个向量化密度矩阵位置标记为唯一整数 index。
    index = i * D + j + offset，其中 offset 依赖于类型 (diag/re/im)

    编码方式：
    - Diagonal: i == j, type = 2   → offset = 0
    - Off-diagonal Re: i < j, type = 1 → offset = D * D
    - Off-diagonal Im: i < j, type = 0 → offset = D * D + D*(D-1)//2

    参数：
    - npy_file_path: str，原始 .npy 文件路径，文件中每行是一个向量化的密度矩阵
    - save_path: str，可选，保存 embedding 的路径，若为 None 则自动命名为 <原文件名>_emb2.3_encoded.npy

    输出：
    - 保存一个 shape = (N, L, 1) 的 embedding 文件，其中每项是唯一编码后的整数 index
    """
    rho = np.load(npy_file_path)  # shape: (N, L)
    N, L = rho.shape

    D = int(np.sqrt(L))
    assert D * D == L, f"向量长度 {L} 无法开方为整数，可能不是 D×D 的向量化密度矩阵"

    embedding = np.zeros((N, L, 1), dtype=np.int32)

    # ========== Diagonal ==========
    # 向量前 D 项是对角线 rho[i, i]
    for i in range(D):
        index = i * D + i  # (i, i)
        embedding[:, i, 0] = index  # no offset

    # ========== Off-diagonal Re/Im ==========
    offset_re = D * D
    offset_im = D * D + D * (D - 1) // 2

    pos = D  # 从这里开始是 off-diagonal
    for i in range(D):
        for j in range(i + 1, D):
            index_flat = i * D + j  # (i, j)
            embedding[:, pos, 0] = offset_re + index_flat
            pos += 1
            embedding[:, pos, 0] = offset_im + index_flat
            pos += 1

    if save_path is None:
        base, ext = os.path.splitext(npy_file_path)
        save_path = f"{base}_emb2.3_encoded.npy"

    np.save(save_path, embedding)
    return save_path

# 示例调用：
# generate_embedding_2_3_encoded("your_density_matrix_vectors.npy")
