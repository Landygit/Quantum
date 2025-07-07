import numpy as np  
def generate_embedding_2_3_triplet(npy_file_path, save_path=None):
    """
    生成 Embedding 2.3 的三元组版本：每个位置为 (i, j, type)
    - type: 2 = diagonal, 1 = Re(off-diag), 0 = Im(off-diag)

    参数：
    - npy_file_path: str，原始 .npy 文件路径，文件中每行是一个向量化的密度矩阵
    - save_path: str，可选，保存 embedding 的路径，若为 None 则自动命名为 <原文件名>_emb2.3_triplet.npy

    输出：
    - 保存一个 shape = (N, L, 3) 的 embedding 文件，每个位置是对应的 (i, j, type)
    """
    rho = np.load(npy_file_path)  # shape: (N, L)
    N, L = rho.shape

    D = int(np.sqrt(L))
    assert D * D == L, f"向量长度 {L} 无法开方为整数，可能不是 D×D 的向量化密度矩阵"

    embedding = np.zeros((N, L, 3), dtype=np.int16)

    # ========== Diagonal ==========
    for i in range(D):
        embedding[:, i, 0] = i     # i
        embedding[:, i, 1] = i     # j
        embedding[:, i, 2] = 2     # type = diagonal

    # ========== Off-diagonal Re/Im ==========
    pos = D
    for i in range(D):
        for j in range(i + 1, D):
            # Re
            embedding[:, pos, 0] = i
            embedding[:, pos, 1] = j
            embedding[:, pos, 2] = 1
            pos += 1
            # Im
            embedding[:, pos, 0] = i
            embedding[:, pos, 1] = j
            embedding[:, pos, 2] = 0
            pos += 1

    if save_path is None:
        base, ext = os.path.splitext(npy_file_path)
        save_path = f"{base}_emb2.3_triplet.npy"

    np.save(save_path, embedding)
    return save_path

# 示例调用：
# generate_embedding_2_3_triplet("your_density_matrix_vectors.npy")
