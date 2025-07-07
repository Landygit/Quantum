import numpy as np
from tqdm import tqdm
from vector_to_hermitian import vec_to_rho
import os
'''| 功能               | 说明                                         |
| ---------------- | ------------------------------------------ |
| ✅ 自动推理 `D`       | 从 vector 长度自动还原矩阵大小 `D x D`，适应 1\~5 qubits |
| ✅ 自动匹配标签文件       | 只需输入目录，自动找配对的 label 文件（按文件名匹配）             |
| ✅ 支持输出子文件夹       | CNN 格式 `.npy` 文件写到 `cnn_X/`, `cnn_y/` 子目录  |
| ✅ 保留 tqdm + 错误处理 | 处理过程中可视化进度 + 错误提示不崩溃                       |
'''




def vector_to_tensor2d(vec: np.ndarray) -> np.ndarray:
    """将向量还原成密度矩阵，并输出 [2, 4, 4] 张量 (real, imag)"""
    mat = vec_to_rho(vec)
    real = np.real(mat)
    imag = np.imag(mat)
    return np.stack([real, imag], axis=0)

def infer_matrix_dim_from_vector(vec: np.ndarray) -> int:
    D = int(np.sqrt(len(vec)))
    if D * D != len(vec):
        raise ValueError(f"向量长度 {len(vec)} 不是完美平方，不能还原成方阵。")
    return D


def convert_one_file_pair(vec_path: str, label_path: str, out_x_path: str, out_y_path: str):
    vectors = np.load(vec_path)           # [N, d]
    labels = np.load(label_path)          # [N,]
    assert vectors.shape[0] == labels.shape[0], "样本数不一致"

    N = vectors.shape[0]
    D = infer_matrix_dim_from_vector(vectors[0])  # 推理出方阵维度
    print(f"🔄 正在处理 {vec_path} (共 {N} 个样本, D={D})")

    x_all = np.zeros((N, 2, D, D), dtype=np.float32)

    for i in tqdm(range(N), desc=os.path.basename(vec_path)):
        x_all[i] = vector_to_tensor2d(vectors[i])

    # 保存结果
    np.save(out_x_path, x_all)
    np.save(out_y_path, labels.astype(np.float32))
    print(f"✅ 输出至: {out_x_path}, {out_y_path}")



def batch_convert_all(input_dir: str, label_dir: str, out_x_dir: str, out_y_dir: str):

    os.makedirs(out_x_dir, exist_ok=True)
    os.makedirs(out_y_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.startswith("input_for_") and f.endswith(".npy")]

    for input_file in input_files:
        label_file = f"magic_labels_for_{input_file}"
        input_path = os.path.join(input_dir, input_file)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"⚠️ 跳过：未找到标签文件 {label_file}")
            continue

        # 输出文件名规则
        base_name = input_file.replace(".npy", "")
        out_x_path = os.path.join(out_x_dir, f"X_{base_name}.npy")
        out_y_path = os.path.join(out_y_dir, f"y_{base_name}.npy")

        try:
            convert_one_file_pair(input_path, label_path, out_x_path, out_y_path)
        except Exception as e:
            print(f"❌ 错误处理文件 {input_file}: {e}")

batch_convert_all(
        input_dir="Input_X",
        label_dir="Output_Y",
        out_x_dir="CNN_X",
        out_y_dir="CNN_Y"
)




'''# 示例调用
if __name__ == "__main__":
    process_batch(
        input_path="/Users/guwenlan/Desktop/DURF/Gernerated_/input_for_2_qubits_mixed_1000000_datapoints.npy",
        label_path="/Users/guwenlan/Desktop/DURF/Gernerated_/magic_labels_for_input_for_2_qubits_mixed_1000000_datapoints.npy",
        output_x="X_2qubit.npy",
        output_y="y_2qubit.npy"
    )
'''
