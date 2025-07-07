import os
import numpy as np
from tqdm import tqdm
from stabilizer_norm_function import get_sn, vec_to_rho

# 输入输出文件夹
input_dir = "Input_X"
output_dir = "Output_Y"
os.makedirs(output_dir, exist_ok=True)

# 限定只处理 5 qubit 的输入文件
for fname in sorted(os.listdir(input_dir)):
    if not fname.startswith("input_for_5_qubits_mixed_"):
        continue
    if not fname.endswith(".npy"):
        continue

    input_path = os.path.join(input_dir, fname)
    output_fname = f"magic_labels_for_{fname}"
    output_path = os.path.join(output_dir, output_fname)

    # 跳过已有输出文件
    if os.path.exists(output_path):
        print(f"\033[93m⏩ 已存在 magic label，跳过: {output_fname}\033[0m")
        continue

    print(f"\n🔍 正在处理: {fname}")
    try:
        data = np.load(input_path)
    except Exception as e:
        print(f"\033[91m❌ 加载失败 {fname}: {e}\033[0m")
        continue

    # 执行 magic label 计算
    results = []
    for item in tqdm(data, desc=f"Computing magic for {fname}"):
        results.append(get_sn(item))  # 用你自己的逻辑替换

    results = np.array(results)
    np.save(output_path, results)
    print(f"\033[92m✅ 已保存: {output_fname}\033[0m")
'''import os
import numpy as np
from stabilizer_norm_function import get_sn, vec_to_rho
  # assuming you move the function here or import properly
from tqdm import tqdm

input_folder = "Input_X"  # or wherever you store your .npy files
input_prefix = "input_for_"  # pattern

input_files = [f for f in os.listdir(input_folder) if f.startswith(input_prefix) and f.endswith(".npy")]
output_folder = "Output_Y"  # where you want to save the magic labels
os.makedirs(output_folder, exist_ok=True)
for input_filename in input_files:
    print(f"🔍 Processing: {input_filename}")
    X = np.load(os.path.join(input_folder, input_filename))
    magic_values = []

    for vec in tqdm(X, desc=f"Computing magic for {input_filename}"):
        rho = vec_to_rho(vec)
        magic = get_sn(rho)
        magic_values.append(magic)

    magic_values = np.array(magic_values)

    output_filename = f"magic_labels_for_{input_filename}"
    np.save(os.path.join(output_folder, output_filename), magic_values)
    print(f"✅ Saved: {output_filename}")





'''