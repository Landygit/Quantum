import os
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
