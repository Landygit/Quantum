import numpy as np
from tqdm import tqdm
import time
import os
import math

# importing your functions here, make sure the files are in the same location

from random_mix import random_mixed_state, validate_rho
from upper_tri import rho_to_vec

# generates a single line for the input file
# - random matrix -> data row
# - by dafault mixes between 1-10 pure states for the mixed state
# - note that density matrix for n qubits will have 2^n x 2^n size 

def generate_input_line(n_qubits, n_mixed=None):
    notvaild = True
    while notvaild:
        dim = 2**n_qubits
        if not n_mixed:
            n_mixed = np.random.randint(1,n_qubits+1)
        rho = random_mixed_state(dim, n_mixed)
        validation = validate_rho(rho)
        if validation == True:
            notvaild = False
            return rho_to_vec(rho)
        else:
            continue

v_generate_input_line = np.vectorize(generate_input_line, signature='(),()->(d)')

def create_input_file(n_arrays, n_qubits, n_mixed=None, filename=None, output_dir="."):
    if not filename:
        if n_mixed == 1:
            filename = f'input_for_{n_qubits}_qubits_pure_{n_arrays}_datapoints.npy'
        else:
            filename = f'input_for_{n_qubits}_qubits_mixed_{n_arrays}_datapoints.npy'
    start_time = time.time()
    data = []
    for _ in tqdm(range(n_arrays), desc=f"Generating {n_qubits}-qubit samples"):
        data.append(generate_input_line(n_qubits, n_mixed))
    data = np.array(data)
    output_path = os.path.join(output_dir, filename)
    np.save(output_path, data)
    elapsed = time.time() - start_time
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Saved as {filename}")
    print(f"File size {math.ceil(os.path.getsize(output_path)/1000)} KB")
    print(f"Elapsed time: {int(hrs):02}:{int(mins):02}:{secs:05.2f}")

def chech_input_file_shape(filename):
    map = np.lib.format.open_memmap(filename, mode='r+')
    print(f"{map.shape[0]} rows")
    print(f"{map.shape[1]} cols")

def read_input_file_line(filename, i_line):
    map = np.lib.format.open_memmap(filename, mode='r+')
    return np.array(map[i_line])

log_dir = "log_all"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "creating_the_input_file.log")

with open(log_path, "a") as log_file:
    for num_qubits in [5]:
        for size in [50000, 60000, 70000, 80000, 100000, 200000]:
            for index in range(1, 4):
                filename = f"input_for_{num_qubits}_qubits_mixed_{index}_{size}_datapoints.npy"
                filepath = f"Input_X/{filename}"
                try:
                    with open(filepath, 'rb') as f:
                        msg = f"⏩ 已存在 {filename}，跳过。"
                        print(msg)
                        log_file.write(msg + "\n")
                        continue
                except FileNotFoundError:
                    pass

                msg = f"🚀 正在生成 {filename}..."
                print(msg)
                log_file.write(msg + "\n")
                create_input_file(size, num_qubits, filename=filename, output_dir='Input_X')
