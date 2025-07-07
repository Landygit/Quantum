#!/bin/bash

echo "📁 创建目录..."
mkdir -p Input_X Output_Y CNN_X CNN_Y log_all

LOG_DIR="log_all"

echo "✨ Step 1: 生成 5-qubit 输入密度矩阵..."
python3 creating_the_input_file.py 2>&1 | tee "$LOG_DIR/creating_the_input_file(5).log"

echo "✨ Step 2: 计算对应的 magic labels..."
python3 trail.py 2>&1 | tee "$LOG_DIR/call_sta_label(5).log"

# echo "🌀 Step 3: 转换为 CNN 输入格式..."
# python3 convert_to_cnn_batch.py 2>&1 | tee "$LOG_DIR/convert_to_cnn_batch.log"

echo "✅ 全部处理完成！"
