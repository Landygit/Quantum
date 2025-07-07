#!/bin/bash

echo "📁 创建目录..."
mkdir -p Input_X Output_Y CNN_X CNN_Y log_all

LOG_DIR="log_all"

echo "🧪 Step 1: 生成密度矩阵向量输入..."
if [ ! -f "$LOG_DIR/creating_the_input_file.log" ]; then
    python creating_the_input_file.py 2>&1 | tee "$LOG_DIR/creating_the_input_file.log"
else
    echo "⏩ 已存在 creating_the_input_file.log，跳过生成。"
fi

echo "✨ Step 2: 计算 magic labels..."
if [ ! -f "$LOG_DIR/call_sta_label.log" ]; then
    python call_sta_label.py 2>&1 | tee "$LOG_DIR/call_sta_label.log"
else
    echo "⏩ 已存在 call_sta_label.log，跳过计算。"
fi

echo "🌀 Step 3: 转换为 CNN 输入格式..."
if [ ! -f "$LOG_DIR/convert_to_cnn_batch.log" ]; then
    python convert_to_cnn_batch.py 2>&1 | tee "$LOG_DIR/convert_to_cnn_batch.log"
else
    echo "⏩ 已存在 convert_to_cnn_batch.log，跳过转换。"
fi

echo "✅ 全部处理完成！"
