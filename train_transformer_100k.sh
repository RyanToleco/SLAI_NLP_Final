#!/bin/bash

# ==========================================
# Transformer模型训练脚本
# 使用 train_100k.jsonl 数据集，训练100个epoch
# ==========================================

set -e  # 遇到错误立即退出

# 脚本目录
SCRIPT_DIR="/data/250010006/workspace/NLP_Final"
cd "$SCRIPT_DIR"

# 定义PID文件路径
PID_DIR="/tmp/nlpfinal_pids"
mkdir -p "$PID_DIR"

# 清理函数 - 在脚本退出时清理后台进程
cleanup() {
    echo "[INFO] 正在清理后台进程..."
    
    # 停止mihomo代理
    if [ -f "$PID_DIR/mihomo.pid" ]; then
        MIHOMO_PID=$(cat "$PID_DIR/mihomo.pid")
        if kill -0 "$MIHOMO_PID" 2>/dev/null; then
            echo "[INFO] 停止mihomo代理 (PID: $MIHOMO_PID)"
            kill "$MIHOMO_PID" 2>/dev/null || true
        fi
        rm -f "$PID_DIR/mihomo.pid"
    fi
    
    echo "[INFO] 清理完成"
}

# 设置退出时清理
trap cleanup EXIT INT TERM

# 激活conda环境
echo "[INFO] 激活conda环境: nlpfinal ..."
source /data/250010006/miniconda3/bin/activate nlpfinal

# 检查环境是否激活成功
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "[ERROR] Conda环境激活失败！"
    exit 1
fi
echo "[INFO] 当前conda环境: $CONDA_DEFAULT_ENV"

# 启动mihomo代理 (在后台运行)
echo "[INFO] 启动mihomo代理..."
cd ~
nohup ./mihomo -f config.yaml > /tmp/mihomo_nlpfinal.log 2>&1 &
MIHOMO_PID=$!
echo $MIHOMO_PID > "$PID_DIR/mihomo.pid"
echo "[INFO] mihomo代理已启动 (PID: $MIHOMO_PID)"

# 等待mihomo代理启动
echo "[INFO] 等待mihomo代理启动..."
sleep 10

# 设置代理环境变量
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
echo "[INFO] 已设置HTTP和HTTPS代理: http://127.0.0.1:7890"

# 返回脚本目录
cd "$SCRIPT_DIR"

# 检查数据文件是否存在
DATA_FILE="dataset/train_100k.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] 数据文件不存在: $DATA_FILE"
    exit 1
fi
echo "[INFO] 使用数据集: $DATA_FILE"

# 训练参数
MODEL_TYPE="t5"
EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=0.0001
D_MODEL=512
N_HEADS=8
N_LAYERS=6
MAX_LEN=50
DECODE_METHOD="greedy"  # 可选: greedy 或 beam
BEAM_WIDTH=5
EVAL_SAMPLES=200

# 显示训练配置
echo "=========================================="
echo "训练配置"
echo "=========================================="
echo "模型类型: $MODEL_TYPE"
echo "数据集: $DATA_FILE"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "模型维度: $D_MODEL"
echo "注意力头数: $N_HEADS"
echo "层数: $N_LAYERS"
echo "最大长度: $MAX_LEN"
echo "解码方法: $DECODE_METHOD"
if [ "$DECODE_METHOD" == "beam" ]; then
    echo "Beam宽度: $BEAM_WIDTH"
fi
echo "评估样本数: $EVAL_SAMPLES"
echo "=========================================="
echo ""

# 开始训练
echo "[INFO] 开始训练 Transformer 模型 ... $(date)"
echo ""

python inference.py \
    --model $MODEL_TYPE \
    --data train_100k.jsonl \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_layers $N_LAYERS \
    --max_len $MAX_LEN \
    --decode_method $DECODE_METHOD \
    --beam_width $BEAM_WIDTH \
    --eval_samples $EVAL_SAMPLES \
    --norm_type layernorm \
    --pos_type absolute


# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "[INFO] 训练完成！ $(date)"
    echo "[INFO] 结果保存在 outputs/ 目录下"
else
    echo ""
    echo "[ERROR] 训练过程中出现错误！"
    exit 1
fi

