#!/bin/bash

# Stage 1: Representation Learning
echo "=========================================="
echo "Starting Stage 1: Representation Learning"
echo "=========================================="

cd stage1_representation

# 训练 base 架构
echo "Training base architecture..."
python train.py --exp_type mae --arch_type base --data_name CSI

# 训练 MHA_RoPE_MoE 架构
echo "Training MHA_RoPE_MoE architecture..."
python train.py --exp_type mae --arch_type MHA_RoPE_MoE --data_name CSI

# 训练 MQA_RoPE_MoE 架构
echo "Training MQA_RoPE_MoE architecture..."
python train.py --exp_type mae --arch_type MQA_RoPE_MoE --data_name CSI

# 训练 GQA_RoPE_MoE 架构
echo "Training GQA_RoPE_MoE architecture..."
python train.py --exp_type mae --arch_type GQA_RoPE_MoE --data_name CSI

cd ..

# Stage 2: Policy Learning
echo "=========================================="
echo "Starting Stage 2: Policy Learning"
echo "=========================================="

cd stage2_policy

# 训练 base
echo "Training base architecture..."
python train.py --arch_type base

# 训练 MHA_RoPE_MoE
echo "Training MHA_RoPE_MoE architecture..."
python train.py --arch_type MHA_RoPE_MoE

# 训练 GQA_RoPE_MoE
echo "Training GQA_RoPE_MoE architecture..."
python train.py --arch_type GQA_RoPE_MoE

# 训练 MQA_RoPE_MoE
echo "Training MQA_RoPE_MoE architecture..."
python train.py --arch_type MQA_RoPE_MoE

cd ..

echo "=========================================="
echo "All training completed!"
echo "=========================================="
