#!/bin/bash
# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
# PhysicsNeMo EV-NSFnet PINN Simple Training Script (Direct execution)
# Developed by: opencode + GitHub Copilot
# Target: Dell R740 server with dual P100 GPUs

# Environment setup for dual P100 GPUs
export OMP_NUM_THREADS=24
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1

# Additional P100 optimizations
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Create necessary directories
mkdir -p checkpoints_simple
mkdir -p outputs_simple
mkdir -p data

echo "=========================================="
echo "PhysicsNeMo EV-NSFnet PINN Simple Training"
echo "=========================================="
echo "Developed by: opencode + GitHub Copilot"
echo "Target: Dell R740 server with dual P100 GPUs"
echo "=========================================="

# Display GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "----------------------------------------"

echo "Job start: $(date)"

# Run distributed training with torchrun
time torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
        physicsnemo_train_simple.py --config-path=conf --config-name=config_simple

echo "Job end: $(date)"

# Display final results
echo "----------------------------------------"
echo "Training completed!"
echo "Check outputs_simple/ for results"
echo "Check checkpoints_simple/ for saved models"

# Show final checkpoint if available
if [ -f "checkpoints_simple/model_simple_test_final.pth" ]; then
    echo "Final model saved: checkpoints_simple/model_simple_test_final.pth"
    ls -lh checkpoints_simple/model_simple_test_final.pth
fi

echo "=========================================="