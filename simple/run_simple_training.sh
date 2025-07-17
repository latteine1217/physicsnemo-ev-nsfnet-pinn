#!/bin/bash
#SBATCH --job-name=PhysicsNeMo_simple
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=108G
#SBATCH --gres=gpu:2

# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
# PhysicsNeMo EV-NSFnet PINN Simple Training Script
# Developed by: opencode + GitHub Copilot
# Target: Dell R740 server with dual P100 GPUs

# Load required modules
ml load mpi

# Environment setup for dual P100 GPUs
export OMP_NUM_THREADS=24
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1

# Additional P100 optimizations
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Activate Python environment (adjust path as needed)
source ~/python/bin/activate

# Create necessary directories
mkdir -p checkpoints_simple
mkdir -p outputs_simple
mkdir -p data

echo "=========================================="
echo "PhysicsNeMo EV-NSFnet PINN Simple Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Developed by: opencode + GitHub Copilot"
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