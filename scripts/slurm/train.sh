#!/usr/bin/env bash
#SBATCH --job-name=ldc-pinns-physicsnemo
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Dell R740 (2x Intel Xeon Gold 5118, 112GB RAM, 2x Tesla P100 16GB)

module purge
module load mpi

export OMP_NUM_THREADS=24
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# P100 compatibility
export TORCH_COMPILE_BACKEND=eager
export TORCHDYNAMO_DISABLE=1

# 兩張GPU以 torchrun 啟動
CONFIG=${CONFIG:-configs/ldc_pinn_advanced.yaml}

echo "=== Launch training with PhysicsNeMo (DDP x2) ==="
echo "Config: ${CONFIG}"

torchrun --nproc_per_node=2 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29501 \
  train_physicsnemo_advanced.py --config ${CONFIG}

