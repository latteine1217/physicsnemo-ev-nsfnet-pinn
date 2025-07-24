#!/bin/bash
#SBATCH --job-name=PhysicsNeMo_PINN
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --gres=gpu:2

# 載入必要模組
ml load mpi

# 環境變數設定
export OMP_NUM_THREADS=24
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1

# Tesla P100 相容性設置 (CUDA capability 6.0)
export TORCH_COMPILE_BACKEND=eager
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# 分散式訓練設定
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE=2

# 啟動 Python 虛擬環境
source ~/python/bin/activate

# 建立必要目錄
mkdir -p checkpoints
mkdir -p outputs
mkdir -p logs

echo "=== PhysicsNeMo PINN Training Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================"

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits
echo "======================="

echo "=== Environment Check ==="
python -c "import physicsnemo; print(f'PhysicsNeMo version: {physicsnemo.__version__}')" || {
    echo "PhysicsNeMo not found. Installing..."
    pip install nvidia-physicsnemo
}
echo "========================="

echo "=== Starting PhysicsNeMo PINN Training ==="
time torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        physicsnemo_train.py

echo "=== Training Completed ==="
echo "End time: $(date)"
echo "Total GPU hours: $(echo "scale=2; 2 * $(date +%s - $SLURM_JOB_START_TIME) / 3600" | bc)"
echo "=========================="