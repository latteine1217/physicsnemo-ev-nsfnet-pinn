# PhysicsNeMo EV-NSFnet PINN - Simple Test Version

## Overview
This is a simplified test version of the PhysicsNeMo EV-NSFnet PINN project, specifically optimized for testing on a Dell R740 server with dual P100 GPUs.

**Developed by**: opencode + GitHub Copilot

## Key Simplifications for Testing

### Reduced Complexity
- **Training epochs**: 2,000 (vs 3,000,000 in full version)
- **Network size**: Smaller networks (4 layers, 60 nodes vs 6 layers, 80 nodes)
- **Data points**: 50,000 interior points (vs 120,000)
- **Single stage**: No multi-stage progressive training

### P100 GPU Optimizations
- **Mixed precision**: Enabled to utilize Tensor Cores
- **Memory management**: Optimized for 16GB P100 memory
- **Gradient accumulation**: Simulates larger batch sizes
- **NCCL backend**: Optimized for dual-GPU communication

## Quick Start

### 1. Server Setup (Dell R740 with dual P100)
```bash
# Verify GPU setup
nvidia-smi

# Should show 2x Tesla P100-PCIE-16GB
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Simple Training

**Option A: Using SLURM (if available)**
```bash
# Submit job to SLURM queue
sbatch run_simple_training.sh
```

**Option B: Direct execution**
```bash
# Run directly on the server
./train_simple.sh
```

## Scripts Available

### SLURM Version: `run_simple_training.sh`
- Follows the same format as `~/Documents/coding/ldc_pinns/NSFnet/ev-NSFnet/train.sh`
- Uses SBATCH directives for r740 partition
- 2-hour time limit for testing
- Automatic job output logging

### Direct Version: `train_simple.sh`
- For servers without SLURM
- Same torchrun command structure
- Immediate execution

## Files Created

### New Files for Simple Testing
- `physicsnemo_train_simple.py` - Simplified training script
- `conf/config_simple.yaml` - Optimized configuration for P100s
- `run_simple_training.sh` - Easy-to-use training launcher
- `README_SIMPLE.md` - This documentation

### Key Differences from Full Version
1. **Single training stage** instead of 6-stage progressive training
2. **Smaller networks** for faster convergence testing
3. **Reduced data size** for memory efficiency
4. **More frequent logging** (every 50 epochs vs 100)
5. **P100-specific optimizations** for 16GB memory

## Expected Runtime
- **Dual P100**: ~30-45 minutes for 2,000 epochs
- **Single P100**: ~60-90 minutes for 2,000 epochs

## Output Files
- **Checkpoints**: `checkpoints_simple/`
- **Logs**: `outputs_simple/`
- **Final model**: `checkpoints_simple/model_simple_test_final.pth`

## Monitoring Training
The training will log every 50 epochs showing:
- Total loss convergence
- Boundary condition satisfaction
- PDE residual terms
- EVM constraint compliance

## Success Criteria
✅ Training completes without CUDA OOM errors  
✅ Loss decreases consistently  
✅ Both GPUs are utilized (check with `nvidia-smi`)  
✅ Final model checkpoint is saved  

## Next Steps
If this simple test runs successfully:
1. Verify loss convergence patterns
2. Check GPU utilization efficiency
3. Scale up to full training version
4. Implement multi-Reynolds testing

## Troubleshooting

### Common Issues
- **CUDA OOM**: Reduce `num_interior_points` in config_simple.yaml
- **NCCL errors**: Check network connectivity between GPUs
- **Slow training**: Verify both GPUs are being used with `nvidia-smi`

### P100-Specific Notes
- P100s have 16GB memory each
- Pascal architecture (older than V100/A100)
- No native FP16 Tensor Cores (mixed precision still helps)
- Optimized for scientific computing workloads

---
**Note**: This is a test version. For production runs, use the full `physicsnemo_train.py` with complete 6-stage training.