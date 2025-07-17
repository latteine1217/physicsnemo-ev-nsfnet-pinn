# 🌊 PhysicsNeMo EV-NSFnet PINN Project

> **Developed by**: opencode + GitHub Copilot 🤖

This project implements an **Entropy Viscosity Navier-Stokes Fourier Network (EV-NSFnet)** using Physics-Informed Neural Networks (PINNs) with NVIDIA PhysicsNeMo for distributed training and optimization.

---

## 📋 Overview

| Component | Description |
|-----------|-------------|
| **🎯 Problem** | Lid-driven cavity flow at Re=5000 with dual neural networks |
| **⚙️ Method** | PINNs + Entropy Viscosity Method (EVM) for numerical stability |
| **🚀 Framework** | NVIDIA PhysicsNeMo for GPU acceleration and distributed training |
| **🏗️ Architecture** | Dual-network system (main flow + eddy viscosity prediction) |

## ⭐ Key Features

### 🧠 EV-NSFnet Implementation
- **🔄 Dual Neural Networks**: 
  - 🎯 **Main network**: Predicts velocity (u,v) and pressure (p)
  - 🌀 **EVM network**: Predicts eddy viscosity for high Reynolds number stability
- **📈 6-Stage Progressive Training**: Gradually reduces alpha_evm from 0.05 to 0.002
- **❄️ Adaptive EVM Freezing**: Alternates between frozen/unfrozen EVM network training

### ⚡ PhysicsNeMo Integration
- **🔧 Optimized Neural Networks**: Uses PhysicsNeMo's FullyConnected layers
- **🖥️ Distributed Training**: Multi-GPU support with DistributedManager
- **🧮 Automatic Differentiation**: Efficient PDE residual computation
- **📊 Professional Logging**: Comprehensive logging and checkpointing

## 📦 Installation

### Step 1: Install PhysicsNeMo
```bash
pip install nvidia-physicsnemo
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Create Required Directories
```bash
mkdir -p checkpoints outputs data
```

---

## 🚀 Usage

| Training Mode | Command | Description |
|---------------|---------|-------------|
| **🖥️ Single GPU** | `python physicsnemo_train.py` | Basic training |
| **⚡ Multi-GPU** | `./run_training.sh 4` | Distributed training on 4 GPUs |
| **🧪 Testing** | `python physicsnemo_test.py` | Model validation |
| **🔍 Single Test** | `pytest physicsnemo_test.py::test_function_name` | Specific test |

## 📁 Project Structure

```
📦 PhysicsNeMo EV-NSFnet PINN
├── 🧠 physicsnemo_solver.py      # Main PINN solver with dual networks
├── 🔗 physicsnemo_net.py         # Neural network architectures
├── 📐 physicsnemo_equations.py   # Navier-Stokes + EVM equations
├── 📊 physicsnemo_data.py        # Cavity flow dataset with boundary conditions
├── 🏃‍♂️ physicsnemo_train.py       # 6-stage progressive training script
├── 🧪 physicsnemo_test.py        # Multi-Reynolds validation script
├── ⚙️ conf/config.yaml          # Hydra configuration
├── 📋 requirements.txt          # Python dependencies
├── 🚀 run_training.sh          # Training execution script
└── 📖 AGENTS.md               # Development guidelines
```

## ⚙️ Configuration

> Key parameters in `conf/config.yaml`

| Parameter | Value | Description |
|-----------|-------|-------------|
| **🌊 reynolds_number** | 5000 | Target Reynolds number |
| **⚡ alpha_evm** | 0.03 | Entropy viscosity regularization weight |
| **🎯 alpha_boundary** | 10.0 | Boundary condition loss weight |
| **📐 alpha_equation** | 1.0 | PDE residual loss weight |

### 📈 Training Stages
The implementation uses **6 progressive training stages**:

| Stage | 🔧 alpha_evm | 📚 Learning Rate | ⏱️ Epochs |
|-------|-------------|-----------------|----------|
| **1️⃣** | 0.05 | 1e-3 | 500k |
| **2️⃣** | 0.03 | 2e-4 | 500k |
| **3️⃣** | 0.01 | 4e-5 | 500k |
| **4️⃣** | 0.005 | 1e-5 | 500k |
| **5️⃣** | 0.002 | 2e-6 | 500k |
| **6️⃣** | 0.002 | 2e-6 | 500k |

## 🔬 Technical Details

### 🌊 Lid-Driven Cavity Flow
| Aspect | Details |
|--------|---------|
| **📐 Domain** | [0,1] × [0,1] square cavity |
| **🔄 Boundary Conditions** | |
| - 🔝 Top wall | Moving lid: u=1-cosh(50(x-0.5))/cosh(25), v=0 |
| - 🏠 Other walls | No-slip: u=0, v=0 |
| **🌀 Reynolds Number** | 5000 (high Re requiring stabilization) |

### ⚡ Entropy Viscosity Method
- **🎯 Purpose**: Provides numerical stability for high Re flows
- **🧠 Implementation**: Additional neural network predicts local eddy viscosity
- **🔗 Constraint**: Links eddy viscosity to local flow residuals
- **📅 Training Schedule**: Alternating freeze/unfreeze cycles for EVM network

### 🔧 PhysicsNeMo Features Used
```python
physicsnemo.models.mlp.fully_connected.FullyConnected
physicsnemo.distributed.DistributedManager
physicsnemo.launch.logging and physicsnemo.launch.utils
physicsnemo.utils.io.ValidateInput
```

## 📊 Expected Performance

| Metric | Performance |
|--------|-------------|
| **⚡ Training Speedup** | 2-5x faster than standard PyTorch |
| **📈 Scaling** | Linear scaling across multiple GPUs |
| **🎯 Convergence** | Improved stability at high Reynolds numbers |
| **✅ Accuracy** | Target <2% error for velocity fields |

---

## 💡 Development Notes

This project demonstrates:
- 🔗 Integration of advanced PINN methods with modern ML frameworks
- 📈 Multi-stage training strategies for challenging fluid dynamics problems
- 🏢 Professional-grade distributed training and logging
- ⚡ Entropy viscosity stabilization for high Reynolds number flows

> 📖 For detailed development guidelines, see [AGENTS.md](AGENTS.md).

---

## 📚 Citation

If you use this code, please cite:
- 🔧 NVIDIA PhysicsNeMo framework
- 📄 Original EV-NSFnet methodology
- 🤖 Note development assistance from **opencode + GitHub Copilot**