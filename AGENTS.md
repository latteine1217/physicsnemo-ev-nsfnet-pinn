# AGENTS.md - PhysicsNeMo EV-NSFnet PINN Project Guidelines

## Project Overview
- **EV-NSFnet**: Entropy Viscosity Navier-Stokes Fourier Network using PINNs + EVM
- **Problem**: Lid-driven cavity flow at Re=5000 with dual neural networks
- **Framework**: NVIDIA PhysicsNeMo for distributed training and optimization
- **Developed by**: opencode + GitHub Copilot

## Build/Test Commands
- **Training**: `python physicsnemo_train.py` or `./run_training.sh [num_gpus]`
- **Testing**: `python physicsnemo_test.py`
- **Single test**: Use `pytest physicsnemo_test.py::test_function_name` for specific tests
- **Linting**: `black .` and `isort .` (optional dev tools)
- **Install deps**: `pip install -r requirements.txt`

## Code Style & Conventions
- **Copyright**: Always include `# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.` at top
- **Imports**: Standard lib, third-party (torch, numpy), then local imports
- **Type hints**: Use `from typing import List, Optional, Dict` for complex types
- **Classes**: PascalCase (e.g., `PhysicsNeMoPINNSolver`)
- **Functions/vars**: snake_case (e.g., `training_step`, `boundary_data`)
- **Config**: Use Hydra/OmegaConf for configuration management
- **Docstrings**: Brief description for public methods
- **Error handling**: Use try/except with logger.error() for failures
- **Device handling**: Use `solver.dist.device` for GPU/CPU placement
- **Distributed**: Check `solver.dist.rank == 0` before logging/saving
- **EVM parameters**: Use alpha_evm for entropy viscosity regularization weight

## Project Structure
- Core solver: `physicsnemo_solver.py` (dual-network PINN solver)
- Neural networks: `physicsnemo_net.py` (main + EVM networks)
- PDE equations: `physicsnemo_equations.py` (Navier-Stokes + EVM constraint)
- Data handling: `physicsnemo_data.py` (cavity flow with boundary conditions)
- Main training: `physicsnemo_train.py` (6-stage progressive training)
- Testing: `physicsnemo_test.py` (multi-Reynolds validation)

## Rules
- Do not push to GitHub automatically
- README.md should note development by opencode + GitHub Copilot
- Use multi-stage training with decreasing alpha_evm values
- Maintain dual-network architecture for main flow + eddy viscosity