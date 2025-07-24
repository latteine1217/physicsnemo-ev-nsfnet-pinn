# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig
from physicsnemo.constants import tf_dt
from physicsnemo.models.module import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.io import ValidateInput

# P100 GPU 相容性設置
if torch.cuda.is_available():
    device_capability = torch.cuda.get_device_capability(0)
    major, minor = device_capability
    cuda_capability = major + minor * 0.1
    
    if cuda_capability < 7.0:  # P100 是 6.0
        # 設置環境變數以避免 Triton 編譯器錯誤
        os.environ.setdefault('TORCH_COMPILE_BACKEND', 'eager')
        os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
        
        # 抑制 torch.compile 相關錯誤
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
        except ImportError:
            pass

from physicsnemo_net import CombinedPhysicsNeMoNet
from physicsnemo_equations import NavierStokesEVM, EVMConstraint
from physicsnemo_data import CavityDataset


@ValidateInput(tf_dt)
class PhysicsNeMoPINNSolver(Module):
    """
    PhysicsNeMo-based PINN solver for cavity flow with eddy viscosity modeling
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing model parameters
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        
        # Initialize distributed manager
        self.dist = DistributedManager()
        
        # Model configuration
        self.reynolds_number = cfg.reynolds_number
        self.alpha_evm = cfg.alpha_evm
        self.alpha_boundary = cfg.alpha_boundary
        self.alpha_equation = cfg.alpha_equation
        self.alpha_evm_constraint = cfg.alpha_evm_constraint
        
        # Initialize networks
        self._init_networks()
        
        # Initialize equations
        self._init_equations()
        
        # Initialize dataset
        self._init_dataset()
        
        # Training parameters
        self.current_stage = ""
        self.vis_t0 = 20.0 / self.reynolds_number
        self.vis_t_minus = None
        
    def _init_networks(self):
        """Initialize neural networks"""
        
        main_net_config = {
            "input_keys": ["x", "y"],
            "output_keys": ["u", "v", "p"],
            "nr_layers": self.cfg.main_net.nr_layers,
            "layer_size": self.cfg.main_net.layer_size,
            "activation_fn": self.cfg.main_net.activation_fn,
        }
        
        evm_net_config = {
            "input_keys": ["x", "y"],
            "output_keys": ["evm"],
            "nr_layers": self.cfg.evm_net.nr_layers,
            "layer_size": self.cfg.evm_net.layer_size,
            "activation_fn": self.cfg.evm_net.activation_fn,
        }
        
        self.model = CombinedPhysicsNeMoNet(
            main_net_config=main_net_config,
            evm_net_config=evm_net_config
        )
        
        # Move to appropriate device
        self.model = self.model.to(self.dist.device)
        
        # Wrap with distributed training if needed
        if self.dist.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
            )
            
    def _init_equations(self):
        """Initialize PDE equations"""
        
        self.navier_stokes = NavierStokesEVM(
            nu=1.0/self.reynolds_number,
            rho=1.0,
            dim=2
        )
        
        self.evm_constraint = EVMConstraint(
            alpha_evm=self.alpha_evm,
            dim=2
        )
        
        # Move to device
        self.navier_stokes = self.navier_stokes.to(self.dist.device)
        self.evm_constraint = self.evm_constraint.to(self.dist.device)
        
    def _init_dataset(self):
        """Initialize dataset"""
        
        self.dataset = CavityDataset(
            data_dir=self.cfg.data_dir,
            num_samples=self.cfg.num_interior_points,
            num_boundary_samples=self.cfg.num_boundary_points,
            reynolds_number=self.reynolds_number,
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        return self.model(batch)
        
    def loss(self, invar: Dict[str, torch.Tensor], pred_outvar: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss
        
        Parameters
        ----------
        invar : Dict[str, torch.Tensor]
            Input variables
        pred_outvar : Dict[str, torch.Tensor] 
            Predicted output variables
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Loss dictionary
        """
        
        losses = {}
        
        # Boundary loss
        if "boundary_u" in invar and "boundary_v" in invar:
            boundary_loss_u = torch.mean((pred_outvar["u"] - invar["boundary_u"]) ** 2)
            boundary_loss_v = torch.mean((pred_outvar["v"] - invar["boundary_v"]) ** 2)
            losses["boundary_loss"] = boundary_loss_u + boundary_loss_v
        
        # PDE residual losses
        if "x" in invar and "y" in invar:
            # Compute derivatives and PDE residuals
            residuals = self._compute_pde_residuals(invar, pred_outvar)
            
            # Navier-Stokes residuals
            losses["continuity_loss"] = torch.mean(residuals["continuity"] ** 2)
            losses["momentum_x_loss"] = torch.mean(residuals["momentum_x"] ** 2) 
            losses["momentum_y_loss"] = torch.mean(residuals["momentum_y"] ** 2)
            
            # EVM constraint loss
            losses["evm_constraint_loss"] = torch.mean(residuals["evm_constraint"] ** 2)
        
        # Total loss
        total_loss = (
            self.alpha_boundary * losses.get("boundary_loss", 0.0) +
            self.alpha_equation * (
                losses.get("continuity_loss", 0.0) +
                losses.get("momentum_x_loss", 0.0) +
                losses.get("momentum_y_loss", 0.0)
            ) +
            self.alpha_evm_constraint * losses.get("evm_constraint_loss", 0.0)
        )
        
        losses["total_loss"] = total_loss
        
        return losses
        
    def _compute_pde_residuals(
        self, 
        invar: Dict[str, torch.Tensor], 
        pred_outvar: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute PDE residuals using automatic differentiation"""
        
        # Prepare input for equation modules
        input_dict = {
            "x": invar["x"],
            "y": invar["y"], 
            "u": pred_outvar["u"],
            "v": pred_outvar["v"],
            "p": pred_outvar["p"],
            "evm": pred_outvar["evm"]
        }
        
        # Compute Navier-Stokes residuals
        ns_residuals = self.navier_stokes(input_dict)
        
        # Compute EVM constraint residual
        evm_residuals = self.evm_constraint(input_dict)
        
        # Combine all residuals
        residuals = {**ns_residuals, **evm_residuals}
        
        return residuals
        
    def set_alpha_evm(self, alpha: float):
        """Set alpha_evm parameter"""
        self.alpha_evm = alpha
        
    def freeze_evm_net(self):
        """Freeze EVM network parameters"""
        if hasattr(self.model, 'module'):  # DDP wrapped
            for param in self.model.module.evm_net.parameters():
                param.requires_grad = False
        else:
            for param in self.model.evm_net.parameters():
                param.requires_grad = False
                
    def unfreeze_evm_net(self):
        """Unfreeze EVM network parameters"""
        if hasattr(self.model, 'module'):  # DDP wrapped
            for param in self.model.module.evm_net.parameters():
                param.requires_grad = True
        else:
            for param in self.model.evm_net.parameters():
                param.requires_grad = True
                
    def evaluate(self, x: torch.Tensor, y: torch.Tensor, u_ref: torch.Tensor, 
                 v_ref: torch.Tensor, p_ref: torch.Tensor) -> Dict[str, float]:
        """Evaluate model against reference data"""
        
        with torch.no_grad():
            input_dict = {"x": x, "y": y}
            pred_dict = self.model(input_dict)
            
            u_pred = pred_dict["u"]
            v_pred = pred_dict["v"]
            p_pred = pred_dict["p"]
            
            # Compute relative L2 errors
            error_u = torch.norm(u_ref - u_pred) / torch.norm(u_ref) * 100
            error_v = torch.norm(v_ref - v_pred) / torch.norm(v_ref) * 100
            
            # Handle pressure with NaN masking
            mask_p = ~torch.isnan(p_ref)
            if mask_p.any():
                error_p = torch.norm(p_ref[mask_p] - p_pred[mask_p]) / torch.norm(p_ref[mask_p]) * 100
            else:
                error_p = torch.tensor(float('nan'))
                
        return {
            "error_u": error_u.item(),
            "error_v": error_v.item(), 
            "error_p": error_p.item() if not torch.isnan(error_p) else 0.0
        }