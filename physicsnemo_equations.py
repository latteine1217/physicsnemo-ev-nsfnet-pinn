# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from physicsnemo.utils.io import ValidateInput
from physicsnemo.constants import tf_dt


@ValidateInput(tf_dt)
class NavierStokesEVM(nn.Module):
    """
    Navier-Stokes equations with Eddy Viscosity Model for PhysicsNeMo
    
    This implementation uses automatic differentiation to compute PDE residuals
    for the incompressible Navier-Stokes equations with eddy viscosity modeling.
    """
    
    def __init__(self, nu: float = 0.01, rho: float = 1.0, dim: int = 2):
        super().__init__()
        self.nu = nu
        self.rho = rho
        self.dim = dim
        
    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute Navier-Stokes PDE residuals with EVM
        
        Args:
            input_dict: Dictionary containing coordinates and predictions
                - 'x', 'y': spatial coordinates
                - 'u', 'v': velocity components  
                - 'p': pressure
                - 'evm': eddy viscosity
                
        Returns:
            Dictionary of PDE residuals
        """
        x = input_dict["x"]
        y = input_dict["y"]
        u = input_dict["u"]
        v = input_dict["v"] 
        p = input_dict["p"]
        evm = input_dict["evm"]
        
        # Enable gradients for automatic differentiation
        x.requires_grad_(True)
        y.requires_grad_(True)
        u.requires_grad_(True)
        v.requires_grad_(True)
        p.requires_grad_(True)
        evm.requires_grad_(True)
        
        # Compute first derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        # Compute second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        
        # Continuity equation
        continuity = u_x + v_y
        
        # x-momentum equation with eddy viscosity
        momentum_x = (
            u * u_x + v * u_y +  # convection
            (1/self.rho) * p_x -  # pressure gradient
            (self.nu + evm) * (u_xx + u_yy)  # viscous terms
        )
        
        # y-momentum equation with eddy viscosity  
        momentum_y = (
            u * v_x + v * v_y +  # convection
            (1/self.rho) * p_y -  # pressure gradient
            (self.nu + evm) * (v_xx + v_yy)  # viscous terms
        )
        
        return {
            "continuity": continuity,
            "momentum_x": momentum_x,
            "momentum_y": momentum_y
        }


@ValidateInput(tf_dt)
class EVMConstraint(nn.Module):
    """
    Eddy Viscosity Model constraint equation for PhysicsNeMo
    
    This implements the entropy viscosity constraint that relates
    the eddy viscosity to local flow residuals.
    """
    
    def __init__(self, alpha_evm: float = 0.03, dim: int = 2):
        super().__init__()
        self.alpha_evm = alpha_evm
        self.dim = dim
        
    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute EVM constraint residual
        
        Args:
            input_dict: Dictionary containing coordinates and predictions
                - 'x', 'y': spatial coordinates
                - 'u', 'v': velocity components
                - 'evm': eddy viscosity
                
        Returns:
            Dictionary containing EVM constraint residual
        """
        x = input_dict["x"]
        y = input_dict["y"] 
        u = input_dict["u"]
        v = input_dict["v"]
        evm = input_dict["evm"]
        
        # Enable gradients
        x.requires_grad_(True)
        y.requires_grad_(True)
        u.requires_grad_(True)
        v.requires_grad_(True)
        
        # Compute velocity derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        # Compute residual-based eddy viscosity constraint
        # This relates eddy viscosity to convective terms and velocity magnitude
        residual_term = (u - 0.5) * (u * u_x + v * u_y) + (v - 0.5) * (u * v_x + v * v_y)
        
        # EVM constraint: residual term should equal eddy viscosity scaled by alpha
        evm_constraint = residual_term - self.alpha_evm * evm
        
        return {
            "evm_constraint": evm_constraint
        }