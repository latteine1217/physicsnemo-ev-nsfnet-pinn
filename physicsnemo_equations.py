# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import torch
import torch.nn as nn
from typing import Dict
from physicsnemo.pdes import PDE
from physicsnemo.utils.derivatives import gradient


class NavierStokesEVMEquation(PDE):
    """
    PhysicsNeMo 納維-史托克斯方程式與熵黏性法 (EVM)
    
    實作不可壓縮納維-史托克斯方程式：
    ∂u/∂t + u∇u = -∇p + (1/Re + νₜ)∇²u
    ∇·u = 0
    
    其中 νₜ 是 EVM 網路預測的渦黏度
    """
    
    def __init__(self, reynolds_number: float = 5000, alpha_evm: float = 0.03):
        super().__init__()
        
        self.reynolds_number = reynolds_number
        self.alpha_evm = alpha_evm
        
        # 計算黏度係數
        self.nu = 1.0 / reynolds_number
        
    def forward(
        self, 
        input_vars: Dict[str, torch.Tensor], 
        model_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """計算納維-史托克斯方程式殘差"""
        
        x = input_vars["x"]
        y = input_vars["y"]
        
        u = model_output["u"]
        v = model_output["v"] 
        p = model_output["p"]
        nu_t = model_output["nu_t"]  # EVM 預測的渦黏度
        
        # 使用 PhysicsNeMo 的 gradient 函數
        u_x = gradient(u, x)
        u_y = gradient(u, y)
        u_xx = gradient(u_x, x)
        u_yy = gradient(u_y, y)
        
        v_x = gradient(v, x)
        v_y = gradient(v, y)
        v_xx = gradient(v_x, x)
        v_yy = gradient(v_y, y)
        
        p_x = gradient(p, x)
        p_y = gradient(p, y)
        
        # 計算渦黏度導數
        nu_t_x = gradient(nu_t, x)
        nu_t_y = gradient(nu_t, y)
        
        # 有效黏度 = 分子黏度 + 渦黏度
        nu_eff = self.nu + self.alpha_evm * nu_t
        
        # x 方向動量方程式 (穩態)
        momentum_x = (
            u * u_x + v * u_y + p_x
            - nu_eff * (u_xx + u_yy)
            - self.alpha_evm * (nu_t_x * u_x + nu_t_y * u_y)
        )
        
        # y 方向動量方程式 (穩態)
        momentum_y = (
            u * v_x + v * v_y + p_y
            - nu_eff * (v_xx + v_yy)
            - self.alpha_evm * (nu_t_x * v_x + nu_t_y * v_y)
        )
        
        # 連續性方程式 (不可壓縮)
        continuity = u_x + v_y
        
        return {
            "momentum_x": momentum_x,
            "momentum_y": momentum_y, 
            "continuity": continuity
        }
    
    def compute_residual(
        self, 
        input_vars: Dict[str, torch.Tensor], 
        model_output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """計算總 PDE 殘差"""
        
        equations = self.forward(input_vars, model_output)
        
        # 計算均方誤差
        momentum_x_loss = torch.mean(equations["momentum_x"]**2)
        momentum_y_loss = torch.mean(equations["momentum_y"]**2)
        continuity_loss = torch.mean(equations["continuity"]**2)
        
        # 總殘差
        total_residual = momentum_x_loss + momentum_y_loss + continuity_loss
        
        return total_residual


class EVMConstraintEquation(PDE):
    """
    PhysicsNeMo EVM 約束方程式
    
    實作熵黏性約束：
    νₜ = α_evm × |R| 
    其中 R 是局部流動殘差
    """
    
    def __init__(self, alpha_constraint: float = 1.0):
        super().__init__()
        
        self.alpha_constraint = alpha_constraint
        
    def forward(
        self, 
        input_vars: Dict[str, torch.Tensor], 
        model_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """計算 EVM 約束殘差"""
        
        x = input_vars["x"]
        y = input_vars["y"]
        
        u = model_output["u"]
        v = model_output["v"]
        nu_t = model_output["nu_t"]
        
        # 計算局部流動殘差 (使用原始 ev-NSFnet 方法)
        u_x = gradient(u, x)
        u_y = gradient(u, y)
        v_x = gradient(v, x)
        v_y = gradient(v, y)
        
        # 實作原始 ev-NSFnet 的 EVM 約束
        # residual_term = (u-0.5) * (u*u_x + v*u_y) + (v-0.5) * (u*v_x + v*v_y)
        residual_term = (u - 0.5) * (u * u_x + v * u_y) + (v - 0.5) * (u * v_x + v * v_y)
        
        # EVM 約束：殘差項應該等於渦黏度
        evm_constraint = residual_term - nu_t
        
        return {
            "evm_constraint": evm_constraint
        }
    
    def compute_residual(
        self, 
        input_vars: Dict[str, torch.Tensor], 
        model_output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """計算 EVM 約束殘差"""
        
        constraints = self.forward(input_vars, model_output)
        
        # 計算約束損失
        evm_loss = torch.mean(constraints["evm_constraint"]**2)
        
        return self.alpha_constraint * evm_loss


class BoundaryConditionEquation(PDE):
    """
    PhysicsNeMo 邊界條件方程式
    
    實作腔體流動邊界條件：
    - 頂部移動蓋子：u = 1-cosh(50*(x-0.5))/cosh(25), v = 0
    - 其他壁面：u = 0, v = 0 (無滑移)
    """
    
    def __init__(self, alpha_boundary: float = 10.0):
        super().__init__()
        
        self.alpha_boundary = alpha_boundary
        
    def compute_moving_lid_velocity(self, x: torch.Tensor) -> torch.Tensor:
        """計算移動蓋子的速度分布"""
        
        # 光滑的移動蓋子速度分布
        # u = 1 - cosh(50*(x-0.5)) / cosh(25)
        u_lid = 1.0 - torch.cosh(50.0 * (x - 0.5)) / torch.cosh(torch.tensor(25.0))
        
        return u_lid
    
    def forward(
        self, 
        input_vars: Dict[str, torch.Tensor], 
        model_output: Dict[str, torch.Tensor],
        boundary_type: str = "wall"
    ) -> Dict[str, torch.Tensor]:
        """計算邊界條件殘差"""
        
        x = input_vars["x"]
        y = input_vars["y"]
        
        u = model_output["u"]
        v = model_output["v"]
        
        if boundary_type == "moving_lid":
            # 頂部移動蓋子 (y = 1)
            u_target = self.compute_moving_lid_velocity(x)
            v_target = torch.zeros_like(v)
            
        elif boundary_type == "wall":
            # 其他壁面無滑移條件
            u_target = torch.zeros_like(u)
            v_target = torch.zeros_like(v)
            
        else:
            raise ValueError(f"未知邊界類型: {boundary_type}")
        
        # 邊界條件殘差
        u_bc_residual = u - u_target
        v_bc_residual = v - v_target
        
        return {
            "u_boundary": u_bc_residual,
            "v_boundary": v_bc_residual
        }
    
    def compute_residual(
        self, 
        input_vars: Dict[str, torch.Tensor], 
        model_output: Dict[str, torch.Tensor],
        boundary_type: str = "wall"
    ) -> torch.Tensor:
        """計算邊界條件損失"""
        
        bc_residuals = self.forward(input_vars, model_output, boundary_type)
        
        # 計算邊界損失
        u_bc_loss = torch.mean(bc_residuals["u_boundary"]**2)
        v_bc_loss = torch.mean(bc_residuals["v_boundary"]**2)
        
        total_bc_loss = u_bc_loss + v_bc_loss
        
        return self.alpha_boundary * total_bc_loss