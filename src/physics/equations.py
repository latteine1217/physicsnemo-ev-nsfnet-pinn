"""
物理方程模組 - 實現Navier-Stokes方程和Entropy Viscosity Method

基於ev-NSFnet的物理實現，包含：
1. NavierStokesEquations: 連續性方程、動量方程
2. EntropyViscosityMethod: entropy residual、人工粘滯度計算
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import math


class NavierStokesEquations:
    """
    Navier-Stokes方程組實現
    
    包含：
    - 連續性方程: ∂u/∂x + ∂v/∂y = 0
    - X方向動量方程: u∂u/∂x + v∂u/∂y + ∂p/∂x - (1/Re + ν_t)(∂²u/∂x² + ∂²u/∂y²) = 0
    - Y方向動量方程: u∂v/∂x + v∂v/∂y + ∂p/∂y - (1/Re + ν_t)(∂²v/∂x² + ∂²v/∂y²) = 0
    """
    
    def __init__(self, reynolds_number: float):
        self.Re = reynolds_number
        
    def compute_continuity_equation(self, u_x: torch.Tensor, v_y: torch.Tensor) -> torch.Tensor:
        """
        計算連續性方程殘差
        
        Args:
            u_x: u對x的導數
            v_y: v對y的導數
            
        Returns:
            連續性方程殘差: ∂u/∂x + ∂v/∂y
        """
        return u_x + v_y
    
    def compute_momentum_x_equation(self, 
                                  u: torch.Tensor, v: torch.Tensor, p: torch.Tensor,
                                  u_x: torch.Tensor, u_y: torch.Tensor, p_x: torch.Tensor,
                                  u_xx: torch.Tensor, u_yy: torch.Tensor,
                                  vis_t: torch.Tensor) -> torch.Tensor:
        """
        計算X方向動量方程殘差
        
        Args:
            u, v, p: 速度和壓力場
            u_x, u_y: u的一階導數
            p_x: 壓力對x的導數
            u_xx, u_yy: u的二階導數
            vis_t: 人工粘滯度
            
        Returns:
            X方向動量方程殘差
        """
        # 對流項
        convection = u * u_x + v * u_y
        
        # 壓力項
        pressure = p_x
        
        # 粘性項（分子粘滯度 + 人工粘滯度）
        total_viscosity = 1.0 / self.Re + vis_t
        diffusion = total_viscosity * (u_xx + u_yy)
        
        return convection + pressure - diffusion
    
    def compute_momentum_y_equation(self, 
                                  u: torch.Tensor, v: torch.Tensor, p: torch.Tensor,
                                  v_x: torch.Tensor, v_y: torch.Tensor, p_y: torch.Tensor,
                                  v_xx: torch.Tensor, v_yy: torch.Tensor,
                                  vis_t: torch.Tensor) -> torch.Tensor:
        """
        計算Y方向動量方程殘差
        
        Args:
            u, v, p: 速度和壓力場
            v_x, v_y: v的一階導數
            p_y: 壓力對y的導數
            v_xx, v_yy: v的二階導數
            vis_t: 人工粘滯度
            
        Returns:
            Y方向動量方程殘差
        """
        # 對流項
        convection = u * v_x + v * v_y
        
        # 壓力項
        pressure = p_y
        
        # 粘性項（分子粘滯度 + 人工粘滯度）
        total_viscosity = 1.0 / self.Re + vis_t
        diffusion = total_viscosity * (v_xx + v_yy)
        
        return convection + pressure - diffusion


class EntropyViscosityMethod:
    """
    Entropy Viscosity Method實現
    
    計算entropy residual和人工粘滯度，增強PINN訓練的穩定性
    """
    
    def __init__(self, reynolds_number: float, alpha_evm: float = 0.03, beta: Optional[float] = None):
        self.Re = reynolds_number
        self.alpha_evm = alpha_evm
        self.beta = beta if beta is not None else 1.0
        
        # 基礎粘滯度參數
        self.vis_t0 = 20.0 / self.Re
        
        # GPU上的粘滯度緩存
        self.vis_t_minus_gpu: Optional[torch.Tensor] = None
        self.lock_vis_t_minus = False
        
    def compute_entropy_residual(self, 
                               eq1: torch.Tensor, eq2: torch.Tensor,
                               u: torch.Tensor, v: torch.Tensor,
                               e_raw: torch.Tensor) -> torch.Tensor:
        """
        計算entropy residual方程
        
        Args:
            eq1, eq2: X和Y方向動量方程殘差
            u, v: 速度場
            e_raw: 神經網路預測的原始entropy值
            
        Returns:
            entropy residual: (eq1*(u-0.5) + eq2*(v-0.5)) - e_raw
        """
        # 使用cavity flow的特徵速度0.5作為參考
        entropy_production = eq1 * (u - 0.5) + eq2 * (v - 0.5)
        return entropy_production - e_raw
    
    def compute_artificial_viscosity(self, e_raw: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        計算人工粘滯度
        
        Args:
            e_raw: 神經網路預測的原始entropy值
            batch_size: 批次大小
            
        Returns:
            人工粘滯度vis_t
        """
        # 計算非負的EVM貢獻
        nu_e = self._compute_nu_e(e_raw)
        
        # 更新vis_t_minus (如果未鎖定)
        if not self.lock_vis_t_minus:
            cap_val = self.beta / self.Re
            vis_t_minus_new = torch.minimum(
                self.alpha_evm * nu_e.detach(),
                torch.full_like(nu_e, cap_val)
            )
            self.vis_t_minus_gpu = vis_t_minus_new
        
        # 計算當前步的人工粘滯度
        return self._compute_vis_t_optimized(batch_size, e_raw)
    
    def _compute_nu_e(self, e_raw: torch.Tensor) -> torch.Tensor:
        """計算非負的EVM貢獻"""
        return torch.abs(e_raw)
    
    def _compute_vis_t_optimized(self, batch_size: int, e: torch.Tensor) -> torch.Tensor:
        """
        優化的vis_t計算，避免CPU-GPU轉換
        
        Args:
            batch_size: 批次大小
            e: entropy值
            
        Returns:
            人工粘滯度張量
        """
        device = e.device
        
        if hasattr(self, 'vis_t_minus_gpu') and self.vis_t_minus_gpu is not None:
            # 確保尺寸匹配
            if self.vis_t_minus_gpu.shape[0] != batch_size:
                if self.vis_t_minus_gpu.shape[0] > batch_size:
                    vis_t_minus_batch = self.vis_t_minus_gpu[:batch_size]
                else:
                    # GPU上的重複操作
                    repeat_times = (batch_size + self.vis_t_minus_gpu.shape[0] - 1) // self.vis_t_minus_gpu.shape[0]
                    vis_t_minus_batch = self.vis_t_minus_gpu.repeat(repeat_times, 1)[:batch_size]
            else:
                vis_t_minus_batch = self.vis_t_minus_gpu
            
            # 在GPU上計算minimum
            vis_t0_tensor = torch.full_like(vis_t_minus_batch, self.vis_t0)
            beta_cap = torch.full_like(vis_t_minus_batch, self.beta / self.Re)
            vis_t = torch.minimum(torch.minimum(vis_t0_tensor, vis_t_minus_batch), beta_cap)
        else:
            # 首次運行或沒有前一步數據
            vis_t = torch.full((batch_size, 1), self.vis_t0, device=device, dtype=torch.float32)
            
        return vis_t
    
    def update_alpha_evm(self, new_alpha: float):
        """更新alpha_evm參數（用於多階段訓練）"""
        self.alpha_evm = new_alpha
    
    def freeze_viscosity_update(self):
        """凍結粘滯度更新（用於特定訓練階段）"""
        self.lock_vis_t_minus = True
    
    def unfreeze_viscosity_update(self):
        """解凍粘滯度更新"""
        self.lock_vis_t_minus = False


class GradientComputer:
    """
    梯度計算工具類
    
    提供批量化的一階和二階梯度計算，優化自動微分性能
    """
    
    @staticmethod
    def compute_gradients_batch(outputs: List[torch.Tensor], 
                              inputs: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """
        批量計算多個輸出對多個輸入的梯度
        
        Args:
            outputs: 輸出張量列表 [u, v, p]
            inputs: 輸入張量列表 [x, y]
            
        Returns:
            梯度列表，每個輸出對每個輸入的梯度
        """
        batch_gradients = []
        
        for output in outputs:
            grad_outputs = [torch.ones_like(output, device=output.device)]
            grads = torch.autograd.grad(
                [output],
                inputs,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            # 處理None梯度
            processed_grads = [g if g is not None else torch.zeros_like(inputs[i]) 
                             for i, g in enumerate(grads)]
            batch_gradients.append(processed_grads)
            
        return batch_gradients
    
    @staticmethod
    def compute_second_gradients_batch(first_grads: List[torch.Tensor], 
                                     inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        批量計算二階梯度
        
        Args:
            first_grads: 一階梯度列表
            inputs: 對應的輸入張量列表
            
        Returns:
            二階梯度列表
        """
        second_grads = []
        
        for i, grad in enumerate(first_grads):
            input_tensor = inputs[i]
            grad_outputs = [torch.ones_like(grad, device=grad.device)]
            second_grad = torch.autograd.grad(
                [grad],
                [input_tensor],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            
            if second_grad is None:
                second_grad = torch.zeros_like(input_tensor)
            second_grads.append(second_grad)
            
        return second_grads


class PhysicsEquations:
    """
    統一的物理方程介面
    
    整合Navier-Stokes方程和Entropy Viscosity Method
    """
    
    def __init__(self, reynolds_number: float, alpha_evm: float = 0.03, beta: Optional[float] = None):
        self.navier_stokes = NavierStokesEquations(reynolds_number)
        self.entropy_viscosity = EntropyViscosityMethod(reynolds_number, alpha_evm, beta)
        self.gradient_computer = GradientComputer()
        
    def compute_physics_residuals(self, 
                                x: torch.Tensor, y: torch.Tensor,
                                u: torch.Tensor, v: torch.Tensor, p: torch.Tensor,
                                e_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        計算所有物理方程殘差
        
        Args:
            x, y: 空間座標
            u, v, p: 速度和壓力場
            e_raw: 神經網路預測的entropy值
            
        Returns:
            (eq1, eq2, eq3, eq4): X動量、Y動量、連續性、entropy residual方程殘差
        """
        # 批量計算一階梯度
        outputs = [u, v, p]
        grads = self.gradient_computer.compute_gradients_batch(outputs, [x, y])
        
        u_x, u_y = grads[0]
        v_x, v_y = grads[1]
        p_x, p_y = grads[2]
        
        # 批量計算二階梯度
        second_order_outputs = [u_x, u_y, v_x, v_y]
        second_order_inputs = [x, y, x, y]
        second_grads = self.gradient_computer.compute_second_gradients_batch(
            second_order_outputs, second_order_inputs
        )
        
        u_xx, u_yy, v_xx, v_yy = second_grads
        
        # 計算人工粘滯度
        batch_size = x.shape[0]
        vis_t = self.entropy_viscosity.compute_artificial_viscosity(e_raw, batch_size)
        
        # 計算各方程殘差
        eq1 = self.navier_stokes.compute_momentum_x_equation(
            u, v, p, u_x, u_y, p_x, u_xx, u_yy, vis_t
        )
        
        eq2 = self.navier_stokes.compute_momentum_y_equation(
            u, v, p, v_x, v_y, p_y, v_xx, v_yy, vis_t
        )
        
        eq3 = self.navier_stokes.compute_continuity_equation(u_x, v_y)
        
        eq4 = self.entropy_viscosity.compute_entropy_residual(eq1, eq2, u, v, e_raw)
        
        return eq1, eq2, eq3, eq4
    
    def update_evm_parameters(self, alpha_evm: float):
        """更新EVM參數"""
        self.entropy_viscosity.update_alpha_evm(alpha_evm)
    
    def freeze_evm_update(self):
        """凍結EVM更新"""
        self.entropy_viscosity.freeze_viscosity_update()
    
    def unfreeze_evm_update(self):
        """解凍EVM更新"""
        self.entropy_viscosity.unfreeze_viscosity_update()