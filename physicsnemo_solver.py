# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from omegaconf import DictConfig
from physicsnemo.solver import Solver
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.loggers import get_logger
from physicsnemo.models.model import Model

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

from physicsnemo_net import PhysicsNeMoCombinedPINNModel
from physicsnemo_equations import NavierStokesEVMEquation, EVMConstraintEquation
from physicsnemo_data import PhysicsNeMoCavityDataset


class PhysicsNeMoPINNSolver(Solver):
    """
    PhysicsNeMo 標準 PINN 求解器 - EV-NSFnet 實作
    
    這個求解器實作了熵黏性納維-史托克斯傅立葉網路 (EV-NSFnet)
    用於高雷諾數腔體流動模擬。
    
    Parameters
    ----------
    cfg : DictConfig
        PhysicsNeMo 配置字典，包含模型和訓練參數
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        # 獲取 PhysicsNeMo 日誌器
        self.logger = get_logger(__name__)
        
        # 初始化分散式管理器
        self.dist = DistributedManager()
        
        # 模型配置
        self.reynolds_number = cfg.reynolds_number
        self.alpha_evm = cfg.alpha_evm
        self.alpha_boundary = cfg.alpha_boundary
        self.alpha_equation = cfg.alpha_equation
        self.alpha_evm_constraint = cfg.alpha_evm_constraint
        
        # 初始化模型組件
        self._init_model()
        self._init_equations()
        self._init_dataset()
        
        # 訓練狀態
        self.current_stage = ""
        self.evm_frozen = False
        
        self.logger.info(f"PhysicsNeMo PINN 求解器初始化完成 - Re={self.reynolds_number}")
        
    def _init_model(self):
        """初始化 PhysicsNeMo 神經網路模型"""
        
        # PINN 網路配置
        pinn_config = {
            "input_keys": ["x", "y"],
            "output_keys": ["u", "v", "p"],
            "nr_layers": self.cfg.main_net.nr_layers,
            "layer_size": self.cfg.main_net.layer_size,
            "activation_fn": self.cfg.main_net.activation_fn,
        }
        
        # EVM 網路配置
        evm_config = {
            "input_keys": ["x", "y"],
            "output_keys": ["nu_t"],  # 渦黏度
            "nr_layers": self.cfg.evm_net.nr_layers,
            "layer_size": self.cfg.evm_net.layer_size,
            "activation_fn": self.cfg.evm_net.activation_fn,
        }
        
        # 建立結合模型
        self.model = PhysicsNeMoCombinedPINNModel(
            pinn_config=pinn_config,
            evm_config=evm_config
        )
        
        # 移動到適當裝置
        self.model = self.model.to(self.dist.device)
        
        if self.dist.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.dist.local_rank]
            )
            
    def _init_equations(self):
        """初始化 PDE 方程式"""
        
        # 納維-史托克斯 EVM 方程式
        self.ns_equation = NavierStokesEVMEquation(
            reynolds_number=self.reynolds_number,
            alpha_evm=self.alpha_evm
        )
        
        # EVM 約束方程式
        self.evm_constraint = EVMConstraintEquation(
            alpha_constraint=self.alpha_evm_constraint
        )
        
    def _init_dataset(self):
        """初始化資料集"""
        
        self.dataset = PhysicsNeMoCavityDataset(
            data_dir=self.cfg.data_dir,
            reynolds_number=self.reynolds_number,
            num_interior_points=self.cfg.num_interior_points,
            num_boundary_points=self.cfg.num_boundary_points,
            dist=self.dist
        )
        
    def compute_losses(
        self, 
        input_vars: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """計算所有損失項"""
        
        # 獲取模型預測
        model_output = self.model(input_vars)
        
        # 納維-史托克斯方程式損失
        ns_loss = self.ns_equation.compute_residual(input_vars, model_output)
        
        # EVM 約束損失
        evm_loss = self.evm_constraint.compute_residual(input_vars, model_output)
        
        # 邊界條件損失
        boundary_loss = self._compute_boundary_loss(input_vars, model_output)
        
        # 總損失
        total_loss = (
            self.alpha_equation * ns_loss +
            self.alpha_evm_constraint * evm_loss +
            self.alpha_boundary * boundary_loss
        )
        
        return {
            "total_loss": total_loss,
            "ns_loss": ns_loss,
            "evm_loss": evm_loss,
            "boundary_loss": boundary_loss
        }
    
    def _compute_boundary_loss(
        self, 
        input_vars: Dict[str, torch.Tensor], 
        model_output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """計算邊界條件損失"""
        
        # 獲取邊界資料
        boundary_data = self.dataset.get_boundary_data()
        
        # 頂部移動蓋子邊界條件
        top_loss = torch.mean((model_output["u"] - boundary_data["u_top"])**2)
        top_loss += torch.mean((model_output["v"] - boundary_data["v_top"])**2)
        
        # 其他壁面無滑移條件
        wall_loss = torch.mean(model_output["u"]**2) + torch.mean(model_output["v"]**2)
        
        return top_loss + wall_loss
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """PhysicsNeMo 標準訓練步驟"""
        
        # 計算損失
        losses = self.compute_losses(batch)
        
        return losses
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """PhysicsNeMo 標準驗證步驟"""
        
        with torch.no_grad():
            losses = self.compute_losses(batch)
            
        return losses
    
    def set_training_stage(self, stage_name: str, alpha_evm: float):
        """設置訓練階段參數"""
        
        self.current_stage = stage_name
        self.alpha_evm = alpha_evm
        
        # 更新方程式中的 alpha_evm
        self.ns_equation.alpha_evm = alpha_evm
        
        self.logger.info(f"設置訓練階段: {stage_name}, alpha_evm={alpha_evm}")
    
    def freeze_evm_network(self):
        """凍結 EVM 網路"""
        if hasattr(self.model, 'module'):
            self.model.module.freeze_evm()
        else:
            self.model.freeze_evm()
            
        self.evm_frozen = True
        self.logger.info("EVM 網路已凍結")
    
    def unfreeze_evm_network(self):
        """解凍 EVM 網路"""
        if hasattr(self.model, 'module'):
            self.model.module.unfreeze_evm()
        else:
            self.model.unfreeze_evm()
            
        self.evm_frozen = False
        self.logger.info("EVM 網路已解凍")
    
    def evaluate(
        self, 
        x_eval: torch.Tensor, 
        y_eval: torch.Tensor,
        u_ref: torch.Tensor, 
        v_ref: torch.Tensor, 
        p_ref: torch.Tensor
    ) -> Dict[str, float]:
        """評估模型精度"""
        
        self.model.eval()
        
        with torch.no_grad():
            input_vars = {
                "x": x_eval.to(self.dist.device),
                "y": y_eval.to(self.dist.device)
            }
            
            model_output = self.model(input_vars)
            
            # 計算相對誤差
            u_pred = model_output["u"].cpu()
            v_pred = model_output["v"].cpu()
            p_pred = model_output["p"].cpu()
            
            error_u = torch.mean((u_pred - u_ref)**2) / torch.mean(u_ref**2) * 100
            error_v = torch.mean((v_pred - v_ref)**2) / torch.mean(v_ref**2) * 100
            error_p = torch.mean((p_pred - p_ref)**2) / torch.mean(p_ref**2) * 100
            
        self.model.train()
        
        return {
            "error_u": error_u.item(),
            "error_v": error_v.item(),
            "error_p": error_p.item()
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