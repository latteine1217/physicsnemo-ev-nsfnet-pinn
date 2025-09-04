# SPDX-FileCopyrightText: Copyright (c) 2024 LDC-PINNs
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================
# LDC-PINNs: Physics-Informed Neural Networks for Lid-Driven Cavity Flow
# 基於 NVIDIA PhysicsNeMo 框架實現
# ==============================================================================

import os
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

# 本專案的物理模組
from src.physics.equations import PhysicsEquations, EntropyViscosityMethod
from src.models.activations import TSAActivation, get_activation_function, compute_activation_regularization

# PyTorch optimization
from torch.optim import Adam, LBFGS
from torch.optim import lr_scheduler

# 簡單的日誌模組
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_output_directory(cfg: DictConfig) -> str:
    """創建輸出目錄"""
    output_dir = cfg.outputs.save_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """設置神經網路模型"""
    class FullyConnectedNetwork(torch.nn.Module):
        def __init__(self, in_features, out_features, num_layers, layer_size):
            super().__init__()
            layers = []
            prev_size = in_features
            
            for _ in range(num_layers):
                layers.append(torch.nn.Linear(prev_size, layer_size))
                
                # 使用進階激活函數（如果啟用）
                if cfg.model.get('advanced_activation', {}).get('enabled', False):
                    activation_config = cfg.model.advanced_activation.copy()
                    activation_config['num_neurons'] = layer_size
                    activation = get_activation_function(activation_config)
                else:
                    activation = torch.nn.SiLU()
                
                layers.append(activation)
                prev_size = layer_size
            
            # 輸出層
            layers.append(torch.nn.Linear(prev_size, out_features))
            
            self.network = torch.nn.Sequential(*layers)
            
            # Xavier初始化
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        def forward(self, x):
            return self.network(x)
    
    # 主網路 (u, v, p)
    main_network = FullyConnectedNetwork(
        in_features=cfg.model.in_features,
        out_features=cfg.model.out_features,
        num_layers=cfg.model.num_layers,
        layer_size=cfg.model.layer_size
    ).to(device)
    
    return main_network


def setup_physics(cfg: DictConfig, device: torch.device) -> PhysicsInformer:
    """設置物理方程式和PhysicsInformer"""
    # 創建Navier-Stokes方程式
    ns = NavierStokes(
        nu=cfg.physics.nu,
        rho=cfg.physics.rho,
        dim=cfg.physics.dim,
        time=cfg.physics.time
    )
    
    # 創建PhysicsInformer
    phy_inf = PhysicsInformer(
        required_outputs=["continuity", "momentum_x", "momentum_y"],
        equations=ns,
        grad_method="autodiff",
        device=device,
    )
    
    return phy_inf


def setup_geometry_dataloaders(cfg: DictConfig, device: torch.device):
    """設置幾何和資料載入器"""
    # 創建幾何對象 (正方形cavity)
    bounds = cfg.geometry.domain.bounds
    rec = Rectangle(
        (bounds[0][0], bounds[0][1]),  # lower left
        (bounds[1][0], bounds[1][1])   # upper right  
    )
    
    # 邊界條件資料載入器
    bc_dataloader = GeometryDatapipe(
        geom_objects=[rec],
        batch_size=cfg.geometry.sampling.batch_size,
        num_points=cfg.geometry.sampling.boundary_points,
        sample_type="surface",
        device=device,
        num_workers=cfg.geometry.sampling.num_workers,
        requested_vars=["x", "y"],
    )
    
    # 內部點資料載入器
    interior_dataloader = GeometryDatapipe(
        geom_objects=[rec],
        batch_size=cfg.geometry.sampling.batch_size,
        num_points=cfg.geometry.sampling.interior_points,
        sample_type="volume",
        device=device,
        num_workers=cfg.geometry.sampling.num_workers,
        requested_vars=["x", "y", "sdf"],
    )
    
    return bc_dataloader, interior_dataloader, rec


def setup_optimizer_and_scheduler(cfg: DictConfig, model: torch.nn.Module):
    """設置優化器和學習率調度器"""
    optimizer = Adam(
        model.parameters(), 
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    scheduler = lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: cfg.training.scheduler.decay_rate ** step
    )
    
    return optimizer, scheduler


def create_inference_grid(cfg: DictConfig, device: torch.device):
    """創建推理網格"""
    resolution = cfg.inference.grid_resolution
    domain = cfg.inference.domain
    
    x = np.linspace(domain[0][0], domain[0][1], resolution)
    y = np.linspace(domain[1][0], domain[1][1], resolution)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    
    xx = torch.from_numpy(xx).to(torch.float).to(device)
    yy = torch.from_numpy(yy).to(torch.float).to(device)
    
    return xx, yy


def apply_boundary_conditions(cfg: DictConfig, bc_data: dict, model: torch.nn.Module):
    """應用邊界條件並計算邊界損失"""
    cavity_size = cfg.physics.cavity_size
    height = cavity_size
    
    y_vals = bc_data[0]["y"]
    
    # 分離不同的邊界
    mask_no_slip = y_vals < height / 2  # bottom, left, right walls
    mask_top_wall = y_vals == height / 2  # moving top wall
    
    # 提取邊界點
    no_slip = {}
    top_wall = {}
    
    for k in bc_data[0].keys():
        no_slip[k] = (bc_data[0][k][mask_no_slip]).reshape(-1, 1)
        top_wall[k] = (bc_data[0][k][mask_top_wall]).reshape(-1, 1)
    
    # 模型推理
    no_slip_coords = torch.cat([no_slip["x"], no_slip["y"]], dim=1)
    top_wall_coords = torch.cat([top_wall["x"], top_wall["y"]], dim=1)
    
    no_slip_out = model(no_slip_coords)
    top_wall_out = model(top_wall_coords)
    
    # 計算邊界條件損失
    bc_cfg = cfg.boundary_conditions
    
    # 無滑移邊界 (u=0, v=0)
    v_no_slip = torch.mean(no_slip_out[:, 1:2] ** 2) * bc_cfg.no_slip_walls.weight
    u_no_slip = torch.mean(no_slip_out[:, 0:1] ** 2) * bc_cfg.no_slip_walls.weight
    
    # 移動頂壁 (u=1, v=0) 
    u_slip = torch.mean(
        ((top_wall_out[:, 0:1] - bc_cfg.moving_lid.u) ** 2) * 
        (1 - bc_cfg.moving_lid.edge_attenuation * torch.abs(top_wall["x"]))
    ) * bc_cfg.moving_lid.weight
    
    v_slip = torch.mean(
        (top_wall_out[:, 1:2] - bc_cfg.moving_lid.v) ** 2
    ) * bc_cfg.moving_lid.weight
    
    return u_no_slip + v_no_slip + u_slip + v_slip


def apply_physics_constraints(cfg: DictConfig, int_data: dict, model: torch.nn.Module, 
                            phy_inf: PhysicsInformer):
    """應用物理約束並計算物理損失"""
    interior = {}
    
    # 準備內部點資料
    for k, v in int_data[0].items():
        if k in ["x", "y"]:
            requires_grad = True
        else:
            requires_grad = False
        interior[k] = v.reshape(-1, 1).requires_grad_(requires_grad)
    
    # 模型前向傳播
    coords = torch.cat([interior["x"], interior["y"]], dim=1)
    interior_out = model(coords)
    
    # 使用PhysicsInformer計算物理殘差
    phy_loss_dict = phy_inf.forward({
        "coordinates": coords,
        "u": interior_out[:, 0:1],
        "v": interior_out[:, 1:2],
        "p": interior_out[:, 2:3],
    })
    
    # 使用SDF權重物理損失 (僅在域內部應用約束)
    sdf = interior["sdf"]
    cont = phy_loss_dict["continuity"] * sdf
    mom_x = phy_loss_dict["momentum_x"] * sdf
    mom_y = phy_loss_dict["momentum_y"] * sdf
    
    # 加權物理損失
    weights = cfg.loss_weights
    physics_loss = (
        weights.continuity * torch.mean(cont ** 2) +
        weights.momentum_x * torch.mean(mom_x ** 2) +
        weights.momentum_y * torch.mean(mom_y ** 2)
    )
    
    return physics_loss


def plot_results(cfg: DictConfig, model: torch.nn.Module, xx: torch.Tensor, 
                yy: torch.Tensor, epoch: int, loss_value: float, output_dir: str):
    """繪製結果"""
    if not cfg.outputs.save_plots:
        return
        
    resolution = cfg.inference.grid_resolution
    
    with torch.no_grad():
        # 推理
        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
        inf_out = model(coords)
        out_np = inf_out.detach().cpu().numpy()
        
        # 創建圖像
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # u速度
        im = axes[0].imshow(
            out_np[:, 0].reshape(resolution, resolution), 
            origin="lower", extent=cfg.inference.domain[0] + cfg.inference.domain[1]
        )
        fig.colorbar(im, ax=axes[0])
        axes[0].set_title(f"u velocity (epoch {epoch})")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        
        # v速度
        im = axes[1].imshow(
            out_np[:, 1].reshape(resolution, resolution), 
            origin="lower", extent=cfg.inference.domain[0] + cfg.inference.domain[1]
        )
        fig.colorbar(im, ax=axes[1])
        axes[1].set_title(f"v velocity (epoch {epoch})")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        
        # 壓力
        im = axes[2].imshow(
            out_np[:, 2].reshape(resolution, resolution), 
            origin="lower", extent=cfg.inference.domain[0] + cfg.inference.domain[1]
        )
        fig.colorbar(im, ax=axes[2])
        axes[2].set_title(f"Pressure (epoch {epoch})")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        
        # 速度幅度
        u_mag = ((out_np[:, 0] ** 2 + out_np[:, 1] ** 2) ** 0.5).reshape(resolution, resolution)
        im = axes[3].imshow(
            u_mag, origin="lower", 
            extent=cfg.inference.domain[0] + cfg.inference.domain[1]
        )
        fig.colorbar(im, ax=axes[3])
        axes[3].set_title(f"Velocity Magnitude (epoch {epoch})")
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("y")
        
        # 添加標題
        fig.suptitle(f"LDC Flow (Re={cfg.physics.Re}) - Loss: {loss_value:.2e}", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ldc_results_epoch_{epoch:06d}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()


@hydra.main(version_base="1.3", config_path="configs", config_name="ldc_pinn")
def ldc_trainer(cfg: DictConfig) -> None:
    """主訓練函數"""
    # 初始化分布式管理器
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # 設置日誌
    log = PythonLogger(name="ldc-physicsnemo")
    if cfg.logging.file_logging:
        log.file_logging()
    
    if dist.rank == 0:
        log.info("🚀 啟動PhysicsNeMo LDC-PINN訓練")
        log.info(f"📊 設備: {dist.device}")
        log.info(f"🔧 Reynolds數: {cfg.physics.Re}")
        log.info(f"🎯 最大epochs: {cfg.training.max_epochs}")
    
    # 創建輸出目錄
    output_dir = create_output_directory(cfg)
    
    # 設置模型
    model = setup_model(cfg, dist.device)
    
    # 設置物理方程式
    phy_inf = setup_physics(cfg, dist.device)
    
    # 設置幾何和資料載入器
    bc_dataloader, interior_dataloader, geometry = setup_geometry_dataloaders(cfg, dist.device)
    
    # 設置優化器和調度器
    optimizer, scheduler = setup_optimizer_and_scheduler(cfg, model)
    
    # 創建推理網格
    xx, yy = create_inference_grid(cfg, dist.device)
    
    if dist.rank == 0:
        log.info("✅ 初始化完成，開始訓練")
    
    # 訓練循環
    for epoch in range(cfg.training.max_epochs):
        for bc_data, int_data in zip(bc_dataloader, interior_dataloader):
            optimizer.zero_grad()
            
            # 計算邊界條件損失
            boundary_loss = apply_boundary_conditions(cfg, bc_data, model)
            
            # 計算物理約束損失
            physics_loss = apply_physics_constraints(cfg, int_data, model, phy_inf)
            
            # 總損失
            total_loss = boundary_loss + physics_loss
            
            # 反向傳播
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
        # 日誌和繪圖
        if dist.rank == 0:
            if epoch % cfg.training.log_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                log.info(f"Epoch {epoch:6d}: Loss = {total_loss.item():.2e}, "
                        f"LR = {current_lr:.2e}")
            
            if epoch % cfg.training.plot_freq == 0:
                plot_results(cfg, model, xx, yy, epoch, total_loss.item(), output_dir)
    
    if dist.rank == 0:
        log.info("🎉 訓練完成!")


if __name__ == "__main__":
    ldc_trainer()