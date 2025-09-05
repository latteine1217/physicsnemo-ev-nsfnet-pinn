# SPDX-FileCopyrightText: Copyright (c) 2024 LDC-PINNs
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================
# 完整PhysicsNeMo框架實作 - 整合人工粘滯性與多階段訓練
# Complete PhysicsNeMo Framework with Artificial Viscosity & Multi-stage Training
# ==============================================================================

import os
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
from omegaconf import DictConfig
from typing import Dict, Tuple, Optional, Any, List

# PhysicsNeMo imports
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.geometry.geometry_dataloader import GeometryDatapipe
from physicsnemo.sym.geometry.primitives_2d import Rectangle

# 本專案模組
from src.models.activations import get_activation_function, compute_activation_regularization
from src.physics.equations import EntropyViscosityMethod
from src.physics.nemo_sym import NemoSymEquations


class EntropyResidualNetwork(nn.Module):
    """
    熵殘差網路 - 用於計算人工粘滯性的副網路
    基於ev-NSFnet的架構：4層 × 40神經元
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_layers: int = 4,
                 hidden_size: int = 40,
                 activation_config: Optional[Dict] = None):
        super().__init__()
        
        layers = []
        prev_size = input_dim
        
        # 建構隱藏層
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # 使用進階激活函數或標準激活函數
            if activation_config and activation_config.get('enabled', False):
                activation_config['num_neurons'] = hidden_size
                activation = get_activation_function(activation_config)
            else:
                activation = nn.Tanh()
            
            layers.append(activation)
            prev_size = hidden_size
        
        # 輸出層 (entropy residual)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Xavier初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.network(coords)


class AdvancedFullyConnectedNetwork(nn.Module):
    """
    進階全連接網路 - 整合TSA/LAAF激活函數
    基於ev-NSFnet的架構：6層 × 80神經元（主網路）
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 output_dim: int = 3,
                 hidden_layers: int = 6,
                 hidden_size: int = 80,
                 activation_config: Optional[Dict] = None):
        super().__init__()
        
        layers = []
        prev_size = input_dim
        
        # 建構隱藏層
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # 使用進階激活函數
            if activation_config and activation_config.get('enabled', False):
                activation_config['num_neurons'] = hidden_size
                activation = get_activation_function(activation_config)
            else:
                activation = nn.SiLU()  # 預設使用SiLU
            
            layers.append(activation)
            prev_size = hidden_size
        
        # 輸出層
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Xavier初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.network(coords)


class MultiStageTrainingManager:
    """
    多階段訓練管理器 - 實現ev-NSFnet的5階段訓練策略
    """
    
    def __init__(self, config: DictConfig):
        self.total_epochs = config.training.max_epochs
        self.stages = config.training.stages
        self.current_stage = 0
        
        # 預設5階段配置（如果配置中未指定）
        if not hasattr(config.training, 'stages') or not config.training.stages:
            self.stages = self._get_default_stages()
    
    def _get_default_stages(self) -> List[Dict]:
        """取得預設的5階段訓練配置"""
        epochs_per_stage = self.total_epochs // 5
        
        return [
            {
                'name': 'Stage_1_Initial',
                'epochs': epochs_per_stage,
                'alpha_evm': 0.03,
                'learning_rate': 1e-3,
                'optimizer': 'Adam',
                'use_lbfgs': False
            },
            {
                'name': 'Stage_2_Refinement',
                'epochs': epochs_per_stage,
                'alpha_evm': 0.01,
                'learning_rate': 8e-4,
                'optimizer': 'Adam',
                'use_lbfgs': False
            },
            {
                'name': 'Stage_3_Mixed',
                'epochs': epochs_per_stage,
                'alpha_evm': 0.003,
                'learning_rate': 5e-4,
                'optimizer': 'Mixed',  # 60% Adam + 40% L-BFGS
                'use_lbfgs': True,
                'lbfgs_ratio': 0.4
            },
            {
                'name': 'Stage_4_Precision',
                'epochs': epochs_per_stage,
                'alpha_evm': 0.001,
                'learning_rate': 3e-4,
                'optimizer': 'Mixed',
                'use_lbfgs': True,
                'lbfgs_ratio': 0.4
            },
            {
                'name': 'Stage_5_Final',
                'epochs': epochs_per_stage,
                'alpha_evm': 0.0002,
                'learning_rate': 1e-4,
                'optimizer': 'Mixed',
                'use_lbfgs': True,
                'lbfgs_ratio': 0.5
            }
        ]
    
    def get_current_stage_config(self, epoch: int) -> Dict:
        """取得當前epoch對應的階段配置"""
        cumulative_epochs = 0
        
        for i, stage in enumerate(self.stages):
            cumulative_epochs += stage['epochs']
            if epoch < cumulative_epochs:
                stage_config = stage.copy()
                stage_config['stage_index'] = i
                stage_config['stage_start_epoch'] = cumulative_epochs - stage['epochs']
                stage_config['stage_end_epoch'] = cumulative_epochs
                stage_config['stage_progress'] = (epoch - stage_config['stage_start_epoch']) / stage['epochs']
                return stage_config
        
        # 如果超過所有階段，返回最後一個階段
        last_stage = self.stages[-1].copy()
        last_stage['stage_index'] = len(self.stages) - 1
        return last_stage
    
    def should_switch_to_lbfgs(self, stage_config: Dict) -> bool:
        """判斷是否應該切換到L-BFGS優化器"""
        if not stage_config.get('use_lbfgs', False):
            return False
        
        lbfgs_threshold = 1.0 - stage_config.get('lbfgs_ratio', 0.4)
        return stage_config['stage_progress'] >= lbfgs_threshold
    
    def print_stage_transition(self, stage_config: Dict):
        """列印階段轉換資訊"""
        print(f"\n{'='*60}")
        print(f"🚀 進入訓練階段 {stage_config['stage_index'] + 1}: {stage_config['name']}")
        print(f"📊 Alpha EVM: {stage_config['alpha_evm']}")
        print(f"📈 學習率: {stage_config['learning_rate']:.2e}")
        print(f"🔧 優化器: {stage_config['optimizer']}")
        if stage_config.get('use_lbfgs'):
            print(f"⚡ L-BFGS比例: {stage_config.get('lbfgs_ratio', 0.4):.1%}")
        print(f"📅 Epochs: {stage_config['stage_start_epoch']} - {stage_config['stage_end_epoch']}")
        print(f"{'='*60}")


class DualNetworkPINNSolver:
    """
    雙網路PINN求解器
    主網路: [u, v, p] - 6層×80神經元
    副網路: [entropy_residual] - 4層×40神經元
    """
    
    def __init__(self, config: DictConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # 建立主/副網路：優先使用PhysicsNeMo FullyConnected，失敗則回退到本地網路
        try:
            self.main_network = FullyConnected(
                in_features=2,
                out_features=3,
                num_layers=config.model.main_network.num_layers,
                layer_size=config.model.main_network.layer_size,
            ).to(device)
        except Exception:
            self.main_network = AdvancedFullyConnectedNetwork(
                input_dim=2,
                output_dim=3,
                hidden_layers=config.model.main_network.num_layers,
                hidden_size=config.model.main_network.layer_size,
                activation_config=config.model.get('advanced_activation', None)
            ).to(device)

        try:
            self.entropy_network = FullyConnected(
                in_features=2,
                out_features=1,
                num_layers=config.model.entropy_network.num_layers,
                layer_size=config.model.entropy_network.layer_size,
            ).to(device)
        except Exception:
            self.entropy_network = EntropyResidualNetwork(
                input_dim=2,
                hidden_layers=config.model.entropy_network.num_layers,
                hidden_size=config.model.entropy_network.layer_size,
                activation_config=config.model.get('advanced_activation', None)
            ).to(device)
        
        # 物理方程式：優先使用PhysicsNeMo-Sym封裝（含Informer+EVM），避免破壞原有資料流
        self.physics_equations = NemoSymEquations(
            reynolds_number=config.physics.Re,
            alpha_evm=config.physics.alpha_evm,
            beta=config.physics.get('beta', 1.0)
        )
        
        # 多階段訓練管理器
        self.training_manager = MultiStageTrainingManager(config)
        
        # 優化器（將在訓練過程中動態創建）
        self.main_optimizer = None
        self.entropy_optimizer = None
        self.lbfgs_optimizer = None
        self.current_stage_config = None
    
    def create_optimizers(self, stage_config: Dict):
        """根據階段配置創建優化器"""
        lr = stage_config['learning_rate']
        
        # 主網路優化器
        self.main_optimizer = Adam(
            self.main_network.parameters(),
            lr=lr,
            weight_decay=self.config.training.get('weight_decay', 0.0)
        )
        
        # 副網路優化器
        self.entropy_optimizer = Adam(
            self.entropy_network.parameters(),
            lr=lr,
            weight_decay=self.config.training.get('weight_decay', 0.0)
        )
        
        # L-BFGS優化器（如果需要）
        if stage_config.get('use_lbfgs', False):
            # 合併所有參數用於L-BFGS
            all_params = list(self.main_network.parameters()) + list(self.entropy_network.parameters())
            self.lbfgs_optimizer = LBFGS(
                all_params,
                lr=lr * 0.8,  # L-BFGS使用較小的學習率
                max_iter=20,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=100,
                line_search_fn='strong_wolfe'
            )
    
    def compute_physics_loss(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """計算物理損失"""
        x, y = coords[:, 0:1], coords[:, 1:2]
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        # 主網路前向傳播
        main_output = self.main_network(coords)
        u, v, p = main_output[:, 0:1], main_output[:, 1:2], main_output[:, 2:3]
        
        # 副網路前向傳播
        e_raw = self.entropy_network(coords)
        
        # 計算物理方程殘差
        eq1, eq2, eq3, eq4 = self.physics_equations.compute_residuals(x, y, u, v, p, e_raw)
        
        # 計算損失
        losses = {
            'momentum_x': torch.mean(eq1 ** 2),
            'momentum_y': torch.mean(eq2 ** 2),
            'continuity': torch.mean(eq3 ** 2),
            'entropy_residual': torch.mean(eq4 ** 2)
        }
        
        return losses
    
    def compute_boundary_loss(self, bc_data: Dict) -> torch.Tensor:
        """計算邊界條件損失"""
        cavity_size = self.config.physics.cavity_size
        
        coords = bc_data['coords']
        y_vals = coords[:, 1:2]
        
        # 分離不同邊界
        mask_no_slip = torch.abs(y_vals + cavity_size/2) < 1e-6  # bottom wall
        mask_no_slip = mask_no_slip | (torch.abs(coords[:, 0:1] + cavity_size/2) < 1e-6)  # left wall
        mask_no_slip = mask_no_slip | (torch.abs(coords[:, 0:1] - cavity_size/2) < 1e-6)  # right wall
        
        mask_moving_lid = torch.abs(y_vals - cavity_size/2) < 1e-6  # top wall
        
        # 模型推理
        output = self.main_network(coords)
        u, v = output[:, 0:1], output[:, 1:2]
        
        # 無滑移邊界條件
        no_slip_loss = torch.tensor(0.0, device=self.device)
        if mask_no_slip.any():
            no_slip_coords = coords[mask_no_slip.squeeze()]
            if no_slip_coords.numel() > 0:
                no_slip_output = self.main_network(no_slip_coords)
                no_slip_u, no_slip_v = no_slip_output[:, 0:1], no_slip_output[:, 1:2]
                no_slip_loss = torch.mean(no_slip_u**2 + no_slip_v**2)
        
        # 移動頂壁邊界條件
        moving_lid_loss = torch.tensor(0.0, device=self.device)
        if mask_moving_lid.any():
            lid_coords = coords[mask_moving_lid.squeeze()]
            if lid_coords.numel() > 0:
                lid_output = self.main_network(lid_coords)
                lid_u, lid_v = lid_output[:, 0:1], lid_output[:, 1:2]
                
                # u = 1 (移動速度), v = 0
                lid_u_target = 1.0
                lid_v_target = 0.0
                moving_lid_loss = torch.mean((lid_u - lid_u_target)**2 + (lid_v - lid_v_target)**2)
        
        return no_slip_loss + moving_lid_loss
    
    def training_step(self, interior_data: Dict, boundary_data: Dict, 
                     stage_config: Dict, use_lbfgs: bool = False) -> Dict[str, float]:
        """單步訓練"""
        
        def closure():
            """L-BFGS closure函數"""
            if use_lbfgs and self.lbfgs_optimizer:
                self.lbfgs_optimizer.zero_grad()
            else:
                self.main_optimizer.zero_grad()
                self.entropy_optimizer.zero_grad()
            
            # 計算物理損失
            physics_losses = self.compute_physics_loss(interior_data['coords'])
            
            # 計算邊界損失
            boundary_loss = self.compute_boundary_loss(boundary_data)
            
            # 權重配置
            weights = self.config.loss_weights
            
            # 總損失
            total_loss = (
                weights.momentum_x * physics_losses['momentum_x'] +
                weights.momentum_y * physics_losses['momentum_y'] +
                weights.continuity * physics_losses['continuity'] +
                stage_config['alpha_evm'] * physics_losses['entropy_residual'] +
                weights.boundary * boundary_loss
            )
            
            # 激活函數正則化
            if self.config.model.get('advanced_activation', {}).get('enabled', False):
                activation_reg = compute_activation_regularization(self.main_network) + \
                               compute_activation_regularization(self.entropy_network)
                total_loss += 0.01 * activation_reg
            
            total_loss.backward()
            
            return total_loss
        
        # 執行優化步驟
        if use_lbfgs and self.lbfgs_optimizer:
            loss = self.lbfgs_optimizer.step(closure)
        else:
            loss = closure()
            self.main_optimizer.step()
            self.entropy_optimizer.step()
        
        # 返回損失資訊
        return {
            'total_loss': loss.item(),
            'optimizer': 'L-BFGS' if use_lbfgs else 'Adam'
        }
    
    def update_stage(self, new_stage_config: Dict):
        """更新訓練階段"""
        if self.current_stage_config != new_stage_config:
            self.current_stage_config = new_stage_config
            
            # 更新物理方程參數
            self.physics_equations.update_evm_parameters(new_stage_config['alpha_evm'])
            
            # 重新創建優化器
            self.create_optimizers(new_stage_config)
            
            # 列印階段轉換資訊
            self.training_manager.print_stage_transition(new_stage_config)


def create_data_loaders(config: DictConfig, device: torch.device):
    """創建數據載入器"""
    # 創建幾何對象
    bounds = config.geometry.domain.bounds
    geometry = Rectangle(
        (bounds[0][0], bounds[0][1]),  # lower left
        (bounds[1][0], bounds[1][1])   # upper right
    )
    
    # 邊界條件數據載入器
    bc_dataloader = GeometryDatapipe(
        geom_objects=[geometry],
        batch_size=1,
        num_points=config.geometry.sampling.boundary_points,
        sample_type="surface",
        device=device,
        num_workers=config.geometry.sampling.num_workers,
        requested_vars=["x", "y"]
    )
    
    # 內部點數據載入器
    interior_dataloader = GeometryDatapipe(
        geom_objects=[geometry],
        batch_size=1,
        num_points=config.geometry.sampling.interior_points,
        sample_type="volume",
        device=device,
        num_workers=config.geometry.sampling.num_workers,
        requested_vars=["x", "y", "sdf"]
    )
    
    return bc_dataloader, interior_dataloader


def plot_results(solver: DualNetworkPINNSolver, config: DictConfig, 
                epoch: int, loss_dict: Dict, output_dir: str):
    """繪製結果"""
    if not config.outputs.save_plots or epoch % config.training.plot_freq != 0:
        return
    
    # 創建推理網格
    resolution = config.inference.grid_resolution
    domain = config.inference.domain
    
    x = np.linspace(domain[0][0], domain[0][1], resolution)
    y = np.linspace(domain[1][0], domain[1][1], resolution)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    
    coords = torch.from_numpy(
        np.column_stack([xx.ravel(), yy.ravel()])
    ).float().to(solver.device)
    
    with torch.no_grad():
        # 主網路推理
        main_output = solver.main_network(coords)
        u_pred = main_output[:, 0].cpu().numpy().reshape(resolution, resolution)
        v_pred = main_output[:, 1].cpu().numpy().reshape(resolution, resolution)
        p_pred = main_output[:, 2].cpu().numpy().reshape(resolution, resolution)
        
        # 副網路推理
        e_pred = solver.entropy_network(coords).cpu().numpy().reshape(resolution, resolution)
    
    # 創建圖像
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 速度場
    im1 = axes[0, 0].imshow(u_pred, origin="lower", extent=domain[0] + domain[1])
    axes[0, 0].set_title(f"u velocity (epoch {epoch})")
    fig.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(v_pred, origin="lower", extent=domain[0] + domain[1])
    axes[0, 1].set_title(f"v velocity (epoch {epoch})")
    fig.colorbar(im2, ax=axes[0, 1])
    
    # 壓力場
    im3 = axes[0, 2].imshow(p_pred, origin="lower", extent=domain[0] + domain[1])
    axes[0, 2].set_title(f"Pressure (epoch {epoch})")
    fig.colorbar(im3, ax=axes[0, 2])
    
    # 速度幅度
    u_magnitude = np.sqrt(u_pred**2 + v_pred**2)
    im4 = axes[1, 0].imshow(u_magnitude, origin="lower", extent=domain[0] + domain[1])
    axes[1, 0].set_title(f"Velocity Magnitude (epoch {epoch})")
    fig.colorbar(im4, ax=axes[1, 0])
    
    # 熵殘差
    im5 = axes[1, 1].imshow(e_pred, origin="lower", extent=domain[0] + domain[1])
    axes[1, 1].set_title(f"Entropy Residual (epoch {epoch})")
    fig.colorbar(im5, ax=axes[1, 1])
    
    # 流線圖
    axes[1, 2].streamplot(xx, yy, u_pred, v_pred, density=2, color='k', linewidth=0.8)
    axes[1, 2].set_title(f"Streamlines (epoch {epoch})")
    axes[1, 2].set_xlim(domain[0])
    axes[1, 2].set_ylim(domain[1])
    
    # 添加整體標題
    stage_info = solver.current_stage_config['name'] if solver.current_stage_config else "Training"
    fig.suptitle(f"LDC Flow (Re={config.physics.Re}) - {stage_info} - Loss: {loss_dict['total_loss']:.2e}", 
                fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ldc_results_epoch_{epoch:06d}.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()


@hydra.main(version_base="1.3", config_path="configs", config_name="ldc_pinn_advanced")
def advanced_ldc_trainer(cfg: DictConfig) -> None:
    """進階LDC-PINN訓練器 - 完整PhysicsNeMo框架"""
    
    # 初始化分布式管理器
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # 設置日誌
    log = PythonLogger(name="advanced-ldc-physicsnemo")
    if cfg.logging.file_logging:
        log.file_logging()
    
    if dist.rank == 0:
        log.info("🚀 啟動進階PhysicsNeMo LDC-PINN訓練")
        log.info(f"📊 設備: {dist.device}")
        log.info(f"🔧 Reynolds數: {cfg.physics.Re}")
        log.info(f"🎯 總epochs: {cfg.training.max_epochs}")
        log.info(f"🌟 使用進階激活函數: {cfg.model.get('advanced_activation', {}).get('enabled', False)}")
        log.info(f"⚡ 多階段訓練: {len(cfg.training.get('stages', [])) > 0}")
    
    # 創建輸出目錄
    output_dir = cfg.outputs.save_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建求解器
    solver = DualNetworkPINNSolver(cfg, dist.device)
    
    # 創建數據載入器
    bc_dataloader, interior_dataloader = create_data_loaders(cfg, dist.device)
    
    if dist.rank == 0:
        log.info("✅ 初始化完成，開始多階段訓練")
    
    # 訓練循環
    for epoch in range(cfg.training.max_epochs):
        # 取得當前階段配置
        stage_config = solver.training_manager.get_current_stage_config(epoch)
        solver.update_stage(stage_config)
        
        # 判斷是否使用L-BFGS
        use_lbfgs = solver.training_manager.should_switch_to_lbfgs(stage_config)
        
        # 執行訓練步驟
        for bc_data, interior_data in zip(bc_dataloader, interior_dataloader):
            # 準備數據
            bc_coords = torch.cat([bc_data[0]["x"].reshape(-1, 1), 
                                 bc_data[0]["y"].reshape(-1, 1)], dim=1)
            interior_coords = torch.cat([interior_data[0]["x"].reshape(-1, 1), 
                                       interior_data[0]["y"].reshape(-1, 1)], dim=1)
            
            boundary_data = {'coords': bc_coords}
            interior_data = {'coords': interior_coords}
            
            # 執行訓練步驟
            loss_dict = solver.training_step(interior_data, boundary_data, stage_config, use_lbfgs)
            
            break  # 每個epoch只處理一個batch
        
        # 日誌輸出
        if dist.rank == 0 and epoch % cfg.training.log_freq == 0:
            optimizer_info = f"({loss_dict['optimizer']})" if 'optimizer' in loss_dict else ""
            log.info(f"Epoch {epoch:6d}: Loss = {loss_dict['total_loss']:.2e} "
                    f"Stage = {stage_config['name']} {optimizer_info}")
        
        # 繪製結果
        if dist.rank == 0:
            plot_results(solver, cfg, epoch, loss_dict, output_dir)
    
    if dist.rank == 0:
        log.info("🎉 進階多階段訓練完成!")


if __name__ == "__main__":
    advanced_ldc_trainer()
