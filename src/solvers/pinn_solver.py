"""
PINN求解器核心模組

整合ConfigManager、FCNet、物理方程，實現完整的PINN訓練流程
基於ev-NSFnet的成熟實現
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

from ..config.config_manager import ConfigManager
from ..models.networks import FCNet
from ..models.activations import LAAFActivation
from ..physics.equations import PhysicsEquations
from ..utils.device_utils import setup_device
from ..utils.logger import LoggerFactory, PINNLogger


warnings.filterwarnings("ignore", message=".*c10d::allreduce_.*autograd kernel.*")


class PINNSolver:
    """
    Physics-Informed Neural Network 求解器
    
    整合所有PINN組件：
    - 主網路（u, v, p預測）
    - EVM網路（entropy預測）  
    - 物理方程計算
    - 訓練循環和優化
    """
    
    def __init__(self, config_path: str):
        """
        初始化PINN求解器
        
        Args:
            config_path: 配置檔案路徑
        """
        # 載入配置
        self.config = ConfigManager.load_config(config_path)
        
        # 分散式訓練設定
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # 初始化logger
        self.logger = LoggerFactory.get_logger(
            name=f"PINN_Re{self.config.physics.reynolds_number}",
            level="INFO",
            rank=self.rank
        )
        
        # 設備設定
        self.device = setup_device(self.local_rank, self.logger)
        
        # 基本參數
        self.Re = self.config.physics.reynolds_number
        self.alpha_evm = self.config.physics.alpha_evm
        self.beta = getattr(self.config.physics, 'beta', 1.0)
        
        # 網路架構參數
        self.layers = self.config.network.layers
        self.hidden_size = self.config.network.hidden_size
        self.layers_1 = self.config.network.layers_1
        self.hidden_size_1 = self.config.network.hidden_size_1
        
        # 訓練參數
        self.N_f = self.config.training.N_f
        self.batch_size = getattr(self.config.training, 'batch_size', self.N_f)
        self.checkpoint_freq = getattr(self.config.training, 'checkpoint_freq', 2000)
        
        # 損失權重
        self.alpha_b = getattr(self.config.training, 'bc_weight', 10.0)
        self.alpha_e = getattr(self.config.training, 'eq_weight', 1.0)
        self.alpha_i = getattr(self.config.training, 'ic_weight', 0.1)
        self.alpha_o = getattr(self.config.training, 'outlet_weight', 1.0)
        self.alpha_s = getattr(self.config.training, 'supervised_data_weight', 1.0)
        
        # 初始化損失
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0
        
        # TensorBoard設定
        self._setup_tensorboard()
        
        # 時間追蹤
        self.epoch_start_time = None
        self.epoch_times = []
        self.stage_start_time = None
        self.training_start_time = None
        self.global_step_offset = 0
        
        # 當前階段
        self.current_stage = ' '
        
        # 初始化神經網路
        self._initialize_networks()
        
        # 初始化物理方程
        self.physics_equations = PhysicsEquations(
            reynolds_number=self.Re,
            alpha_evm=self.alpha_evm,
            beta=self.beta
        )
        
        # 訓練數據（將在load_training_data中設定）
        self.x_f = self.y_f = None  # 方程點
        self.x_b = self.y_b = self.u_b = self.v_b = None  # 邊界條件點
        self.x_i = self.y_i = self.u_i = self.v_i = None  # 初始條件點
        self.x_o = self.y_o = None  # 出口點
        
        # 優化器
        self.opt = None
        self.opt_1 = None
        
        # 輸出初始化訊息
        self._log_initialization_info()
    
    def _setup_tensorboard(self):
        """設定TensorBoard"""
        tb_enabled = getattr(self.config.system, 'tensorboard_enabled', True)
        self.tb_interval = getattr(self.config.system, 'tensorboard_interval', 1000)
        
        if self.rank == 0 and tb_enabled:
            log_dir = f"runs/PINN_Re{self.Re}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            self.logger.info(f"📊 TensorBoard log directory: {log_dir} (interval={self.tb_interval})")
        else:
            self.tb_writer = None
    
    def _initialize_networks(self):
        """初始化神經網路"""
        # 主網路（u, v, p）
        self.net = self._create_network(
            num_ins=2, num_outs=3, 
            num_layers=self.layers, 
            hidden_size=self.hidden_size,
            is_evm=False
        ).to(self.device)
        
        # EVM網路（entropy）
        self.net_1 = self._create_network(
            num_ins=2, num_outs=1,
            num_layers=self.layers_1,
            hidden_size=self.hidden_size_1,
            is_evm=True
        ).to(self.device)
        
        # 確保float32精度
        self.net = self.net.float()
        self.net_1 = self.net_1.float()
        
        # 套用配置後處理
        self._apply_config_post_init()
        
        # DDP包裝（如果需要）
        if self.world_size > 1:
            ddp_broadcast_buffers = getattr(self.config.system, 'ddp_broadcast_buffers', False)
            
            self.net = DDP(
                self.net,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=ddp_broadcast_buffers,
                gradient_as_bucket_view=True
            )
            
            self.net_1 = DDP(
                self.net_1,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=ddp_broadcast_buffers,
                gradient_as_bucket_view=True
            )
    
    def _create_network(self, num_ins: int, num_outs: int, num_layers: int, 
                       hidden_size: int, is_evm: bool = False) -> FCNet:
        """創建神經網路"""
        # 選擇激活函數
        if is_evm:
            activation_name = getattr(self.config.network, 'activation_evm', 'tanh').lower()
        else:
            activation_name = getattr(self.config.network, 'activation_main', 'laaf').lower()
        
        if activation_name == 'laaf':
            init_scale = getattr(self.config.network, 'laaf_init_scale', 1.0)
            max_scale = getattr(self.config.network, 'laaf_max_scale', 20.0)
            
            def activation_factory():
                return LAAFActivation(init_scale=init_scale, max_scale=max_scale)
        else:
            activation_factory = torch.nn.Tanh
        
        return FCNet(
            num_ins=num_ins,
            num_outs=num_outs,
            num_layers=num_layers,
            hidden_size=hidden_size,
            activation=activation_factory
        )
    
    def _apply_config_post_init(self):
        """套用配置相關的後處理"""
        # 層權重縮放
        main_first = getattr(self.config.network, 'first_layer_scale_main', 2.0)
        main_last = getattr(self.config.network, 'last_layer_scale_main', 0.5)
        evm_first = getattr(self.config.network, 'first_layer_scale_evm', 1.2)
        evm_last = getattr(self.config.network, 'last_layer_scale_evm', 0.1)
        
        self._apply_layer_scales(self._get_model(self.net), main_first, main_last)
        self._apply_layer_scales(self._get_model(self.net_1), evm_first, evm_last)
    
    def _apply_layer_scales(self, model: nn.Module, first_scale: float, last_scale: float):
        """套用層權重縮放"""
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            return
        
        # 首層縮放
        linear_layers[0].weight.data.mul_(float(first_scale))
        # 末層縮放
        linear_layers[-1].weight.data.mul_(float(last_scale))
    
    def _get_model(self, model: Union[FCNet, DDP]) -> FCNet:
        """獲取底層模型（處理DDP包裝）"""
        if isinstance(model, DDP):
            return model.module
        else:
            return model
    
    def _log_initialization_info(self):
        """記錄初始化訊息"""
        config_info = {
            "Reynolds數": self.Re,
            "主網路": f"{self.layers} 層 × {self.hidden_size} 神經元",
            "EVM網路": f"{self.layers_1} 層 × {self.hidden_size_1} 神經元",
            "訓練點數": f"{self.N_f:,}",
            "設備": str(self.device),
            "批次大小": "全批次" if self.batch_size == self.N_f else str(self.batch_size)
        }
        
        for key, value in config_info.items():
            self.logger.info(f"{key}: {value}")
    
    def load_training_data(self, x_f: torch.Tensor, y_f: torch.Tensor,
                          x_b: torch.Tensor, y_b: torch.Tensor,
                          u_b: torch.Tensor, v_b: torch.Tensor,
                          x_i: Optional[torch.Tensor] = None, y_i: Optional[torch.Tensor] = None,
                          u_i: Optional[torch.Tensor] = None, v_i: Optional[torch.Tensor] = None,
                          x_o: Optional[torch.Tensor] = None, y_o: Optional[torch.Tensor] = None):
        """
        載入訓練數據
        
        Args:
            x_f, y_f: 方程點座標
            x_b, y_b: 邊界點座標
            u_b, v_b: 邊界條件值
            x_i, y_i: 初始條件點座標（可選）
            u_i, v_i: 初始條件值（可選）
            x_o, y_o: 出口點座標（可選）
        """
        # 方程點
        self.x_f = x_f.to(self.device).requires_grad_(True)
        self.y_f = y_f.to(self.device).requires_grad_(True)
        
        # 邊界條件
        self.x_b = x_b.to(self.device).requires_grad_(True)
        self.y_b = y_b.to(self.device).requires_grad_(True)
        self.u_b = u_b.to(self.device)
        self.v_b = v_b.to(self.device)
        
        # 初始條件（如果提供）
        if x_i is not None:
            self.x_i = x_i.to(self.device).requires_grad_(True)
            self.y_i = y_i.to(self.device).requires_grad_(True)
            self.u_i = u_i.to(self.device)
            self.v_i = v_i.to(self.device)
        
        # 出口條件（如果提供）
        if x_o is not None:
            self.x_o = x_o.to(self.device).requires_grad_(True)
            self.y_o = y_o.to(self.device).requires_grad_(True)
        
        self.logger.info(f"已載入訓練數據:")
        self.logger.info(f"  方程點: {self.x_f.shape[0]}")
        self.logger.info(f"  邊界點: {self.x_b.shape[0]}")
        if self.x_i is not None:
            self.logger.info(f"  初始點: {self.x_i.shape[0]}")
        if self.x_o is not None:
            self.logger.info(f"  出口點: {self.x_o.shape[0]}")
    
    def neural_net_u(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        神經網路前向傳播
        
        Args:
            x, y: 輸入座標
            
        Returns:
            (u, v, p, e): 速度、壓力、entropy預測
        """
        X = torch.cat((x, y), dim=1).to(self.device)
        
        with torch.enable_grad():
            uvp = self.net(X)
            ee = self.net_1(X)
        
        u = uvp[:, 0:1]
        v = uvp[:, 1:2] 
        p = uvp[:, 2:3]
        e = ee[:, 0:1]
        
        return u, v, p, e
    
    def compute_physics_loss(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        計算物理損失
        
        Returns:
            (total_loss, individual_losses): 總損失和各項損失列表
        """
        losses = []
        
        # 邊界條件損失
        if self.x_b.shape[0] > 0:
            u_pred_b, v_pred_b, _, _ = self.neural_net_u(self.x_b, self.y_b)
            
            u_b_flat = self.u_b.view(-1)
            v_b_flat = self.v_b.view(-1)
            u_pred_b_flat = u_pred_b.view(-1)
            v_pred_b_flat = v_pred_b.view(-1)
            
            self.loss_b = torch.mean(torch.square(u_b_flat - u_pred_b_flat)) + \
                         torch.mean(torch.square(v_b_flat - v_pred_b_flat))
        else:
            self.loss_b = torch.tensor(0.0, device=self.device)
        
        # 方程損失
        u_pred_f, v_pred_f, p_pred_f, e_pred_f = self.neural_net_u(self.x_f, self.y_f)
        
        eq1, eq2, eq3, eq4 = self.physics_equations.compute_physics_residuals(
            self.x_f, self.y_f, u_pred_f, v_pred_f, p_pred_f, e_pred_f
        )
        
        # 各方程損失
        self.loss_eq1 = torch.mean(torch.square(eq1))
        self.loss_eq2 = torch.mean(torch.square(eq2))
        self.loss_eq3 = torch.mean(torch.square(eq3))
        self.loss_eq4 = torch.mean(torch.square(eq4))
        
        self.loss_e = self.loss_eq1 + self.loss_eq2 + self.loss_eq3 + 0.1 * self.loss_eq4
        
        # 初始條件損失
        if self.x_i is not None and self.x_i.shape[0] > 0:
            u_pred_i, v_pred_i, _, _ = self.neural_net_u(self.x_i, self.y_i)
            self.loss_i = torch.mean(torch.square(self.u_i - u_pred_i)) + \
                         torch.mean(torch.square(self.v_i - v_pred_i))
        else:
            self.loss_i = torch.tensor(0.0, device=self.device)
        
        # 出口條件損失
        if self.x_o is not None and self.x_o.shape[0] > 0:
            _, _, p_pred_o, _ = self.neural_net_u(self.x_o, self.y_o)
            self.loss_o = torch.mean(torch.square(p_pred_o))  # 出口壓力=0
        else:
            self.loss_o = torch.tensor(0.0, device=self.device)
        
        # 監督數據損失（暫時設為0）
        self.loss_s = torch.tensor(0.0, device=self.device)
        
        # 組合損失
        total_loss = (self.alpha_e * self.loss_e + 
                     self.alpha_b * self.loss_b + 
                     self.alpha_i * self.loss_i + 
                     self.alpha_o * self.loss_o + 
                     self.alpha_s * self.loss_s)
        
        # 損失列表
        losses = [
            self.loss_e,      # 0: 方程損失
            self.loss_b,      # 1: 邊界損失
            self.loss_s,      # 2: 監督損失
            self.loss_eq1,    # 3: X動量方程
            self.loss_eq2,    # 4: Y動量方程
            self.loss_eq3,    # 5: 連續性方程
            self.loss_eq4     # 6: entropy residual
        ]
        
        return total_loss, losses
    
    def setup_optimizer(self, learning_rate: float = 1e-3, weight_decay: float = 0.0):
        """設定優化器"""
        # 獲取模型參數
        main_params = list(self._get_model(self.net).parameters())
        evm_params = list(self._get_model(self.net_1).parameters())
        all_params = main_params + evm_params
        
        # 創建AdamW優化器
        if weight_decay > 0:
            # 分組：有weight decay和無weight decay
            decay_params = []
            no_decay_params = []
            
            for param in all_params:
                if param.requires_grad:
                    if len(param.shape) >= 2:  # 權重矩陣
                        decay_params.append(param)
                    else:  # 偏置項
                        no_decay_params.append(param)
            
            param_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
        else:
            param_groups = [{'params': all_params, 'weight_decay': 0.0}]
        
        self.opt = torch.optim.AdamW(param_groups, lr=learning_rate)
        
        # 設定初始學習率
        for group in self.opt.param_groups:
            group['initial_lr'] = learning_rate
        
        self.logger.info(f"已設定優化器: lr={learning_rate}, weight_decay={weight_decay}")
    
    def train_epoch(self) -> Tuple[float, List[float]]:
        """
        訓練一個epoch
        
        Returns:
            (loss_value, losses_list): 損失值和各項損失列表
        """
        self.opt.zero_grad()
        
        # 計算損失
        loss, losses = self.compute_physics_loss()
        
        # 反向傳播
        loss.backward()
        
        # 梯度裁剪（可選）
        max_grad_norm = getattr(self.config.training, 'max_grad_norm', None)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self._get_model(self.net).parameters()) + 
                list(self._get_model(self.net_1).parameters()),
                max_grad_norm
            )
        
        # 優化器步進
        self.opt.step()
        
        return loss.item(), [l.item() for l in losses]
    
    def log_tensorboard(self, epoch: int, loss_value: float, losses: List[float]):
        """記錄TensorBoard"""
        if self.tb_writer is None or epoch % self.tb_interval != 0:
            return
        
        global_step = self.global_step_offset + epoch
        
        # 主要損失
        self.tb_writer.add_scalar('Loss/Total', loss_value, global_step)
        self.tb_writer.add_scalar('Loss/Equation_Combined', losses[0], global_step)
        self.tb_writer.add_scalar('Loss/Boundary', losses[1], global_step)
        self.tb_writer.add_scalar('Loss/Supervised', losses[2], global_step)
        self.tb_writer.add_scalar('Loss/Equation_NS_X', losses[3], global_step)
        self.tb_writer.add_scalar('Loss/Equation_NS_Y', losses[4], global_step)
        self.tb_writer.add_scalar('Loss/Equation_Continuity', losses[5], global_step)
        self.tb_writer.add_scalar('Loss/Equation_EntropyResidual', losses[6], global_step)
        
        # 學習率
        self.tb_writer.add_scalar('Training/LearningRate', self.opt.param_groups[0]['lr'], global_step)
    
    def save_checkpoint(self, epoch: int, checkpoint_dir: str):
        """保存檢查點"""
        if self.rank != 0:
            return
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        checkpoint = {
            'epoch': epoch,
            'net_state_dict': self._get_model(self.net).state_dict(),
            'net_1_state_dict': self._get_model(self.net_1).state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'Re': self.Re,
            'alpha_evm': self.alpha_evm,
            'current_stage': self.current_stage,
            'global_step_offset': self.global_step_offset
        }
        
        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"✅ 已保存檢查點: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"保存檢查點失敗 {checkpoint_path}: {e}")
    
    def update_evm_parameters(self, alpha_evm: float):
        """更新EVM參數"""
        self.alpha_evm = alpha_evm
        self.physics_equations.update_evm_parameters(alpha_evm)
        self.logger.info(f"已更新alpha_evm: {alpha_evm}")