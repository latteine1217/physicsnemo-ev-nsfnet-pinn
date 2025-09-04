"""
配置管理系統 - 統一管理PINN訓練參數
基於ev-NSFnet/config.py，適配新的模組化架構
"""

import os
import json
import yaml
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict, Any, Union
import torch

# 常量定義
RESULTS_PATH = "results"

@dataclass
class NetworkConfig:
    """神經網路架構配置"""
    # 主網路配置
    main_net_layers: int = 6
    main_net_hidden_size: int = 80
    main_net_activation: str = "tanh"  # tanh | laaf | relu
    main_net_initialization: str = "xavier"
    main_net_first_layer_scale: float = 2.0
    main_net_last_layer_scale: float = 0.5
    
    # EVM網路配置
    evm_net_layers: int = 4
    evm_net_hidden_size: int = 40
    evm_net_activation: str = "tanh"
    evm_net_output_activation: str = "abs_cap"  # abs_cap | softplus_cap
    evm_net_first_layer_scale: float = 1.2
    evm_net_last_layer_scale: float = 0.1
    
    # LAAF激活函數參數
    laaf_init_scale: float = 1.0
    laaf_max_scale: float = 20.0
    laaf_reg_lambda: float = 0.0

@dataclass
class PhysicsConfig:
    """物理參數配置"""
    reynolds_number: float = 3000.0
    
    # 計算域
    domain_x_min: float = 0.0
    domain_x_max: float = 1.0
    domain_y_min: float = 0.0
    domain_y_max: float = 1.0
    
    # EVM參數
    alpha_evm: float = 0.05
    beta: float = 20.0
    
    # 損失函數權重
    boundary_weight: float = 10.0
    equation_weight: float = 1.0
    supervision_weight: float = 5.0

@dataclass
class TrainingStage:
    """單個訓練階段配置"""
    name: str
    epochs: int
    alpha_evm: float
    learning_rate: float
    scheduler: str = "Constant"
    weight_decay: float = 1e-5

@dataclass
class SamplingConfig:
    """採樣配置"""
    n_interior: int = 120000
    n_boundary: int = 4000
    strategy: str = "uniform"  # uniform | adaptive
    sort_by_boundary_distance: bool = False
    pde_distance_weighting: bool = True
    pde_distance_w_min: float = 0.2
    pde_distance_tau: float = 0.2

@dataclass
class LBFGSConfig:
    """L-BFGS優化器配置"""
    enabled: bool = True
    enabled_in_distributed: bool = False
    enable_from_stage: int = 3
    
    # 觸發條件
    trigger_window_per_stage: List[int] = field(default_factory=lambda: [5000, 7500, 10000])
    min_improve_pct_per_stage: List[float] = field(default_factory=lambda: [0.02, 0.015, 0.01])
    ema_gamma: float = 0.95
    
    # 梯度條件
    use_simple_grad_check: bool = True
    grad_median_abs_thresh: float = 0.002
    grad_relative_factor: float = 0.02
    grad_cos_ema_thresh: float = 0.9
    
    # 物理條件
    alpha_evm_threshold: float = 0.02
    cap_ratio_threshold: float = 0.7
    
    # L-BFGS參數
    cooldown_steps: int = 5000
    freeze_evm_during_lbfgs: bool = True
    max_outer_steps: int = 50
    timeout_seconds: int = 300
    max_iter: int = 10
    history_size: int = 20
    tolerance_grad: float = 1e-6
    tolerance_change: float = 1e-8
    line_search_fn: str = "strong_wolfe"
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    checkpoint_before_lbfgs: bool = True

@dataclass
class OptimizationConfig:
    """優化器配置"""
    # Adam參數
    adam_lr: float = 1e-3
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    
    # L-BFGS配置
    lbfgs: LBFGSConfig = field(default_factory=LBFGSConfig)

@dataclass
class TrainingConfig:
    """訓練配置"""
    total_epochs: int = 1000000
    batch_size: Optional[int] = None
    
    # 多階段訓練
    stages: List[TrainingStage] = field(default_factory=lambda: [
        TrainingStage("Stage_1_Initial", 200000, 0.05, 1e-3, "Constant", 1e-5),
        TrainingStage("Stage_2_Reduce", 200000, 0.01, 2e-4, "Constant", 5e-6),
        TrainingStage("Stage_3_Finetune", 200000, 0.005, 4e-5, "Constant", 5e-6),
        TrainingStage("Stage_4_Precision", 200000, 0.002, 1e-5, "Constant", 2e-6),
        TrainingStage("Stage_5_Converge", 200000, 0.001, 2e-6, "Constant", 0.0)
    ])
    
    # 採樣配置
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    # 優化器配置
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

@dataclass
class SupervisionConfig:
    """監督學習配置"""
    enabled: bool = True
    data_points: int = 1
    data_path: str = "data/reference/cavity_Re3000_256_Uniform.mat"
    weight: float = 5.0
    random_seed: int = 42

@dataclass
class SystemConfig:
    """系統配置"""
    device: str = "auto"
    precision: str = "float32"
    distributed: bool = False
    num_workers: int = 4
    
    # Tesla P100兼容性
    p100_compatibility: Dict[str, Any] = field(default_factory=lambda: {
        "torch_compile_backend": "eager",
        "torchdynamo_disable": True
    })
    
    # 性能配置
    memory_limit_gb: float = 14.0
    gradient_clip_norm: float = 1.0
    memory_cleanup_freq: int = 100
    epoch_times_limit: int = 1000
    ddp_broadcast_buffers: bool = False
    
    # 日誌配置
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "log_freq": 100,
        "log_tips": True,
        "tensorboard_enabled": True,
        "tensorboard_interval": 1000,
        "timing_sync_interval": 1000
    })
    
    # 檢查點配置
    checkpoints: Dict[str, Any] = field(default_factory=lambda: {
        "save_freq": 10000,
        "keep_best": 5,
        "auto_save": True
    })

@dataclass
class ExperimentConfig:
    """實驗配置"""
    name: str = "LDC_PINN_Default"
    description: str = "Physics-Informed Neural Network for Lid-Driven Cavity Flow"
    seed: int = 42

class ConfigManager:
    """統一配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置檔案路徑，若為None則使用預設配置
        """
        # 初始化各個配置組件
        self.experiment = ExperimentConfig()
        self.network = NetworkConfig()
        self.physics = PhysicsConfig()
        self.training = TrainingConfig()
        self.supervision = SupervisionConfig()
        self.system = SystemConfig()
        
        # 載入配置檔案
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # 設定P100相容性
        self._setup_p100_compatibility()
    
    def load_config(self, config_path: str) -> None:
        """
        從YAML檔案載入配置
        
        Args:
            config_path: 配置檔案路徑
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 更新各個配置組件
            if 'experiment' in config_data:
                self._update_dataclass(self.experiment, config_data['experiment'])
            
            if 'network' in config_data:
                self._update_network_config(config_data['network'])
            
            if 'physics' in config_data:
                self._update_dataclass(self.physics, config_data['physics'])
            
            if 'training' in config_data:
                self._update_training_config(config_data['training'])
            
            if 'supervision' in config_data:
                self._update_dataclass(self.supervision, config_data['supervision'])
            
            if 'system' in config_data:
                self._update_dataclass(self.system, config_data['system'])
                
            print(f"✅ 配置檔案載入成功: {config_path}")
            
        except Exception as e:
            print(f"❌ 配置檔案載入失敗: {e}")
            raise
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """更新dataclass物件"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _update_network_config(self, data: Dict[str, Any]) -> None:
        """更新網路配置"""
        # 處理嵌套的main_net和evm_net配置
        if 'main_net' in data:
            main_data = data['main_net']
            for key, value in main_data.items():
                attr_name = f"main_net_{key}"
                if hasattr(self.network, attr_name):
                    setattr(self.network, attr_name, value)
        
        if 'evm_net' in data:
            evm_data = data['evm_net']
            for key, value in evm_data.items():
                attr_name = f"evm_net_{key}"
                if hasattr(self.network, attr_name):
                    setattr(self.network, attr_name, value)
        
        if 'laaf' in data:
            laaf_data = data['laaf']
            for key, value in laaf_data.items():
                attr_name = f"laaf_{key}"
                if hasattr(self.network, attr_name):
                    setattr(self.network, attr_name, value)
    
    def _update_training_config(self, data: Dict[str, Any]) -> None:
        """更新訓練配置"""
        # 更新基本配置
        for key, value in data.items():
            if key not in ['stages', 'sampling', 'optimization'] and hasattr(self.training, key):
                setattr(self.training, key, value)
        
        # 更新階段配置
        if 'stages' in data:
            stages = []
            for stage_data in data['stages']:
                stage = TrainingStage(**stage_data)
                stages.append(stage)
            self.training.stages = stages
        
        # 更新採樣配置
        if 'sampling' in data:
            self._update_dataclass(self.training.sampling, data['sampling'])
        
        # 更新優化器配置
        if 'optimization' in data:
            opt_data = data['optimization']
            if 'adam' in opt_data:
                for key, value in opt_data['adam'].items():
                    attr_name = f"adam_{key}"
                    if hasattr(self.training.optimization, attr_name):
                        setattr(self.training.optimization, attr_name, value)
            
            if 'lbfgs' in opt_data:
                self._update_dataclass(self.training.optimization.lbfgs, opt_data['lbfgs'])
    
    def _setup_p100_compatibility(self) -> None:
        """設定Tesla P100相容性"""
        if self.system.p100_compatibility.get('torchdynamo_disable', True):
            os.environ['TORCHDYNAMO_DISABLE'] = '1'
        
        backend = self.system.p100_compatibility.get('torch_compile_backend', 'eager')
        os.environ['TORCH_COMPILE_BACKEND'] = backend
    
    def get_stage_config(self, stage_index: int) -> TrainingStage:
        """
        獲取指定階段的配置
        
        Args:
            stage_index: 階段索引 (0-based)
            
        Returns:
            TrainingStage: 階段配置
        """
        if 0 <= stage_index < len(self.training.stages):
            return self.training.stages[stage_index]
        else:
            raise ValueError(f"階段索引 {stage_index} 超出範圍 [0, {len(self.training.stages)-1}]")
    
    def get_current_alpha_evm(self, stage_index: int) -> float:
        """獲取當前階段的alpha_evm值"""
        return self.get_stage_config(stage_index).alpha_evm
    
    def get_device(self) -> torch.device:
        """獲取計算設備"""
        if self.system.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.system.device)
    
    def get_precision_dtype(self) -> torch.dtype:
        """獲取數值精度類型"""
        if self.system.precision == "float16":
            return torch.float16
        elif self.system.precision == "float64":
            return torch.float64
        else:
            return torch.float32
    
    def save_config(self, output_path: str) -> None:
        """
        保存當前配置到YAML檔案
        
        Args:
            output_path: 輸出檔案路徑
        """
        config_dict = {
            'experiment': asdict(self.experiment),
            'network': asdict(self.network),
            'physics': asdict(self.physics),
            'training': asdict(self.training),
            'supervision': asdict(self.supervision),
            'system': asdict(self.system)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 配置已保存到: {output_path}")
    
    def validate_config(self) -> bool:
        """驗證配置的有效性"""
        try:
            # 驗證階段配置
            if not self.training.stages:
                raise ValueError("訓練階段配置不能為空")
            
            # 驗證Reynolds數
            if self.physics.reynolds_number <= 0:
                raise ValueError("Reynolds數必須為正數")
            
            # 驗證採樣點數
            if self.training.sampling.n_interior <= 0:
                raise ValueError("內部採樣點數必須為正數")
            
            # 驗證L-BFGS配置
            lbfgs = self.training.optimization.lbfgs
            if lbfgs.enabled:
                if lbfgs.enable_from_stage >= len(self.training.stages):
                    raise ValueError("L-BFGS啟用階段超出訓練階段範圍")
            
            print("✅ 配置驗證通過")
            return True
            
        except Exception as e:
            print(f"❌ 配置驗證失敗: {e}")
            return False
    
    def __str__(self) -> str:
        """配置摘要"""
        return f"""
=== LDC-PINNs 配置摘要 ===
實驗: {self.experiment.name}
Reynolds數: {self.physics.reynolds_number}
網路架構: {self.network.main_net_layers}×{self.network.main_net_hidden_size} + {self.network.evm_net_layers}×{self.network.evm_net_hidden_size}
訓練階段: {len(self.training.stages)}階段
總訓練步數: {sum(stage.epochs for stage in self.training.stages)}
採樣點數: {self.training.sampling.n_interior} (內部) + {self.training.sampling.n_boundary} (邊界)
設備: {self.system.device}
精度: {self.system.precision}
L-BFGS: {'啟用' if self.training.optimization.lbfgs.enabled else '停用'}
"""