# SPDX-FileCopyrightText: Copyright (c) 2024 LDC-PINNs
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================
# 進階激活函數模組 - PhysicsNeMo整合版
# Advanced Activation Functions for PhysicsNeMo Framework
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union


class TSAActivation(nn.Module):
    """
    Trainable Sinusoidal Activation (TSA) for PhysicsNeMo
    可訓練正弦激活函數，用於增強PINNs的收斂性和精度
    """
    
    def __init__(self, 
                 num_neurons: int,
                 freq_std: float = 1.0,
                 freq_mean: float = 0.0,
                 trainable_coeffs: bool = True,
                 initialization: str = "uniform"):
        """
        初始化TSA激活函數
        
        Args:
            num_neurons: 神經元數量
            freq_std: 頻率初始化標準差
            freq_mean: 頻率初始化平均值
            trainable_coeffs: 是否訓練正弦/餘弦係數
            initialization: 初始化方法 ("uniform", "normal", "xavier")
        """
        super().__init__()
        
        self.num_neurons = num_neurons
        
        # 初始化頻率參數
        if initialization == "uniform":
            freq_init = torch.rand(num_neurons) * 2 * freq_std - freq_std + freq_mean
        elif initialization == "normal":
            freq_init = torch.normal(freq_mean, freq_std, (num_neurons,))
        elif initialization == "xavier":
            freq_init = torch.randn(num_neurons) * np.sqrt(2.0 / num_neurons)
        else:
            freq_init = torch.ones(num_neurons) * freq_mean
        
        self.freq = nn.Parameter(freq_init)
        
        # 正弦和餘弦係數
        if trainable_coeffs:
            self.c1 = nn.Parameter(torch.ones(1))  # 正弦係數
            self.c2 = nn.Parameter(torch.ones(1))  # 餘弦係數
        else:
            self.register_buffer('c1', torch.ones(1))
            self.register_buffer('c2', torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TSA前向傳播: c1*sin(freq*x) + c2*cos(freq*x)"""
        if x.size(-1) != self.num_neurons:
            raise ValueError(f"輸入維度 {x.size(-1)} != 預期維度 {self.num_neurons}")
        
        freq_x = self.freq * x
        return self.c1 * torch.sin(freq_x) + self.c2 * torch.cos(freq_x)
    
    def get_stats(self) -> Dict[str, float]:
        """取得頻率統計資訊"""
        return {
            'freq_mean': self.freq.mean().item(),
            'freq_std': self.freq.std().item(),
            'freq_min': self.freq.min().item(),
            'freq_max': self.freq.max().item(),
            'c1': self.c1.item(),
            'c2': self.c2.item()
        }


class LAAFActivation(nn.Module):
    """
    Locally Adaptive Activation Function (LAAF)
    局部適應性激活函數，用於多尺度物理問題
    """
    
    def __init__(self, 
                 num_neurons: int, 
                 activation_type: str = "tanh",
                 alpha_init: float = 1.0):
        """
        初始化LAAF激活函數
        
        Args:
            num_neurons: 神經元數量
            activation_type: 基礎激活函數類型 ("tanh", "sin", "swish")
            alpha_init: 可學習參數初始值
        """
        super().__init__()
        
        self.num_neurons = num_neurons
        self.activation_type = activation_type
        
        # 可學習的調節參數
        self.alpha = nn.Parameter(torch.ones(num_neurons) * alpha_init)
        
        # 選擇基礎激活函數
        if activation_type == "tanh":
            self.base_activation = torch.tanh
        elif activation_type == "sin":
            self.base_activation = torch.sin
        elif activation_type == "swish":
            self.base_activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"不支援的激活函數類型: {activation_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAF前向傳播"""
        return self.base_activation(self.alpha * x)
    
    def get_stats(self) -> Dict:
        """取得LAAF統計資訊"""
        return {
            'alpha_mean': self.alpha.mean().item(),
            'alpha_std': self.alpha.std().item(),
            'alpha_min': self.alpha.min().item(),
            'alpha_max': self.alpha.max().item(),
            'activation_type': self.activation_type
        }


class AdaptiveSinusoidalActivation(nn.Module):
    """
    Adaptive Sinusoidal Activation (ASA)
    自適應正弦激活函數，結合TSA和LAAF的優點
    """
    
    def __init__(self, 
                 num_neurons: int,
                 freq_init: float = 1.0,
                 phase_trainable: bool = True):
        """
        初始化ASA激活函數
        
        Args:
            num_neurons: 神經元數量
            freq_init: 頻率初始值
            phase_trainable: 是否訓練相位參數
        """
        super().__init__()
        
        self.num_neurons = num_neurons
        
        # 可學習的頻率和振幅
        self.freq = nn.Parameter(torch.ones(num_neurons) * freq_init)
        self.amplitude = nn.Parameter(torch.ones(num_neurons))
        
        # 可選的相位參數
        if phase_trainable:
            self.phase = nn.Parameter(torch.zeros(num_neurons))
        else:
            self.register_buffer('phase', torch.zeros(num_neurons))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ASA前向傳播"""
        return self.amplitude * torch.sin(self.freq * x + self.phase)


def compute_activation_regularization(model: nn.Module, 
                                    weight: float = 0.01,
                                    activation_types: Optional[List] = None) -> torch.Tensor:
    """
    計算激活函數的正則化損失
    
    Args:
        model: 包含進階激活函數的模型
        weight: 正則化權重
        activation_types: 要正則化的激活函數類型列表
    
    Returns:
        正則化損失
    """
    if activation_types is None:
        activation_types = [TSAActivation, LAAFActivation, AdaptiveSinusoidalActivation]
    
    device = next(model.parameters()).device
    total_reg = torch.tensor(0.0, device=device)
    
    for module in model.modules():
        if any(isinstance(module, act_type) for act_type in activation_types):
            if isinstance(module, TSAActivation):
                # TSA正則化：控制頻率範圍
                freq_reg = torch.mean(torch.abs(module.freq))
                total_reg += freq_reg
            
            elif isinstance(module, LAAFActivation):
                # LAAF正則化：避免參數過大
                alpha_reg = torch.mean(torch.abs(module.alpha - 1.0))
                total_reg += alpha_reg
            
            elif isinstance(module, AdaptiveSinusoidalActivation):
                # ASA正則化：頻率和振幅平衡
                freq_reg = torch.mean(torch.abs(module.freq - 1.0))
                amp_reg = torch.mean(torch.abs(module.amplitude - 1.0))
                total_reg += freq_reg + amp_reg
    
    return weight * total_reg


def get_activation_function(activation_config: Dict) -> nn.Module:
    """
    根據配置創建激活函數
    
    Args:
        activation_config: 激活函數配置字典
    
    Returns:
        激活函數模組
    """
    activation_type = activation_config.get('type', 'SiLU').lower()
    num_neurons = activation_config.get('num_neurons', 512)
    
    if activation_type == 'tsa':
        return TSAActivation(
            num_neurons=num_neurons,
            freq_std=activation_config.get('freq_std', 1.0),
            trainable_coeffs=activation_config.get('trainable_coeffs', True),
            initialization=activation_config.get('initialization', 'uniform')
        )
    
    elif activation_type == 'laaf':
        return LAAFActivation(
            num_neurons=num_neurons,
            activation_type=activation_config.get('base_type', 'tanh'),
            alpha_init=activation_config.get('alpha_init', 1.0)
        )
    
    elif activation_type == 'asa':
        return AdaptiveSinusoidalActivation(
            num_neurons=num_neurons,
            freq_init=activation_config.get('freq_init', 1.0),
            phase_trainable=activation_config.get('phase_trainable', True)
        )
    
    elif activation_type == 'silu' or activation_type == 'swish':
        return nn.SiLU()
    
    elif activation_type == 'tanh':
        return nn.Tanh()
    
    elif activation_type == 'relu':
        return nn.ReLU()
    
    elif activation_type == 'gelu':
        return nn.GELU()
    
    else:
        raise ValueError(f"不支援的激活函數類型: {activation_type}")


class ActivationMonitor:
    """激活函數監控器，用於追蹤訓練過程中激活函數的變化"""
    
    def __init__(self):
        self.history = []
    
    def log_stats(self, model: nn.Module, epoch: int):
        """記錄激活函數統計"""
        stats = {
            'epoch': epoch,
            'activations': {}
        }
        
        for name, module in model.named_modules():
            if hasattr(module, 'get_stats'):
                stats['activations'][name] = module.get_stats()
        
        self.history.append(stats)
    
    def get_latest_stats(self) -> Dict:
        """取得最新的統計資訊"""
        return self.history[-1] if self.history else {}
    
    def print_summary(self):
        """列印激活函數統計摘要"""
        if not self.history:
            print("❌ 無激活函數統計資料")
            return
        
        latest = self.history[-1]
        print(f"📊 激活函數統計 (Epoch {latest['epoch']})")
        print("=" * 50)
        
        for name, stats in latest['activations'].items():
            print(f"🔧 {name}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            print()