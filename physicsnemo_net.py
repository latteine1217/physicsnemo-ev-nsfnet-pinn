# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import torch
import torch.nn as nn
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.models.activations import Activation
from physicsnemo.models.model import Model
from typing import List, Dict, Optional


class PhysicsNeMoPINNNet(Model):
    """PhysicsNeMo 標準 PINN 神經網路"""
    
    def __init__(
        self,
        input_keys: List[str] = ["x", "y"],
        output_keys: List[str] = ["u", "v", "p"],
        nr_layers: int = 6,
        layer_size: int = 80,
        activation_fn: str = "tanh",
        **kwargs
    ):
        super().__init__()
        
        self.input_keys = input_keys
        self.output_keys = output_keys
        
        # 使用正確的 PhysicsNeMo FullyConnected API
        self.net = FullyConnected(
            input_dim=len(input_keys),
            output_dim=len(output_keys),
            nr_layers=nr_layers,
            layer_size=layer_size,
            activation_fn=activation_fn,
            adaptive_activations=False,
            weight_norm=True,
        )
        
    def forward(self, input_vars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """PhysicsNeMo 標準前向傳播"""
        # 按順序拼接輸入
        x = torch.cat([input_vars[key] for key in self.input_keys], dim=-1)
        
        # 神經網路前向傳播
        output = self.net(x)
        
        # 分割輸出為字典格式
        output_dict = {}
        for i, key in enumerate(self.output_keys):
            output_dict[key] = output[:, i:i+1]
            
        return output_dict


class PhysicsNeMoEVMNet(Model):
    """PhysicsNeMo EVM (渦黏度) 網路"""
    
    def __init__(
        self,
        input_keys: List[str] = ["x", "y"],
        output_keys: List[str] = ["nu_t"],
        nr_layers: int = 6,
        layer_size: int = 40,
        activation_fn: str = "tanh",
        **kwargs
    ):
        super().__init__()
        
        self.input_keys = input_keys
        self.output_keys = output_keys
        
        # EVM 網路使用較小的架構
        self.evm_net = FullyConnected(
            input_dim=len(input_keys),
            output_dim=len(output_keys),
            nr_layers=nr_layers,
            layer_size=layer_size,
            activation_fn=activation_fn,
            adaptive_activations=False,
            weight_norm=True,
        )
        
    def forward(self, input_vars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """EVM 網路前向傳播"""
        # 拼接輸入
        x = torch.cat([input_vars[key] for key in self.input_keys], dim=-1)
        
        # EVM 網路前向傳播
        output = self.evm_net(x)
        
        # 分割輸出 - 應用絕對值確保正的渦黏度
        output_dict = {}
        for i, key in enumerate(self.output_keys):
            output_dict[key] = torch.abs(output[:, i:i+1])
            
        return output_dict


class PhysicsNeMoCombinedPINNModel(Model):
    """PhysicsNeMo 結合 PINN + EVM 模型"""
    
    def __init__(
        self,
        pinn_config: Dict,
        evm_config: Dict,
        **kwargs
    ):
        super().__init__()
        
        # 主 PINN 網路
        self.pinn_net = PhysicsNeMoPINNNet(**pinn_config)
        
        # EVM 網路
        self.evm_net = PhysicsNeMoEVMNet(**evm_config)
        
        # 結合所有輸入和輸出鍵
        self.input_keys = list(set(self.pinn_net.input_keys + self.evm_net.input_keys))
        self.output_keys = self.pinn_net.output_keys + self.evm_net.output_keys
        
    def forward(self, input_vars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """結合模型前向傳播"""
        # PINN 網路預測
        pinn_output = self.pinn_net(input_vars)
        
        # EVM 網路預測
        evm_output = self.evm_net(input_vars)
        
        # 結合輸出
        combined_output = {**pinn_output, **evm_output}
        
        return combined_output
    
    def get_pinn_output(self, input_vars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """僅獲取 PINN 網路輸出"""
        return self.pinn_net(input_vars)
    
    def get_evm_output(self, input_vars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """僅獲取 EVM 網路輸出"""
        return self.evm_net(input_vars)
    
    def freeze_evm(self):
        """凍結 EVM 網路參數"""
        for param in self.evm_net.parameters():
            param.requires_grad = False
    
    def unfreeze_evm(self):
        """解凍 EVM 網路參數"""
        for param in self.evm_net.parameters():
            param.requires_grad = True