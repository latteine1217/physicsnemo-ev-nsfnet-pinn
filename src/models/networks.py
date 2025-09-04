"""
神經網路模組 - FCNet全連接網路

基於ev-NSFnet/net.py，適配新的模組化架構
支援多種激活函數和靈活的網路配置
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Optional, Union, Type, Callable

class FCNet(nn.Module):
    """
    全連接神經網路
    
    支援靈活的層配置和多種激活函數
    """
    
    def __init__(
        self,
        num_inputs: int = 3,
        num_outputs: int = 3,
        num_layers: int = 6,
        hidden_size: int = 80,
        hidden_sizes: Optional[List[int]] = None,
        activation: Union[str, Type[nn.Module], Callable] = "tanh",
        initialization: str = "xavier"
    ):
        """
        初始化FCNet
        
        Args:
            num_inputs: 輸入維度
            num_outputs: 輸出維度  
            num_layers: 隱藏層數量
            hidden_size: 隱藏層神經元數量（當hidden_sizes為None時使用）
            hidden_sizes: 每層神經元數量列表，若提供則優先使用
            activation: 激活函數，支援字串或類別
            initialization: 權重初始化方法
        """
        super(FCNet, self).__init__()
        
        # 建立層尺寸列表
        if hidden_sizes is not None:
            if len(hidden_sizes) != num_layers:
                raise ValueError(f"hidden_sizes長度({len(hidden_sizes)})必須等於num_layers({num_layers})")
            layers = [num_inputs] + hidden_sizes + [num_outputs]
        else:
            layers = [num_inputs] + [hidden_size] * num_layers + [num_outputs]
        
        # 網路參數
        self.depth = len(layers) - 1
        self.layers_sizes = layers
        self.activation_name = activation if isinstance(activation, str) else activation.__name__
        self.initialization = initialization
        
        # 解析激活函數
        self.activation_fn = self._parse_activation(activation)
        
        # 建立網路層
        self.layers = self._build_layers(layers)
        
        # 初始化權重
        self._initialize_weights()
    
    def _parse_activation(self, activation: Union[str, Type[nn.Module], Callable]) -> Type[nn.Module]:
        """解析激活函數"""
        if isinstance(activation, str):
            activation_map = {
                "tanh": nn.Tanh,
                "relu": nn.ReLU,
                "sigmoid": nn.Sigmoid,
                "leaky_relu": nn.LeakyReLU,
                "elu": nn.ELU,
                "gelu": nn.GELU,
                "silu": nn.SiLU,
                "laaf": None  # LAAF將從activations模組導入
            }
            
            if activation.lower() == "laaf":
                # 延遲導入LAAF避免循環依賴
                print("警告: LAAF激活函數需要單獨導入，回退到Tanh")
                return nn.Tanh
            
            if activation.lower() in activation_map:
                return activation_map[activation.lower()]
            else:
                raise ValueError(f"不支援的激活函數: {activation}")
        
        elif callable(activation):
            return activation
        else:
            raise ValueError(f"激活函數必須是字串或可調用物件: {type(activation)}")
    
    def _build_layers(self, layer_sizes: List[int]) -> nn.Sequential:
        """建立網路層"""
        layer_list = []
        
        # 建立隱藏層
        for i in range(self.depth - 1):
            # 線性層
            linear_layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            layer_list.append((f'layer_{i}', linear_layer))
            
            # 激活函數
            try:
                # 嘗試不帶參數初始化
                activation_layer = self.activation_fn()
            except TypeError:
                try:
                    # 嘗試帶層尺寸參數初始化（用於LAAF等需要參數的激活函數）
                    activation_layer = self.activation_fn(layer_sizes[i + 1])
                except TypeError:
                    # 如果都失敗，回退到Tanh
                    print(f"警告: 激活函數初始化失敗，回退到Tanh")
                    activation_layer = nn.Tanh()
            
            layer_list.append((f'activation_{i}', activation_layer))
        
        # 輸出層（無激活函數）
        output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        layer_list.append((f'layer_{self.depth - 1}', output_layer))
        
        return nn.Sequential(OrderedDict(layer_list))
    
    def _initialize_weights(self) -> None:
        """
        權重初始化
        
        支援Xavier、He、Normal等初始化方法
        """
        if self.initialization.lower() == "xavier":
            # Xavier初始化，適合tanh激活函數
            gain = nn.init.calculate_gain('tanh')
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                    nn.init.zeros_(module.bias)
        
        elif self.initialization.lower() == "he":
            # He初始化，適合ReLU激活函數
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    nn.init.zeros_(module.bias)
        
        elif self.initialization.lower() == "normal":
            # 正態分布初始化
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.1)
                    nn.init.zeros_(module.bias)
        
        else:
            print(f"警告: 未知的初始化方法 {self.initialization}，使用預設初始化")
    
    def apply_layer_scaling(
        self, 
        first_layer_scale: float = 1.0, 
        last_layer_scale: float = 1.0
    ) -> None:
        """
        應用首末層縮放
        
        Args:
            first_layer_scale: 首層權重縮放係數
            last_layer_scale: 末層權重縮放係數
        """
        layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        
        if len(layers) > 0 and first_layer_scale != 1.0:
            # 縮放首層
            with torch.no_grad():
                layers[0].weight.data *= first_layer_scale
        
        if len(layers) > 1 and last_layer_scale != 1.0:
            # 縮放末層
            with torch.no_grad():
                layers[-1].weight.data *= last_layer_scale
    
    def get_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        獲取所有層的激活值（用於分析和調試）
        
        Args:
            x: 輸入張量
            
        Returns:
            List[torch.Tensor]: 各層激活值列表
        """
        activations = []
        current = x
        
        for i, layer in enumerate(self.layers):
            current = layer(current)
            activations.append(current.clone())
        
        return activations
    
    def count_parameters(self) -> int:
        """計算網路參數總數"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [batch_size, num_inputs]
            
        Returns:
            torch.Tensor: 輸出張量 [batch_size, num_outputs]
        """
        return self.layers(x)
    
    def __str__(self) -> str:
        """網路結構摘要"""
        param_count = self.count_parameters()
        return f"""
FCNet架構摘要:
- 層數: {self.depth} ({self.depth-1}隱藏層 + 1輸出層)
- 層尺寸: {' → '.join(map(str, self.layers_sizes))}
- 激活函數: {self.activation_name}
- 初始化方法: {self.initialization}
- 參數總數: {param_count:,}
"""


def create_pinn_networks(config) -> tuple[FCNet, FCNet]:
    """
    根據配置創建PINN主網路和EVM網路
    
    Args:
        config: 配置對象，包含network配置
        
    Returns:
        tuple[FCNet, FCNet]: (主網路, EVM網路)
    """
    # 主網路 (求解u, v, p)
    main_net = FCNet(
        num_inputs=3,  # (x, y, t)
        num_outputs=3,  # (u, v, p)
        num_layers=config.network.main_net_layers,
        hidden_size=config.network.main_net_hidden_size,
        activation=config.network.main_net_activation,
        initialization=config.network.main_net_initialization
    )
    
    # 應用首末層縮放
    main_net.apply_layer_scaling(
        config.network.main_net_first_layer_scale,
        config.network.main_net_last_layer_scale
    )
    
    # EVM網路 (計算entropy residual)
    evm_net = FCNet(
        num_inputs=3,  # (x, y, t)
        num_outputs=1,  # entropy residual
        num_layers=config.network.evm_net_layers,
        hidden_size=config.network.evm_net_hidden_size,
        activation=config.network.evm_net_activation,
        initialization=config.network.main_net_initialization
    )
    
    # 應用首末層縮放
    evm_net.apply_layer_scaling(
        config.network.evm_net_first_layer_scale,
        config.network.evm_net_last_layer_scale
    )
    
    print(f"✅ 網路創建完成:")
    print(f"📊 主網路: {main_net.count_parameters():,} 參數")
    print(f"📊 EVM網路: {evm_net.count_parameters():,} 參數")
    print(f"📊 總參數: {main_net.count_parameters() + evm_net.count_parameters():,}")
    
    return main_net, evm_net