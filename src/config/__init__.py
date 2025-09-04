"""
配置管理模組

提供統一的配置管理功能，支援YAML配置檔案載入、多階段訓練參數管理等。
"""

from .config_manager import (
    ConfigManager, 
    NetworkConfig, 
    PhysicsConfig, 
    TrainingConfig, 
    TrainingStage,
    LBFGSConfig,
    SupervisionConfig,
    SystemConfig,
    ExperimentConfig
)

__all__ = [
    "ConfigManager",
    "NetworkConfig", 
    "PhysicsConfig", 
    "TrainingConfig", 
    "TrainingStage",
    "LBFGSConfig",
    "SupervisionConfig",
    "SystemConfig",
    "ExperimentConfig"
]