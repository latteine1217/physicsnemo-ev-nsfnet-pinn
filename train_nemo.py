#!/usr/bin/env python3
"""
LDC-PINNs: Physics-Informed Neural Networks for Lid-Driven Cavity Flow
基於 NVIDIA PhysicsNeMo 框架實現

This module implements a Physics-Informed Neural Network (PINN) solution for the 
lid-driven cavity flow problem using PhysicsNeMo framework with symbolic PDE definitions.
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

# 添加當前目錄到 Python path  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ldc_pinn import LidDrivenCavityPINN
from data.cavity_datamodule import CavityDataModule
from utils.physics_loss import PhysicsLoss
from utils.logger import setup_logger


@hydra.main(version_base=None, config_path="configs", config_name="ldc_pinn")
def main(cfg: DictConfig) -> None:
    """
    主訓練函數 - 使用PhysicsNeMo框架訓練lid-driven cavity PINN模型
    
    Args:
        cfg: Hydra配置對象
    """
    # 初始化分布式管理器 (PhysicsNeMo的分布式功能)
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # 設置日誌
    logger = setup_logger(cfg.logging)
    
    if dist.rank == 0:
        logger.info("🚀 開始PhysicsNeMo LDC-PINN訓練")
        logger.info(f"📊 使用設備: {dist.device}")
        logger.info(f"🔧 分布式設置: {dist.world_size} GPUs")
    
    # 設置CUDA優化 (Tesla P100相容性)
    if cfg.optimization.torch_compile and hasattr(torch, 'compile'):
        try:
            # 檢查CUDA capability
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability(dist.local_rank)
                if capability[0] < 7:  # P100是6.0
                    logger.warning("⚠️  Tesla P100檢測到，禁用torch.compile")
                    cfg.optimization.torch_compile = False
        except Exception as e:
            logger.warning(f"⚠️  無法檢測CUDA capability: {e}")
            cfg.optimization.torch_compile = False
    
    # 初始化資料模組
    datamodule = CavityDataModule(
        cfg=cfg.data,
        Re=cfg.physics.Re,
        device=dist.device
    )
    
    # 設置資料
    datamodule.setup()
    
    # 初始化PINN模型 (使用PhysicsNeMo架構)
    model = LidDrivenCavityPINN(
        cfg=cfg.model,
        physics_cfg=cfg.physics,
        device=dist.device
    )
    
    # 設置物理損失計算
    physics_loss = PhysicsLoss(
        cfg=cfg.physics,
        Re=cfg.physics.Re,
        device=dist.device
    )
    
    # 分布式模型包裝
    if dist.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )
    
    # 優化器設置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    # 執行訓練循環
    from trainers.physics_trainer import PhysicsTrainer
    
    trainer = PhysicsTrainer(
        model=model,
        optimizer=optimizer,
        physics_loss=physics_loss,
        datamodule=datamodule,
        cfg=cfg.training,
        logger=logger,
        device=dist.device
    )
    
    # 開始訓練
    trainer.fit()
    
    if dist.rank == 0:
        logger.info("🎉 訓練完成!")


if __name__ == "__main__":
    main()