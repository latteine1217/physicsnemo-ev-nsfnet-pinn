# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import os
import torch
import hydra
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.loggers import get_logger
from physicsnemo.trainer import Trainer
from physicsnemo.optimizers import get_optimizer
from physicsnemo.schedulers import get_scheduler

from physicsnemo_solver import PhysicsNeMoPINNSolver


class PhysicsNeMoPINNTrainer(Trainer):
    """
    PhysicsNeMo 標準 PINN 訓練器
    
    實作 6 階段漸進式訓練策略，用於 EV-NSFnet 模型
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        # 獲取日誌器
        self.logger = get_logger(__name__)
        
        # 初始化分散式環境
        self.dist = DistributedManager()
        
        # 初始化求解器
        self.solver = PhysicsNeMoPINNSolver(cfg)
        
        # 設定優化器和排程器
        self._setup_optimizer_and_scheduler()
        
        # 訓練狀態
        self.current_stage = 0
        self.stage_epoch = 0
        self.total_epoch = 0
        
        # 獲取訓練階段配置
        self.training_stages = cfg.training_stages
        
        self.logger.info(
            f"PhysicsNeMo PINN 訓練器初始化完成 - "
            f"總階段數: {len(self.training_stages)}"
        )
    
    def _setup_optimizer_and_scheduler(self):
        """設定優化器和學習率排程器"""
        
        # 使用 PhysicsNeMo 優化器
        self.optimizer = get_optimizer(
            model_parameters=self.solver.model.parameters(),
            optimizer_config=self.cfg.optimizer
        )
        
        # 使用 PhysicsNeMo 排程器
        if self.cfg.scheduler.enabled:
            self.scheduler = get_scheduler(
                optimizer=self.optimizer,
                scheduler_config=self.cfg.scheduler
            )
        else:
            self.scheduler = None
    
    def training_step(self, batch: dict) -> dict:
        """PhysicsNeMo 標準訓練步驟"""
        
        # 前向傳播並計算損失
        losses = self.solver.training_step(batch)
        
        # 反向傳播
        losses["total_loss"].backward()
        
        return losses
    
    def validation_step(self, batch: dict) -> dict:
        """PhysicsNeMo 標準驗證步驟"""
        
        losses = self.solver.validation_step(batch)
        
        return losses
    
    def configure_optimizers(self):
        """配置優化器 (PhysicsNeMo 標準介面)"""
        
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "total_loss"
                }
            }
        else:
            return {"optimizer": self.optimizer}
    
    def train_single_stage(self, stage_idx: int):
        """訓練單一階段"""
        
        stage_config = self.training_stages[stage_idx]
        stage_name = stage_config["stage_name"]
        epochs = stage_config["epochs"]
        alpha_evm = stage_config["alpha_evm"]
        learning_rate = stage_config["learning_rate"]
        
        self.logger.info(f"開始 {stage_name} - Alpha_EVM={alpha_evm}, LR={learning_rate}")
        
        # 設定階段參數
        self.solver.set_training_stage(stage_name, alpha_evm)
        
        # 更新學習率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # 獲取訓練資料
        interior_data = self.solver.dataset.get_interior_data()
        boundary_data = self.solver.dataset.get_boundary_data()
        
        # 階段訓練迴圈
        for epoch in range(epochs):
            self.solver.model.train()
            
            # 合併內部和邊界資料
            combined_data = {**interior_data, **boundary_data}
            
            # 訓練步驟
            self.optimizer.zero_grad()
            losses = self.training_step(combined_data)
            self.optimizer.step()
            
            # 更新全域計數器
            self.stage_epoch = epoch
            self.total_epoch += 1
            
            # 記錄損失
            if epoch % self.cfg.log_freq == 0 and self.dist.rank == 0:
                self.logger.info(
                    f"{stage_name} Epoch {epoch}/{epochs} - "
                    f"Total Loss: {losses['total_loss'].item():.6f}, "
                    f"NS Loss: {losses['ns_loss'].item():.6f}, "
                    f"EVM Loss: {losses['evm_loss'].item():.6f}, "
                    f"BC Loss: {losses['boundary_loss'].item():.6f}"
                )
            
            # 保存檢查點
            if epoch % self.cfg.checkpoint_freq == 0 and self.dist.rank == 0:
                self._save_checkpoint(stage_idx, epoch)
            
            # 學習率排程
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.logger.info(f"完成 {stage_name}")
    
    def train_all_stages(self):
        """執行完整的 6 階段訓練"""
        
        self.logger.info("開始 PhysicsNeMo EV-NSFnet 6 階段訓練")
        
        for stage_idx in range(len(self.training_stages)):
            
            # EVM 網路凍結/解凍策略 (每兩個階段切換)
            if stage_idx % 2 == 0:
                self.solver.freeze_evm_network()
            else:
                self.solver.unfreeze_evm_network()
            
            # 訓練當前階段
            self.train_single_stage(stage_idx)
            
            # 階段間驗證
            if hasattr(self.solver.dataset, 'reference_data') and self.solver.dataset.reference_data:
                self._validate_stage(stage_idx)
        
        self.logger.info("完成所有 6 階段訓練！")
    
    def _validate_stage(self, stage_idx: int):
        """階段驗證"""
        
        ref_data = self.solver.dataset.get_reference_data()
        
        if ref_data:
            errors = self.solver.evaluate(
                ref_data["x"], ref_data["y"],
                ref_data["u"], ref_data["v"], ref_data["p"]
            )
            
            if self.dist.rank == 0:
                stage_name = self.training_stages[stage_idx]["stage_name"]
                self.logger.info(
                    f"{stage_name} 驗證結果 - "
                    f"U 誤差: {errors['error_u']:.3f}%, "
                    f"V 誤差: {errors['error_v']:.3f}%, "
                    f"P 誤差: {errors['error_p']:.3f}%"
                )
    
    def _save_checkpoint(self, stage_idx: int, epoch: int):
        """保存檢查點"""
        
        checkpoint_path = os.path.join(
            self.cfg.checkpoint_dir,
            f"model_stage_{stage_idx}_epoch_{epoch}.pth"
        )
        
        checkpoint = {
            'model_state_dict': self.solver.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stage_idx': stage_idx,
            'epoch': epoch,
            'total_epoch': self.total_epoch,
            'cfg': self.cfg
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"檢查點已保存: {checkpoint_path}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """PhysicsNeMo PINN 主訓練函數"""
    
    # 初始化分散式環境
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # 獲取日誌器
    logger = get_logger(__name__)
    
    if dist.rank == 0:
        logger.info("🌊 啟動 PhysicsNeMo EV-NSFnet PINN 訓練")
        logger.info(f"雷諾數: {cfg.reynolds_number}")
        logger.info(f"GPU 數量: {dist.world_size}")
    
    # 創建輸出目錄
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    try:
        # 初始化訓練器
        trainer = PhysicsNeMoPINNTrainer(cfg)
        
        # 執行完整訓練
        trainer.train_all_stages()
        
        if dist.rank == 0:
            logger.info("🎉 PhysicsNeMo EV-NSFnet PINN 訓練完成！")
            
    except Exception as e:
        logger.error(f"訓練過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    main()